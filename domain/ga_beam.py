"""Algoritmo Genetico para diseno de refuerzo de flexion en vigas (zona unica).

Dado (b, h, fc, Mu), encuentra la configuracion optima de barras que:
  - Minimiza el peso de acero (kg/m)
  - Cumple E.060 (phi_Mn >= Mu, rho_min <= rho <= rho_max, epsilon_t >= 0.005)
  - Respeta restricciones constructivas (VARILLAS_POR_ANCHO, max 2 capas)

Cromosoma: 5 enteros identicos al espacio de accion del DRL Modelo 3.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mejora del modelo'))

from config.config import ECU, FY, PHI_FLEXION, R1, R2, REBAR_CATALOG, VARILLAS_POR_ANCHO

# ---------------------------------------------------------------------------
# Constantes adicionales (no estan en config.py)
# ---------------------------------------------------------------------------

# Peso por metro lineal de cada diametro (kg/m)
# Indices: 0=1/2", 1=5/8", 2=3/4", 3=1"
REBAR_WEIGHTS_KGM = {0: 0.994, 1: 1.552, 2: 2.235, 3: 3.973}

# Mapeo de configuracion de diametros (igual que DRL)
# Indice -> (indice_diam_principal, indice_diam_secundario o None)
DIAM_CONFIGS = {
    0: (0, None),  # solo 1/2"
    1: (1, None),  # solo 5/8"
    2: (2, None),  # solo 3/4"
    3: (3, None),  # solo 1"
    4: (0, 1),     # 1/2" + 5/8"
    5: (1, 2),     # 5/8" + 3/4"
    6: (2, 3),     # 3/4" + 1"
}

# Si Mu <= este umbral, se retorna acero minimo directamente (sin ejecutar el GA)
MIN_STEEL_THRESHOLD = 0.5  # tonf-m

# Penalizaciones en la funcion de fitness
LAMBDA_M = 1000.0   # capacidad insuficiente (phi_Mn < Mu)
LAMBDA_N = 100.0    # violaciones normativas (rho, epsilon_t)
LAMBDA_G = 10.0     # violaciones constructivas

PENALTY_CONSTRUCTIVE = 50.0  # penalizacion fija por restriccion constructiva


# ---------------------------------------------------------------------------
# Funciones E.060 (mismas formulas que beam_env_quadratic_layers.py)
# ---------------------------------------------------------------------------

def _beta1(fc: float) -> float:
    if fc <= 280:
        return 0.85
    elif fc <= 560:
        return round(1.05 - 0.714 * (fc / 1000), 3)
    return 0.65


def _rho_limits(fc: float, beta1: float) -> tuple:
    rho_min = 0.7 * math.sqrt(fc) / FY  # E.060 Art.10.5, secciones rectangulares
    rho_max = 0.75 * 0.85 * beta1 * (fc / FY) * (6000.0 / (6000.0 + FY))
    return rho_min, rho_max


def _compute_phi_mn(as_total: float, b_cm: float, d_cm: float, fc: float) -> tuple:
    """Calcula phi_Mn (tonf-m), epsilon_t, a, c. Retorna (phi_mn, epsilon_t, a, c)."""
    if as_total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    beta1 = _beta1(fc)
    a = as_total * FY / (0.85 * fc * b_cm)
    c = a / beta1
    epsilon_t = ECU * (d_cm - c) / c if c > 0 else 0.0
    mn = as_total * FY * (d_cm - a / 2.0) / 100_000.0
    return PHI_FLEXION * mn, epsilon_t, a, c


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class BeamGA:
    """Solver de zona unica por Algoritmo Genetico.

    Uso:
        ga = BeamGA(b=0.30, h=0.50, fc=210, mu=15.0, seed=42)
        result = ga.run()
    """

    def __init__(
        self,
        b: float,
        h: float,
        fc: float,
        mu: float,
        pop_size: int = 100,
        n_gen: int = 120,
        tournament_k: int = 4,
        p_cross: float = 0.8,
        p_mut: float = 0.15,
        seed: int = None,
    ):
        self.b = round(b, 2)
        self.h = h
        self.fc = fc
        self.mu = mu  # tonf-m, ya es valor absoluto

        self.pop_size = pop_size
        self.n_gen = n_gen
        self.tournament_k = tournament_k
        self.p_cross = p_cross
        self.p_mut = p_mut

        self.rng = np.random.RandomState(seed)

        # Limites constructivos segun ancho
        if self.b in VARILLAS_POR_ANCHO:
            self.min_v, self.max_v = VARILLAS_POR_ANCHO[self.b]
        else:
            self.min_v, self.max_v = 2, 2  # fallback conservador

        # Rango de genes: [diam_config 0-6, n1_c1 0-max_v, n2_c1 0-max_v, n1_c2 0-max_v, n2_c2 0-max_v]
        self.gene_max = np.array([6, self.max_v, self.max_v, self.max_v, self.max_v], dtype=int)

    # -----------------------------------------------------------------------
    # Codificacion / decodificacion
    # -----------------------------------------------------------------------

    def _decode(self, x: np.ndarray) -> tuple:
        """Convierte cromosoma de 5 genes en (capa1[4], capa2[4])."""
        d1, d2 = DIAM_CONFIGS[int(x[0])]
        capa1 = np.zeros(4, dtype=int)
        capa2 = np.zeros(4, dtype=int)
        capa1[d1] = int(x[1])
        if d2 is not None:
            capa1[d2] = int(x[2])
        capa2[d1] = int(x[3])
        if d2 is not None:
            capa2[d2] = int(x[4])
        return capa1, capa2

    # -----------------------------------------------------------------------
    # Calculo estructural
    # -----------------------------------------------------------------------

    def _compute_weight(self, capa1: np.ndarray, capa2: np.ndarray) -> float:
        """Calcula peso total de acero en kg/m."""
        w = 0.0
        for i in range(4):
            w += (int(capa1[i]) + int(capa2[i])) * REBAR_WEIGHTS_KGM[i]
        return w

    def _evaluate(self, capa1: np.ndarray, capa2: np.ndarray) -> dict:
        """Evaluacion E.060 completa. Retorna dict con todos los resultados."""
        n_c1 = int(np.sum(capa1))
        n_c2 = int(np.sum(capa2))

        b_cm = self.b * 100.0
        has_c2 = n_c2 > 0
        d = self.h - R2 if has_c2 else self.h - R1
        d_cm = d * 100.0

        as_total = sum(
            (int(capa1[i]) + int(capa2[i])) * REBAR_CATALOG[i]['area_cm2']
            for i in range(4)
        )

        beta1 = _beta1(self.fc)
        rho_min, rho_max = _rho_limits(self.fc, beta1)

        if as_total > 0 and d_cm > 0:
            rho = as_total / (b_cm * d_cm)
            phi_mn, epsilon_t, a, c = _compute_phi_mn(as_total, b_cm, d_cm, self.fc)
        else:
            rho, phi_mn, epsilon_t, a, c = 0.0, 0.0, 0.0, 0.0, 0.0

        exceso_pct = (phi_mn - self.mu) / self.mu * 100.0 if self.mu > 0 else 0.0
        cumple = (
            phi_mn >= self.mu
            and rho >= rho_min
            and rho <= rho_max
            and epsilon_t >= 0.005
        )

        return {
            'phi_mn': round(phi_mn, 4),
            'as_total': round(as_total, 4),
            'rho': round(rho, 6),
            'rho_min': round(rho_min, 6),
            'rho_max': round(rho_max, 6),
            'epsilon_t': round(epsilon_t, 6),
            'exceso_pct': round(exceso_pct, 2),
            'cumple': cumple,
            'n_c1': n_c1,
            'n_c2': n_c2,
            'd_cm': round(d_cm, 4),
        }

    # -----------------------------------------------------------------------
    # Restricciones constructivas (mismas 5 del DRL Modelo 3)
    # -----------------------------------------------------------------------

    def _geometric_penalty(self, capa1: np.ndarray, capa2: np.ndarray) -> float:
        """Penalizacion por violaciones constructivas (0.0 si no hay violaciones)."""
        n_c1 = int(np.sum(capa1))
        n_c2 = int(np.sum(capa2))
        penalty = 0.0

        # 1. Capa 1 debe tener al menos 2 varillas
        if n_c1 < 2:
            penalty += PENALTY_CONSTRUCTIVE

        # 2. Capa 1 dentro de rango segun ancho
        if n_c1 < self.min_v or n_c1 > self.max_v:
            penalty += PENALTY_CONSTRUCTIVE

        # 3. Capa 2 no puede exceder max_v
        if n_c2 > self.max_v:
            penalty += PENALTY_CONSTRUCTIVE

        # 4. Al menos un diametro con >= 2 varillas totales
        has_min_2 = any(int(capa1[i]) + int(capa2[i]) >= 2 for i in range(4))
        if not has_min_2:
            penalty += PENALTY_CONSTRUCTIVE

        # 5. Capa 2 solo si capa 1 esta casi llena (>= max_v - 1)
        if n_c2 > 0 and n_c1 < self.max_v - 1:
            penalty += PENALTY_CONSTRUCTIVE

        return penalty

    # -----------------------------------------------------------------------
    # Funcion de fitness (minimizar)
    # -----------------------------------------------------------------------

    def _fitness(self, x: np.ndarray) -> float:
        capa1, capa2 = self._decode(x)
        evr = self._evaluate(capa1, capa2)
        w = self._compute_weight(capa1, capa2)
        g_pen = self._geometric_penalty(capa1, capa2)

        # Penalizacion por capacidad insuficiente
        if self.mu > 0 and evr['phi_mn'] < self.mu:
            p_m = (self.mu - evr['phi_mn']) / self.mu * 100.0
        else:
            p_m = 0.0

        # Penalizacion por cuantias fuera de rango y ductilidad
        p_n = 0.0
        if evr['rho'] < evr['rho_min']:
            p_n += (evr['rho_min'] - evr['rho']) / evr['rho_min'] * 100.0
        if evr['rho'] > evr['rho_max']:
            p_n += (evr['rho'] - evr['rho_max']) / evr['rho_max'] * 100.0
        if evr['epsilon_t'] < 0.005:
            p_n += (0.005 - evr['epsilon_t']) / 0.005 * 100.0

        return w + LAMBDA_M * p_m + LAMBDA_N * p_n + LAMBDA_G * g_pen

    # -----------------------------------------------------------------------
    # Operadores geneticos
    # -----------------------------------------------------------------------

    def _random_individual(self) -> np.ndarray:
        x = np.zeros(5, dtype=int)
        x[0] = self.rng.randint(0, 7)       # diam_config
        for j in range(1, 5):
            x[j] = self.rng.randint(0, self.max_v + 1)
        return self._repair(x)

    def _repair(self, x: np.ndarray) -> np.ndarray:
        """Clip a rangos validos y asegura minimo en capa 1."""
        x = x.copy()
        x[0] = int(np.clip(x[0], 0, 6))
        for j in range(1, 5):
            x[j] = int(np.clip(x[j], 0, self.max_v))
        # Asegurar al menos min_v varillas en capa 1
        d1, d2 = DIAM_CONFIGS[int(x[0])]
        n_c1 = int(x[1]) + (int(x[2]) if d2 is not None else 0)
        if n_c1 < self.min_v:
            deficit = self.min_v - n_c1
            x[1] = int(np.clip(x[1] + deficit, 0, self.max_v))
        return x

    def _tournament_select(self, pop: np.ndarray, fits: np.ndarray) -> np.ndarray:
        """Seleccion por torneo: retorna el individuo con menor fitness."""
        k = min(self.tournament_k, len(pop))
        idxs = self.rng.choice(len(pop), size=k, replace=False)
        best_idx = idxs[np.argmin(fits[idxs])]
        return pop[best_idx].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple:
        """Crossover de punto simple sobre los genes de barras (1-4)."""
        if self.rng.random() > self.p_cross:
            return p1.copy(), p2.copy()
        c1, c2 = p1.copy(), p2.copy()
        # Elegir punto de corte en [1, 4]
        pt = self.rng.randint(1, 5)
        c1[pt:] = p2[pt:]
        c2[pt:] = p1[pt:]
        return self._repair(c1), self._repair(c2)

    def _mutate(self, x: np.ndarray) -> np.ndarray:
        """Mutacion gen a gen con Δ ∈ {-2, -1, +1, +2}."""
        x = x.copy()
        deltas = [-2, -1, 1, 2]
        for j in range(5):
            if self.rng.random() < self.p_mut:
                delta = self.rng.choice(deltas)
                x[j] += delta
        return self._repair(x)

    # -----------------------------------------------------------------------
    # Caso especial: acero minimo
    # -----------------------------------------------------------------------

    def _min_steel_design(self) -> dict:
        """Calcula la configuracion mas liviana que satisface rho_min.
        Usado cuando Mu <= MIN_STEEL_THRESHOLD.
        """
        b_cm = self.b * 100.0
        d_cm = (self.h - R1) * 100.0
        beta1 = _beta1(self.fc)
        rho_min, _ = _rho_limits(self.fc, beta1)
        as_min = rho_min * b_cm * d_cm

        best_x = None
        best_w = float('inf')

        # Busqueda en el espacio de DIAM_CONFIGS con 1 capa (c2 = 0)
        for dc in range(7):
            d1, d2 = DIAM_CONFIGS[dc]
            area1 = REBAR_CATALOG[d1]['area_cm2']
            w1 = REBAR_WEIGHTS_KGM[d1]
            area2 = REBAR_CATALOG[d2]['area_cm2'] if d2 is not None else 0.0
            w2 = REBAR_WEIGHTS_KGM[d2] if d2 is not None else 0.0

            for n1 in range(self.min_v, self.max_v + 1):
                for n2 in range(0, (self.max_v + 1) if d2 is not None else 1):
                    as_total = n1 * area1 + n2 * area2
                    if as_total < as_min:
                        continue
                    w = n1 * w1 + n2 * w2
                    if w < best_w:
                        best_w = w
                        best_x = np.array([dc, n1, n2, 0, 0], dtype=int)

        if best_x is None:
            # Fallback: 2 barras de 1/2"
            best_x = np.array([0, 2, 0, 0, 0], dtype=int)

        capa1, capa2 = self._decode(best_x)
        evr = self._evaluate(capa1, capa2)
        w = self._compute_weight(capa1, capa2)

        return {
            'fitness': w,
            'weight_kgm': round(w, 4),
            'phi_mn': evr['phi_mn'],
            'cumple': evr['cumple'],
            'exceso_pct': evr['exceso_pct'],
            'as_total': evr['as_total'],
            'rho': evr['rho'],
            'epsilon_t': evr['epsilon_t'],
            'design_str': self._build_design_str(capa1, capa2),
            'is_min_steel': True,
            'generations': 0,
            'best_x': best_x.tolist(),
        }

    def _build_design_str(self, capa1: np.ndarray, capa2: np.ndarray) -> str:
        parts_c1 = [
            f'{int(capa1[i])}O{REBAR_CATALOG[i]["name"]}'
            for i in range(4) if int(capa1[i]) > 0
        ]
        parts_c2 = [
            f'{int(capa2[i])}O{REBAR_CATALOG[i]["name"]}'
            for i in range(4) if int(capa2[i]) > 0
        ]
        c1 = 'C1: ' + ' + '.join(parts_c1) if parts_c1 else 'C1: vacia'
        c2 = 'C2: ' + ' + '.join(parts_c2) if parts_c2 else 'C2: vacia'
        return f'{c1} | {c2}'

    # -----------------------------------------------------------------------
    # Bucle principal del GA
    # -----------------------------------------------------------------------

    def run(self) -> dict:
        """Ejecuta el GA y retorna el mejor diseno encontrado."""
        # Caso especial: acero minimo
        if self.mu <= MIN_STEEL_THRESHOLD:
            return self._min_steel_design()

        # Inicializar poblacion
        pop = np.array([self._random_individual() for _ in range(self.pop_size)])
        fits = np.array([self._fitness(x) for x in pop])

        n_elite = 3
        best_x = pop[np.argmin(fits)].copy()
        best_fit = fits.min()

        for gen in range(self.n_gen):
            new_pop = []

            # Elitismo: preservar los n_elite mejores
            elite_idxs = np.argsort(fits)[:n_elite]
            for idx in elite_idxs:
                new_pop.append(pop[idx].copy())

            # Generar el resto por seleccion + crossover + mutacion
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fits)
                p2 = self._tournament_select(pop, fits)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop = np.array(new_pop)
            fits = np.array([self._fitness(x) for x in pop])
            # fits_3bests = np.argsort(fits)[:3]
            # print(fits_3bests)
            gen_best_fit = fits.min()
            if gen_best_fit < best_fit:
                # print(f"Gen {gen+1}/{self.n_gen} - Mejor fitness: {gen_best_fit:.4f}")
                best_fit = gen_best_fit
                best_x = pop[np.argmin(fits)].copy()
            
        # Construir resultado final
        capa1, capa2 = self._decode(best_x)
        evr = self._evaluate(capa1, capa2)
        w = self._compute_weight(capa1, capa2)

        return {
            'fitness': round(best_fit, 4),
            'weight_kgm': round(w, 4),
            'phi_mn': evr['phi_mn'],
            'cumple': evr['cumple'],
            'exceso_pct': evr['exceso_pct'],
            'as_total': evr['as_total'],
            'rho': evr['rho'],
            'epsilon_t': evr['epsilon_t'],
            'design_str': self._build_design_str(capa1, capa2),
            'is_min_steel': False,
            'generations': self.n_gen,
            'best_x': best_x.tolist(),
        }


# ---------------------------------------------------------------------------
# Test rapido
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    print('=== Test BeamGA ===\n')

    casos = [
        {'b': 0.30, 'h': 0.50, 'fc': 210, 'mu': 15.0, 'label': 'Normal (Mu=15 tonf-m)'},
        {'b': 0.30, 'h': 0.50, 'fc': 280, 'mu': 40.0, 'label': 'Alto (Mu=40 tonf-m)'},
        {'b': 0.20, 'h': 0.35, 'fc': 210, 'mu': 0.0,  'label': 'Mu=0 (acero minimo)'},
        {'b': 0.25, 'h': 0.40, 'fc': 350, 'mu': 0.3,  'label': 'Mu pequeno (<umbral)'},
        {'b': 0.50, 'h': 0.70, 'fc': 350, 'mu': 80.0, 'label': 'Viga grande (Mu=80)'},
    ]

    for c in casos:
        t0 = time.time()
        ga = BeamGA(b=c['b'], h=c['h'], fc=c['fc'], mu=c['mu'], seed=42)
        r = ga.run()
        dt = time.time() - t0
        flag = ' [MIN]' if r['is_min_steel'] else ''
        ok = 'OK' if r['cumple'] else 'FAIL'
        print(
            f"[{ok}]{flag} {c['label']}\n"
            f"       Design: {r['design_str']}\n"
            f"       phi_Mn={r['phi_mn']:.2f} | exceso={r['exceso_pct']:.1f}% | "
            f"w={r['weight_kgm']:.3f} kg/m | {dt:.2f}s\n"
        )
