"""BeamGANSpans: GA para diseno de corrido + bastones en vigas de N tramos.

Optimiza simultaneamente:
  - 2 diametros globales (diam_A <= diam_B)
  - Corrido TOP y BOT (opcionalmente simetricos)
  - Bastones en 6*n_spans zonas (LEFT/MID/RIGHT x TOP/BOT por tramo)

Retorna top-3 disenos (feasible primero, luego menor peso).
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'GA_viga_completa'))
sys.path.insert(0, os.path.join(_HERE, '..', 'mejora del modelo'))
sys.path.insert(0, os.path.join(_HERE, '..', 'GA beam'))

from config import REBAR_CATALOG, VARILLAS_POR_ANCHO, R1
from chromosome import n_slots_for_beam, block_size

from config_n import (
    N_CAPAS_MAX,
    generate_zone_ids,
    parse_zone_id,
    scaled_ga_params,
    CORRIDO_SIMETRICO,
    TOURNAMENT_K, N_ELITE,
    P_CROSS, P_MUT_ONI, P_MUT_CHOICE, P_MUT_DIAM, P_MUT_RESET,
    HOF_POOL_SIZE, RESTART_PAT, RESTART_RATIO,
)
from chromosome_n import (
    chrom_length_n, decode_n, encode_n, repair_n,
)
from fitness_n import evaluate_and_fitness_n


class BeamGANSpans:
    def __init__(self, beam: dict,
                 pop_size: int = None,
                 n_gen: int = None,
                 tournament_k: int = TOURNAMENT_K,
                 seed: int = None,
                 corrido_simetrico: bool = CORRIDO_SIMETRICO):
        self.beam = beam
        self.n_spans = beam['n_spans']
        self.corrido_simetrico = corrido_simetrico
        self.tournament_k = tournament_k

        # Auto-escalar si no se proveen
        params = scaled_ga_params(self.n_spans)
        self.pop_size = pop_size if pop_size is not None else params['pop_size']
        self.n_gen = n_gen if n_gen is not None else params['n_gen']
        self.early_stop_pat = params['early_stop_pat']

        if seed is not None:
            np.random.seed(seed)

        inp = beam['inputs']
        self.b = round(float(inp['b_m']), 2)
        self.h = float(inp['h_m'])
        self.fc = float(inp['fc_kg_cm2'])

        self.b_cm = self.b * 100.0
        self.h_cm = self.h * 100.0
        self.d_cm = (self.h - R1) * 100.0

        self.min_v, self.max_v = VARILLAS_POR_ANCHO[self.b]
        self.n_slots = n_slots_for_beam(self.b)
        self.BLOCK = block_size(self.n_slots)
        self.chrom_L = chrom_length_n(self.n_slots, self.n_spans)

        self.zone_ids = generate_zone_ids(self.n_spans)
        self.n_zones = len(self.zone_ids)

        # Demandas aproximadas por zona (para inicializacion)
        self.mu_zones = self._compute_mu_zones()

    # -------------------------------------------------------------------
    # Demandas aproximadas
    # -------------------------------------------------------------------

    def _compute_mu_zones(self) -> dict:
        xs = np.array(self.beam['outputs']['x_m'], dtype=float)
        outs = self.beam['outputs']
        ms_max = np.array(
            outs.get('M_tonf_m_max', outs.get('M_tonf_m', [])), dtype=float)
        ms_min = np.array(
            outs.get('M_tonf_m_min', outs.get('M_tonf_m', [])), dtype=float)

        mu = {}
        for span_i, sp in enumerate(self.beam['spans']):
            x0, x1 = sp['x0'], sp['x1']
            L3 = (x1 - x0) / 3.0
            tercios = [
                ('LEFT', x0, x0 + L3),
                ('MID', x0 + L3, x0 + 2 * L3),
                ('RIGHT', x0 + 2 * L3, x1),
            ]
            mask = (xs >= x0 - 1e-6) & (xs <= x1 + 1e-6)

            for pos, xl, xh in tercios:
                tmask = mask & (xs >= xl - 1e-6) & (xs <= xh + 1e-6)
                for face in ('TOP', 'BOT'):
                    zid = f'{pos}_{face}_T{span_i + 1}'
                    ms_face = ms_max[tmask] if face == 'BOT' else ms_min[tmask]
                    if face == 'BOT':
                        vals = ms_face[ms_face > 0]
                        mu[zid] = float(vals.max()) if len(vals) > 0 else 0.0
                    else:
                        vals = np.abs(ms_face[ms_face < 0])
                        mu[zid] = float(vals.max()) if len(vals) > 0 else 0.0
        return mu

    # -------------------------------------------------------------------
    # Inicializacion
    # -------------------------------------------------------------------

    def _init_population(self) -> np.ndarray:
        pop = np.zeros((self.pop_size, self.chrom_L), dtype=np.int8)
        idx = 0

        # Arquetipo A (30%): corrido minimo, bastones OFF
        n_A = max(1, int(self.pop_size * 0.30))
        for _ in range(n_A):
            if idx >= self.pop_size:
                break
            pop[idx] = self._repair(self._archetype_A())
            idx += 1

        # Arquetipo B (40%): corrido minimo + bastones informados
        n_B = max(1, int(self.pop_size * 0.40))
        for _ in range(n_B):
            if idx >= self.pop_size:
                break
            pop[idx] = self._repair(self._archetype_B())
            idx += 1

        # Arquetipo C: aleatorio
        while idx < self.pop_size:
            pop[idx] = self._repair(self._archetype_C())
            idx += 1

        return pop

    def _make_base(self, dA: int = 1, dB: int = 2) -> np.ndarray:
        z = np.zeros(self.chrom_L, dtype=np.int8)
        z[0] = dA
        z[1] = dB
        return z

    def _activate_corners(self, z: np.ndarray, block_offset: int,
                          diam_choice: int = 0):
        z[block_offset + 0] = diam_choice
        z[block_offset + 1] = 1
        last = block_offset + (self.n_slots - 1) * 2
        z[last] = diam_choice
        z[last + 1] = 1

    def _repair(self, z: np.ndarray) -> np.ndarray:
        return repair_n(z, self.n_slots, self.n_spans,
                        self.min_v, self.max_v,
                        self.corrido_simetrico,
                        beam=self.beam)

    def _archetype_A(self) -> np.ndarray:
        dA = np.random.randint(0, 4)
        dB = np.random.randint(dA, 4)
        z = self._make_base(dA, dB)
        # Corrido TOP capa_0
        self._activate_corners(z, 2, diam_choice=0)
        # Corrido BOT capa_0
        self._activate_corners(z, 2 + self.BLOCK, diam_choice=0)
        return z

    def _archetype_B(self) -> np.ndarray:
        dA = np.random.randint(0, 4)
        dB = np.random.randint(dA, 4)
        z = self._make_base(dA, dB)
        self._activate_corners(z, 2, diam_choice=0)
        self._activate_corners(z, 2 + self.BLOCK, diam_choice=0)

        as_corr = 2 * REBAR_CATALOG[dA]['area_cm2']

        for zone_ord, zone_id in enumerate(self.zone_ids):
            mu = self.mu_zones.get(zone_id, 0.0)
            if mu <= 0.01:
                continue
            mn_approx = as_corr * 4200 * self.d_cm / 100_000
            if mn_approx >= mu:
                continue
            n_extra = min(2, self.n_slots)
            bast_offset = 2 + 2 * self.BLOCK + zone_ord * self.BLOCK
            mid = self.n_slots // 2
            for j in range(n_extra):
                slot = mid + j
                if slot < self.n_slots:
                    base = bast_offset + slot * 2
                    z[base] = 1      # diam_B
                    z[base + 1] = 1  # ON
        return z

    def _archetype_C(self) -> np.ndarray:
        dA = np.random.randint(0, 4)
        dB = np.random.randint(dA, 4)
        z = self._make_base(dA, dB)
        for offset in range(2, self.chrom_L, 2):
            z[offset] = np.random.randint(0, 2)
            z[offset + 1] = 1 if np.random.random() < 0.25 else 0
        return z

    # -------------------------------------------------------------------
    # Seleccion
    # -------------------------------------------------------------------

    def _tournament(self, fits: np.ndarray) -> int:
        candidates = np.random.choice(len(fits), self.tournament_k,
                                      replace=False)
        return int(candidates[np.argmin(fits[candidates])])

    # -------------------------------------------------------------------
    # Crossover
    # -------------------------------------------------------------------

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple:
        c1, c2 = p1.copy(), p2.copy()
        if np.random.random() > P_CROSS:
            return c1, c2

        # Bloques: corrido_top + corrido_bot + 6*n_spans bastones
        n_blocks = 2 + self.n_zones
        for bi in range(n_blocks):
            if np.random.random() < 0.5:
                off = 2 + bi * self.BLOCK
                end = off + self.BLOCK
                c1[off:end], c2[off:end] = c2[off:end].copy(), c1[off:end].copy()
                if np.random.random() < 0.3:
                    pt = np.random.randint(1, self.BLOCK)
                    c1[off:off+pt], c2[off:off+pt] = (
                        c2[off:off+pt].copy(), c1[off:off+pt].copy())
        return c1, c2

    # -------------------------------------------------------------------
    # Mutacion
    # -------------------------------------------------------------------

    def _mutate(self, z: np.ndarray) -> np.ndarray:
        z = z.copy()

        if np.random.random() < P_MUT_DIAM:
            which = np.random.randint(0, 2)
            delta = np.random.choice([-1, 1])
            z[which] = int(np.clip(z[which] + delta, 0, 3))
            if z[0] > z[1]:
                z[0], z[1] = z[1], z[0]

        n_blocks = 2 + self.n_zones
        offset = 2
        for bi in range(n_blocks):
            # Reset solo bloques baston (bi >= 2)
            if bi >= 2 and np.random.random() < P_MUT_RESET:
                z[offset: offset + self.BLOCK] = 0
                offset += self.BLOCK
                continue

            for gi in range(self.BLOCK // 2):
                gene_off = offset + gi * 2
                if np.random.random() < P_MUT_ONI:
                    z[gene_off + 1] = 1 - z[gene_off + 1]
                if z[gene_off + 1] == 1 and np.random.random() < P_MUT_CHOICE:
                    z[gene_off] = 1 - z[gene_off]
            offset += self.BLOCK

        return z

    # -------------------------------------------------------------------
    # Hall of Fame
    # -------------------------------------------------------------------

    @staticmethod
    def _corrido_signature(ev: dict) -> tuple:
        n_bars_top = len(ev.get('corr_bars_top', []))
        n_bars_bot = len(ev.get('corr_bars_bot', []))
        dec = ev['decoded']
        return (dec['diam_A'], dec['diam_B'], n_bars_top, n_bars_bot)

    def _update_hof(self, hof: list, ev: dict, fit: float, z: np.ndarray):
        sig = self._corrido_signature(ev)
        entry = {
            'eval': ev,
            'fitness': fit,
            'z': z.copy(),
            'weight': ev['total_weight_kg'],
            'feasible': ev['feasible'],
            'sig': sig,
        }

        def _key(e):
            return (0 if e['feasible'] else 1, e['weight'])

        for i, existing in enumerate(hof):
            if existing['sig'] == sig:
                dominates = (
                    (entry['feasible'] and not existing['feasible'])
                    or (entry['feasible'] == existing['feasible']
                        and entry['weight'] < existing['weight'])
                )
                if dominates:
                    hof[i] = entry
                    hof.sort(key=_key)
                return

        hof.append(entry)
        hof.sort(key=_key)
        hof[:] = hof[:HOF_POOL_SIZE]

    # -------------------------------------------------------------------
    # Loop principal
    # -------------------------------------------------------------------

    def run(self) -> dict:
        pop = self._init_population()
        fits = np.full(self.pop_size, float('inf'))
        evals = [None] * self.pop_size

        hof = []
        history_best = []
        history_mean = []
        history_worst = []

        no_improve_count = 0
        best_ever = float('inf')
        early_stop_gen = None
        restart_count = 0

        t0 = time.time()
        PRINT_EVERY = max(1, self.n_gen // 20)

        for gen in range(self.n_gen):
            for i in range(self.pop_size):
                ev, fit = evaluate_and_fitness_n(
                    pop[i], self.beam, self.n_slots, self.n_spans)
                fits[i] = fit
                evals[i] = ev
                pop[i] = ev['z_repaired']
                self._update_hof(hof, ev, fit, pop[i])

            best_fit = float(fits.min())
            mean_fit = float(fits.mean())
            worst_fit = float(fits.max())
            history_best.append(best_fit)
            history_mean.append(mean_fit)
            history_worst.append(worst_fit)

            if best_fit < best_ever - 1e-4:
                best_ever = best_fit
                no_improve_count = 0
            else:
                no_improve_count += 1

            elapsed = time.time() - t0
            if (gen + 1) % PRINT_EVERY == 0 or no_improve_count == 0:
                hof_w = f'{hof[0]["weight"]:.2f}kg' if hof else '---'
                n_ok = sum(1 for v in hof[0]['eval']['zone_results'].values()
                           if v['ok']) if hof else 0
                print(f'  Gen {gen+1:>3}/{self.n_gen} | '
                      f'best={best_fit:.2f} | '
                      f'mean={mean_fit:.2f} | '
                      f'HoF={hof_w} ({n_ok}/{self.n_zones} OK) | '
                      f'sin mejora={no_improve_count:>2} | '
                      f'{elapsed:.1f}s')

            if no_improve_count >= self.early_stop_pat:
                early_stop_gen = gen + 1
                print(f'  [Early stop] Gen {gen+1} — sin mejora por '
                      f'{self.early_stop_pat} gens ({elapsed:.1f}s)')
                break

            if no_improve_count > 0 and no_improve_count % RESTART_PAT == 0:
                restart_count += 1
                n_replace = int(self.pop_size * RESTART_RATIO)
                worst_idx = np.argsort(fits)[-n_replace:]
                for wi in worst_idx:
                    pop[wi] = self._repair(self._archetype_C())
                    fits[wi] = float('inf')
                print(f'  [Reinicio #{restart_count}] Gen {gen+1} — '
                      f'reemplazados {n_replace} ({elapsed:.1f}s)')

            elite_idx = np.argsort(fits)[:N_ELITE]
            elites = pop[elite_idx].copy()

            new_pop = np.zeros_like(pop)
            new_pop[:N_ELITE] = elites
            new_idx = N_ELITE

            while new_idx < self.pop_size:
                p1_idx = self._tournament(fits)
                p2_idx = self._tournament(fits)
                c1, c2 = self._crossover(pop[p1_idx], pop[p2_idx])
                c1 = self._repair(self._mutate(c1))
                c2 = self._repair(self._mutate(c2))
                new_pop[new_idx] = c1
                if new_idx + 1 < self.pop_size:
                    new_pop[new_idx + 1] = c2
                new_idx += 2

            pop = new_pop

        # Preparar top-3
        top3 = []
        for rank, entry in enumerate(hof[:3], start=1):
            ev = entry['eval']
            dec = ev['decoded']
            top3.append({
                'rank': rank,
                'diam_A': REBAR_CATALOG[dec['diam_A']]['name'],
                'diam_B': REBAR_CATALOG[dec['diam_B']]['name'],
                'diam_A_idx': dec['diam_A'],
                'diam_B_idx': dec['diam_B'],
                'corrido_top_bars': ev['corr_bars_top'],
                'corrido_bot_bars': ev['corr_bars_bot'],
                'corrido_top_matrix': dec['corrido_top'],
                'corrido_bot_matrix': dec['corrido_bot'],
                'bastones_matrices': dec['bastones'],
                'phi_mn_corrido': ev['phi_mn_corrido'],
                'as_corrido_bot': ev['as_corrido_bot'],
                'as_corrido_top': ev['as_corrido_top'],
                'zone_results': ev['zone_results'],
                'zone_extents': ev['zone_extents'],
                'support_detail': ev['support_detail'],
                'support_cells': ev.get('support_cells', {}),
                'corrido_anchorage': ev.get('corrido_anchorage',
                                            {'TOP': [], 'BOT': []}),
                'total_weight_kg': ev['total_weight_kg'],
                'corrido_weight_kg': ev['corrido_weight_kg'],
                'baston_weight_kg': ev['baston_weight_kg'],
                'feasible': ev['feasible'],
                'zones_ok': sum(1 for v in ev['zone_results'].values()
                                if v['ok']),
                'violations': ev['violations'],
                'fitness': entry['fitness'],
                'z': entry['z'],
            })

        elapsed = time.time() - t0
        gens_run = early_stop_gen if early_stop_gen else self.n_gen

        return {
            'top3': top3,
            'history_best': history_best,
            'history_mean': history_mean,
            'history_worst': history_worst,
            'generations_run': gens_run,
            'early_stop': early_stop_gen is not None,
            'elapsed_s': round(elapsed, 1),
        }


# ---------------------------------------------------------------------------
# Test rapido
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import json

    test_path = os.path.join(_HERE, 'test_beam.json')
    with open(test_path, 'r') as f:
        beam = json.load(f)

    print(f"Viga: {beam['id']}, n_spans={beam['n_spans']}")
    print(f"b={beam['inputs']['b_m']}, h={beam['inputs']['h_m']}")

    ga = BeamGANSpans(beam, pop_size=20, n_gen=10, seed=42)
    print(f"chrom_L={ga.chrom_L}, n_slots={ga.n_slots}, "
          f"n_zones={ga.n_zones}")

    result = ga.run()
    print(f"\nGeneraciones: {result['generations_run']}")
    print(f"Tiempo: {result['elapsed_s']}s")

    for r in result['top3']:
        print(f"\nRank {r['rank']}: W={r['total_weight_kg']:.2f}kg, "
              f"feasible={r['feasible']}, "
              f"zones_ok={r['zones_ok']}/{ga.n_zones}")
