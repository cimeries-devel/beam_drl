"""BeamGANSpans: GA para diseno de corrido + bastones en vigas de N tramos.

Cromosoma nuevo (sin genes globales diam_A/diam_B):
  - Cada slot: [dO, on_i]  con dO en REBAR_CATALOG_N (0=3/8" .. 5=1")
  - Offsets de bloques: corrido_top=0, corrido_bot=BLOCK, bastones=2*BLOCK+zone*BLOCK

Retorna top-3 disenos (feasible primero, luego menor peso).
"""

import time

import numpy as np

from config.config import VARILLAS_POR_ANCHO, R1
from ga.chromosome import n_slots_for_beam, block_size

from .config_n import (
    N_CAPAS_MAX,
    generate_zone_ids,
    parse_zone_id,
    scaled_ga_params,
    CORRIDO_SIMETRICO,
    TOURNAMENT_K, N_ELITE,
    P_CROSS, P_MUT_ONI, P_MUT_RESET, P_MUT_DO,
    HOF_POOL_SIZE, RESTART_PAT, RESTART_RATIO,
    REBAR_CATALOG_N, N_DIAM,
)
from .chromosome_n import (
    chrom_length_n, decode_n, encode_n, repair_n,
)
from .fitness_n import evaluate_and_fitness_n


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

        n_A = max(1, int(self.pop_size * 0.30))
        for _ in range(n_A):
            if idx >= self.pop_size:
                break
            pop[idx] = self._repair(self._archetype_A())
            idx += 1

        n_B = max(1, int(self.pop_size * 0.40))
        for _ in range(n_B):
            if idx >= self.pop_size:
                break
            pop[idx] = self._repair(self._archetype_B())
            idx += 1

        while idx < self.pop_size:
            pop[idx] = self._repair(self._archetype_C())
            idx += 1

        return pop

    def _activate_corners(self, z: np.ndarray, block_offset: int, dO: int = 2):
        """Activa slots de esquina (capa 0, slot 0 y slot n_slots-1) con dO dado."""
        z[block_offset + 0] = dO
        z[block_offset + 1] = 1
        last = block_offset + (self.n_slots - 1) * 2
        z[last] = dO
        z[last + 1] = 1

    def _repair(self, z: np.ndarray) -> np.ndarray:
        return repair_n(z, self.n_slots, self.n_spans,
                        self.min_v, self.max_v,
                        self.corrido_simetrico,
                        beam=self.beam)

    def _archetype_A(self) -> np.ndarray:
        """Corrido minimo (esquinas capa 0), bastones OFF."""
        z = np.zeros(self.chrom_L, dtype=np.int8)
        dO_corr = np.random.randint(1, 4)  # 1/2" a 3/4"
        self._activate_corners(z, 0, dO=dO_corr)               # corrido_top
        self._activate_corners(z, self.BLOCK, dO=dO_corr)      # corrido_bot
        return z

    def _archetype_B(self) -> np.ndarray:
        """Corrido minimo + bastones informados por demanda."""
        z = np.zeros(self.chrom_L, dtype=np.int8)
        dO_corr = np.random.randint(1, 4)
        self._activate_corners(z, 0, dO=dO_corr)
        self._activate_corners(z, self.BLOCK, dO=dO_corr)

        as_corr = 2 * REBAR_CATALOG_N[dO_corr]['area_cm2']
        # Slots interiores (no esquinas): no estan bloqueados por el corrido
        free_slots = list(range(1, self.n_slots - 1)) or [self.n_slots // 2]

        for zone_ord, zone_id in enumerate(self.zone_ids):
            mu = self.mu_zones.get(zone_id, 0.0)
            if mu <= 0.01:
                continue
            mn_approx = as_corr * 4200 * self.d_cm / 100_000
            if mn_approx >= mu:
                continue
            bast_offset = 2 * self.BLOCK + zone_ord * self.BLOCK
            dO_bast = min(dO_corr + 1, N_DIAM - 1)

            # Capa 0: solo slots libres (evitar esquinas bloqueadas por corrido)
            for slot in free_slots:
                base = bast_offset + slot * 2
                z[base] = dO_bast
                z[base + 1] = 1

            # Para demanda muy alta, activar tambien capa 1 completa
            if mu > 2.0 * mn_approx:
                dO_capa1 = min(dO_corr + 2, N_DIAM - 1)
                for s in range(self.n_slots):
                    base = bast_offset + self.n_slots * 2 + s * 2
                    z[base] = dO_capa1
                    z[base + 1] = 1
        return z

    def _archetype_C(self) -> np.ndarray:
        """Completamente aleatorio."""
        z = np.zeros(self.chrom_L, dtype=np.int8)
        for gi in range(self.chrom_L // 2):
            off = gi * 2
            z[off] = np.random.randint(0, N_DIAM)
            z[off + 1] = 1 if np.random.random() < 0.25 else 0
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

        n_blocks = 2 + self.n_zones
        for bi in range(n_blocks):
            if np.random.random() < 0.5:
                off = bi * self.BLOCK
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

        n_blocks = 2 + self.n_zones
        for bi in range(n_blocks):
            off = bi * self.BLOCK

            # Reset aleatorio solo en bloques baston (bi >= 2)
            if bi >= 2 and np.random.random() < P_MUT_RESET:
                z[off: off + self.BLOCK] = 0
                continue

            for gi in range(self.BLOCK // 2):
                gene_off = off + gi * 2
                on_i = int(z[gene_off + 1])

                # Flip on/off
                if np.random.random() < P_MUT_ONI:
                    z[gene_off + 1] = 1 - on_i
                    on_i = 1 - on_i

                # Mutar dO en slots activos
                if on_i == 1 and np.random.random() < P_MUT_DO:
                    delta = np.random.choice([-1, 1])
                    z[gene_off] = int(np.clip(int(z[gene_off]) + delta, 0, N_DIAM - 1))

        return z

    # -------------------------------------------------------------------
    # Hall of Fame
    # -------------------------------------------------------------------

    @staticmethod
    def _corrido_signature(ev: dict) -> tuple:
        n_bars_top = len(ev.get('corr_bars_top', []))
        n_bars_bot = len(ev.get('corr_bars_bot', []))
        dec = ev['decoded']
        top_active = dec['corrido_top'][:, :, 0][dec['corrido_top'][:, :, 1] == 1]
        med_top = int(np.median(top_active)) if len(top_active) > 0 else -1
        return (med_top, n_bars_top, n_bars_bot)

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
            # Feasible primero, luego por fitness (= peso para feasible,
            # peso + penalizaciones para infactible → premia mas zonas cubiertas)
            return (0 if e['feasible'] else 1, e['fitness'])

        def _dominates(new, old):
            if new['feasible'] and not old['feasible']:
                return True
            if new['feasible'] == old['feasible']:
                return new['fitness'] < old['fitness']
            return False

        for i, existing in enumerate(hof):
            if existing['sig'] == sig:
                if _dominates(entry, existing):
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
                print(f'  [Early stop] Gen {gen+1} -- sin mejora por '
                      f'{self.early_stop_pat} gens ({elapsed:.1f}s)')
                break

            if no_improve_count > 0 and no_improve_count % RESTART_PAT == 0:
                restart_count += 1
                n_replace = int(self.pop_size * RESTART_RATIO)
                worst_idx = np.argsort(fits)[-n_replace:]
                for wi in worst_idx:
                    pop[wi] = self._repair(self._archetype_C())
                    fits[wi] = float('inf')
                print(f'  [Reinicio #{restart_count}] Gen {gen+1} -- '
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
        print(f"  violations: {r['violations']}")
