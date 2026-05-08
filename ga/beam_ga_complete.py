"""BeamGAComplete: GA completo para diseño de corrido + bastones de viga.

Optimiza simultáneamente:
  - 2 diámetros globales (diam_A ≤ diam_B)
  - Corrido (mismas barras en TOP y BOT, corre toda la viga)
  - Bastones en 6 zonas (Z1-Z3 × TOP+BOT)

Retorna top-3 diseños (feasible primero, luego menor peso).
"""

import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(str(_HERE), '..', 'mejora del modelo'))
sys.path.insert(0, os.path.join(str(_HERE), '..', 'GA beam'))
sys.path.insert(0, os.path.join(str(_HERE), '..', 'GA_beam_compatibilizado'))

from config.config import REBAR_CATALOG, VARILLAS_POR_ANCHO, R1
from config.config_ga import (
    POP_SIZE, N_GEN, TOURNAMENT_K, N_ELITE,
    P_CROSS, P_MUT_ONI, P_MUT_CHOICE, P_MUT_DIAM, P_MUT_RESET,
    EARLY_STOP_PAT, HOF_POOL_SIZE, RESTART_PAT, RESTART_RATIO,
    N_CAPAS_MAX, ZONE_IDS,
)
from ga.chromosome import (
    n_slots_for_beam, chrom_length, block_size,
    repair,
)
from ga.fitness import evaluate_and_fitness
from ga import chromosome_utils

N_ZONES = 6
_ZONE_ORDER = ['LEFT_TOP', 'MID_TOP', 'RIGHT_TOP', 'LEFT_BOT', 'MID_BOT', 'RIGHT_BOT']


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class BeamGAComplete:
    def __init__(self, beam: dict,
                 pop_size: int = POP_SIZE,
                 n_gen: int = N_GEN,
                 tournament_k: int = TOURNAMENT_K,
                 seed: int = None,
                 p_mut_oni: float = P_MUT_ONI,
                 p_mut_choice: float = P_MUT_CHOICE,
                 p_mut_diam: float = P_MUT_DIAM,
                 p_mut_reset: float = P_MUT_RESET):
        self.beam = beam
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.tournament_k = tournament_k
        self.p_mut_oni = p_mut_oni
        self.p_mut_choice = p_mut_choice
        self.p_mut_diam = p_mut_diam
        self.p_mut_reset = p_mut_reset

        if seed is not None:
            np.random.seed(seed)

        inp = beam['inputs']
        self.b   = round(float(inp['b_m']), 2)
        self.h   = float(inp['h_m'])
        self.fc  = float(inp['fc_kg_cm2'])
        self.L   = float(beam['outputs']['x_m'][-1])

        self.b_cm = self.b * 100.0
        self.h_cm = self.h * 100.0
        self.d_cm = (self.h - R1) * 100.0

        self.min_v, self.max_v = VARILLAS_POR_ANCHO[self.b]
        self.n_slots  = n_slots_for_beam(self.b)
        self.chrom_L  = chrom_length(self.n_slots)
        self.BLOCK    = block_size(self.n_slots)

        xs = beam['outputs']['x_m']
        outs = beam['outputs']
        if 'M_tonf_m_max' in outs and 'M_tonf_m_min' in outs:
            ms_max = outs['M_tonf_m_max']
            ms_min = outs['M_tonf_m_min']
            ms = ms_max
        else:
            ms = outs['M_tonf_m']
            ms_max = None
            ms_min = None
        self.xs = xs
        self.ms = ms

        def _mu_approx(xs_arr, ms_arr, x_lo, x_hi, face):
            if face == 'BOT':
                vals = [m for x, m in zip(xs_arr, ms_arr) if x_lo <= x <= x_hi and m > 0]
                return max(vals) if vals else 0.0
            else:
                vals = [abs(m) for x, m in zip(xs_arr, ms_arr) if x_lo <= x <= x_hi and m < 0]
                return max(vals) if vals else 0.0

        t1, t2 = self.L / 3, 2 * self.L / 3
        ms_for_bot = ms_max if ms_max is not None else ms
        ms_for_top = ms_min if ms_min is not None else ms
        self.mu_zones = {
            'LEFT_TOP':  _mu_approx(xs, ms_for_top, 0,  t1, 'TOP'),
            'MID_TOP':   _mu_approx(xs, ms_for_top, t1, t2, 'TOP'),
            'RIGHT_TOP': _mu_approx(xs, ms_for_top, t2, self.L, 'TOP'),
            'LEFT_BOT':  _mu_approx(xs, ms_for_bot, 0,  t1, 'BOT'),
            'MID_BOT':   _mu_approx(xs, ms_for_bot, t1, t2, 'BOT'),
            'RIGHT_BOT': _mu_approx(xs, ms_for_bot, t2, self.L, 'BOT'),
        }

    # -----------------------------------------------------------------------
    # Inicialización de la población
    # -----------------------------------------------------------------------

    def _init_population(self) -> np.ndarray:
        pop = np.zeros((self.pop_size, self.chrom_L), dtype=np.int8)
        idx = 0

        n_A = max(1, int(self.pop_size * 0.30))
        for _ in range(n_A):
            if idx >= self.pop_size:
                break
            z = chromosome_utils._archetype_A(self.chrom_L, self.n_slots)
            pop[idx] = repair(z, self.n_slots, self.min_v, self.max_v)
            idx += 1

        n_B = max(1, int(self.pop_size * 0.40))
        for _ in range(n_B):
            if idx >= self.pop_size:
                break
            z = chromosome_utils._archetype_B(self.chrom_L, self.n_slots, self.BLOCK, _ZONE_ORDER, self.mu_zones, self.d_cm)
            pop[idx] = repair(z, self.n_slots, self.min_v, self.max_v)
            idx += 1

        if idx < self.pop_size:
            z_ws = chromosome_utils._warm_start(self.chrom_L, self.beam, self.n_slots)
            if z_ws is not None:
                pop[idx] = repair(z_ws, self.n_slots, self.min_v, self.max_v)
                idx += 1

        while idx < self.pop_size:
            z = chromosome_utils._archetype_C(self.chrom_L)
            pop[idx] = repair(z, self.n_slots, self.min_v, self.max_v)
            idx += 1

        return pop

    # -----------------------------------------------------------------------
    # Selección por torneo
    # -----------------------------------------------------------------------

    def _tournament(self, fits: np.ndarray) -> int:
        candidates = np.random.choice(len(fits), self.tournament_k, replace=False)
        return int(candidates[np.argmin(fits[candidates])])

    # -----------------------------------------------------------------------
    # Hall of Fame (top-3)
    # -----------------------------------------------------------------------

    @staticmethod
    def _corrido_signature(ev: dict) -> tuple:
        dec = ev['decoded']
        n_bars = len(ev['corr_bars'])
        return (dec['diam_A'], dec['diam_B'], n_bars)

    def _update_hof(self, hof: list, ev: dict, fit: float, z: np.ndarray):
        sig = self._corrido_signature(ev)
        entry = {
            'eval':      ev,
            'fitness':   fit,
            'z':         z.copy(),
            'weight':    ev['total_weight_kg'],
            'feasible':  ev['feasible'],
            'sig':       sig,
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

    # -----------------------------------------------------------------------
    # Loop principal
    # -----------------------------------------------------------------------

    def _evolve_generation(self, fits, pop):
        elite_idx = np.argsort(fits)[:N_ELITE]
        elites = pop[elite_idx].copy()

        new_pop = np.zeros_like(pop)
        new_pop[:N_ELITE] = elites
        new_idx = N_ELITE

        while new_idx < self.pop_size:
            p1_idx = self._tournament(fits)
            p2_idx = self._tournament(fits)
            c1, c2 = chromosome_utils._crossover(pop[p1_idx], pop[p2_idx], self.BLOCK, P_CROSS)
            c1 = repair(chromosome_utils._mutate(c1, self.BLOCK, self.p_mut_diam, self.p_mut_reset, self.p_mut_oni, self.p_mut_choice), self.n_slots, self.min_v, self.max_v)
            c2 = repair(chromosome_utils._mutate(c2, self.BLOCK, self.p_mut_diam, self.p_mut_reset, self.p_mut_oni, self.p_mut_choice), self.n_slots, self.min_v, self.max_v)
            new_pop[new_idx] = c1
            if new_idx + 1 < self.pop_size:
                new_pop[new_idx + 1] = c2
            new_idx += 2
        return new_pop

    def _eval_population(self, pop, fits, hof, evals):
        for i in range(self.pop_size):
            ev, fit = evaluate_and_fitness(pop[i], self.beam, self.n_slots)
            fits[i]  = fit
            evals[i] = ev
            pop[i] = ev['z_repaired']
            self._update_hof(hof, ev, fit, pop[i])

    def run(self) -> dict:
        pop = self._init_population()
        fits = np.full(self.pop_size, float('inf'))
        evals = [None] * self.pop_size

        hof = []
        history_best  = []
        history_mean  = []
        history_worst = []

        no_improve_count = 0
        best_ever = float('inf')
        early_stop_gen = None
        restart_count = 0

        t0 = time.time()

        PRINT_EVERY = max(1, self.n_gen // 20)

        for gen in range(self.n_gen):
            self._eval_population(pop, fits, hof, evals)

            best_fit  = float(fits.min())
            mean_fit  = float(fits.mean())
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
                print(f'  Gen {gen+1:>3}/{self.n_gen} | '
                      f'best={best_fit:.2f} | '
                      f'mean={mean_fit:.2f} | '
                      f'worst={worst_fit:.2f} | '
                      f'HoF={hof_w} | '
                      f'sin mejora={no_improve_count:>2} | '
                      f'{elapsed:.1f}s')

            if no_improve_count >= EARLY_STOP_PAT:
                early_stop_gen = gen + 1
                print(f'  [Early stop] Gen {gen+1} — sin mejora por {EARLY_STOP_PAT} gens '
                      f'({elapsed:.1f}s)')
                break

            if no_improve_count > 0 and no_improve_count % RESTART_PAT == 0:
                restart_count += 1
                n_replace = int(self.pop_size * RESTART_RATIO)
                worst_idx = np.argsort(fits)[-n_replace:]
                for wi in worst_idx:
                    z_new = chromosome_utils._archetype_C(self.chrom_L)
                    pop[wi] = repair(z_new, self.n_slots, self.min_v, self.max_v)
                    fits[wi] = float('inf')
                print(f'  [Reinicio #{restart_count}] Gen {gen+1} — '
                      f'reemplazados {n_replace} individuos ({elapsed:.1f}s)')

            pop = self._evolve_generation(fits, pop)

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
                'corrido_bars': ev['corr_bars'],
                'corrido_matrix': dec['corrido'],
                'bastones_matrices': dec['bastones'],
                'phi_mn_corrido': ev['phi_mn_corrido'],
                'as_corrido': ev['as_corrido'],
                'zone_results': ev['zone_results'],
                'zone_extents': ev['zone_extents'],
                'total_weight_kg': ev['total_weight_kg'],
                'corrido_weight_kg': ev['corrido_weight_kg'],
                'baston_weight_kg': ev['baston_weight_kg'],
                'feasible': ev['feasible'],
                'zones_ok': sum(1 for v in ev['zone_results'].values() if v['ok']),
                'violations': ev['violations'],
                'fitness': entry['fitness'],
                'z': entry['z'],
            })

        elapsed = time.time() - t0
        gens_run = early_stop_gen if early_stop_gen else self.n_gen

        return {
            'top3': top3,
            'history_best':  history_best,
            'history_mean':  history_mean,
            'history_worst': history_worst,
            'generations_run': gens_run,
            'early_stop': early_stop_gen is not None,
            'elapsed_s': round(elapsed, 1),
        }