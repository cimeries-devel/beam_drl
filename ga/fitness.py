"""Función de fitness para GA_viga_completa.

Evalúa un cromosoma contra el diagrama de momentos de una viga y retorna
un dict con peso, capacidades por zona, violaciones y fitness total.

Definición de zonas (basada en cruces de M(x) con φMn_corrido):
  LEFT_{face}:  M_face(x=0) > φMn  → bastón desde x=0 hasta primer cruce + extensión
  RIGHT_{face}: M_face(x=L) > φMn  → bastón desde último cruce − extensión hasta x=L
  MID_{face}:   ∃ región interior donde M_face(x) > φMn (no toca extremos)
                bastón entre cruces ± extensión

Extensión E.060 Art. 12.10.3: max(d, 12·db)
"""


import os
import sys
import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, '..', 'mejora del modelo'))
sys.path.insert(0, os.path.join(_HERE, '..', 'GA beam'))
sys.path.insert(0, os.path.join(_HERE, '..', 'GA_beam_compatibilizado'))

from config.config import REBAR_CATALOG, VARILLAS_POR_ANCHO, R1
from domain.ga_beam import REBAR_WEIGHTS_KGM
from config.config_ga import (
    N_CAPAS_MAX, ZONE_IDS,
    LAMBDA_M, LAMBDA_N, LAMBDA_G, PENALTY_FIXED, LAMBDA_EXC, LAMBDA_CORR_EXC,
    LAMBDA_CONSTR_BAST,
    ET_MIN_DUCTILITY,
)
from ga.chromosome import decode, repair, get_active_bars, get_combined_bars, n_slots_for_beam
from domain.section_calculator import (
    compute_section_mn, compute_rho_min, compute_beta1,
)

from ga import fitness_components

# Mapa zona_id → (region_type, face)
_ZONE_MAP = {
    'LEFT_TOP':  ('LEFT',  'TOP'),
    'MID_TOP':   ('MID',   'TOP'),
    'RIGHT_TOP': ('RIGHT', 'TOP'),
    'LEFT_BOT':  ('LEFT',  'BOT'),
    'MID_BOT':   ('MID',   'BOT'),
    'RIGHT_BOT': ('RIGHT', 'BOT'),
}

# Orden interno de zonas (debe coincidir con chromosome._ZONE_ORDER)
_ZONE_ORDER_INTERNAL = ['LEFT_TOP', 'MID_TOP', 'RIGHT_TOP',
                        'LEFT_BOT', 'MID_BOT', 'RIGHT_BOT']


# ---------------------------------------------------------------------------
# Detección de zonas por cruces del diagrama de momentos
# ---------------------------------------------------------------------------

def _find_threshold_crossings(xs: list, ms: list, threshold: float,
                               face: str) -> list:
    signed = threshold if face == 'BOT' else -threshold
    crossings = []
    for i in range(len(ms) - 1):
        di = ms[i] - signed
        dj = ms[i + 1] - signed
        if di * dj < 0:
            t = di / (di - dj)
            crossings.append(xs[i] + t * (xs[i + 1] - xs[i]))
    return sorted(crossings)


def _compute_zone_info(xs: list, ms: list,
                       phi_pos: float, phi_neg: float,
                       d_m: float, db_A_m: float, db_B_m: float,
                       L: float,
                       ms_max: list = None, ms_min: list = None) -> dict:
    extension = max(d_m, 12.0 * max(db_A_m, db_B_m))
    result = {}
    xs_arr = np.array(xs, dtype=float)

    for face in ('TOP', 'BOT'):
        phi = phi_pos if face == 'BOT' else phi_neg

        if ms_max is not None and ms_min is not None:
            ms_face = ms_max if face == 'BOT' else ms_min
        else:
            ms_face = ms

        mf = np.array([max(m, 0.0) for m in ms_face] if face == 'BOT'
                      else [max(-m, 0.0) for m in ms_face], dtype=float)

        mf0 = float(mf[0])
        mfL = float(mf[-1])

        crossings = _find_threshold_crossings(xs, ms_face, phi, face)

        x_events = sorted(set([0.0] + crossings + [float(L)]))
        segments = []
        for i in range(len(x_events) - 1):
            xa, xb = x_events[i], x_events[i + 1]
            x_mid = (xa + xb) / 2.0
            mf_mid = float(np.interp(x_mid, xs_arr, mf))
            if mf_mid > phi:
                segments.append((xa, xb))

        def _mu_range(xa, xb):
            pts = [float(mf[i]) for i in range(len(xs)) if xa <= xs[i] <= xb]
            return max(pts) if pts else 0.0

        _EMPTY = {'exists': False,
                  'x_start': 0.0, 'x_end': 0.0,
                  'x_theor_start': 0.0, 'x_theor_end': 0.0, 'Mu': 0.0}

        left_seg = next((s for s in segments if abs(s[0]) < 1e-9), None)
        if left_seg is not None:
            x_te = left_seg[1]
            result[f'LEFT_{face}'] = {
                'exists': True,
                'x_start': 0.0,
                'x_end': min(L, x_te + extension),
                'x_theor_start': 0.0,
                'x_theor_end': x_te,
                'Mu': _mu_range(0.0, x_te),
            }
        else:
            result[f'LEFT_{face}'] = dict(_EMPTY)

        right_seg = next(
            (s for s in reversed(segments)
             if abs(s[1] - L) < 1e-9 and abs(s[0]) > 1e-9),
            None
        )
        if right_seg is not None:
            x_ts = right_seg[0]
            result[f'RIGHT_{face}'] = {
                'exists': True,
                'x_start': max(0.0, x_ts - extension),
                'x_end': L,
                'x_theor_start': x_ts,
                'x_theor_end': L,
                'Mu': _mu_range(x_ts, L),
            }
        else:
            result[f'RIGHT_{face}'] = dict(_EMPTY)

        mid_segs = [s for s in segments
                    if abs(s[0]) > 1e-9 and abs(s[1] - L) > 1e-9]
        if mid_segs:
            best = max(mid_segs, key=lambda s: _mu_range(s[0], s[1]))
            x_ts, x_te = best
            result[f'MID_{face}'] = {
                'exists': True,
                'x_start': max(0.0, x_ts - extension),
                'x_end': min(L, x_te + extension),
                'x_theor_start': x_ts,
                'x_theor_end': x_te,
                'Mu': _mu_range(x_ts, x_te),
            }
        else:
            result[f'MID_{face}'] = dict(_EMPTY)

    return result


# ---------------------------------------------------------------------------
# Evaluación del individuo
# ---------------------------------------------------------------------------

def eval_individual(z: np.ndarray, beam: dict, n_slots: int) -> dict:
    inp = beam['inputs']
    b   = round(float(inp['b_m']), 2)
    h   = float(inp['h_m'])
    fc  = float(inp['fc_kg_cm2'])
    L   = float(beam['outputs']['x_m'][-1])

    b_cm = b * 100.0
    h_cm = h * 100.0
    d_cm = (h - R1) * 100.0

    min_v, max_v = VARILLAS_POR_ANCHO[b]

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

    z_rep = repair(z, n_slots, min_v, max_v)
    dec   = decode(z_rep, n_slots)

    diam_A = dec['diam_A']
    diam_B = dec['diam_B']

    d_m    = h - R1
    db_A_m = REBAR_CATALOG[diam_A]['diam_cm'] / 100.0
    db_B_m = REBAR_CATALOG[diam_B]['diam_cm'] / 100.0

    corr_bars_bot = get_active_bars(dec['corrido'], 'BOT', diam_A, diam_B, h_cm)
    corr_bars_top = get_active_bars(dec['corrido'], 'TOP', diam_A, diam_B, h_cm)

    if corr_bars_bot:
        res_corr_pos    = compute_section_mn(corr_bars_bot, b_cm, h_cm, fc)
        phi_mn_corr_pos = res_corr_pos['positive']['phi_Mn']
        eps_t_pos       = res_corr_pos['positive']['epsilon_t']
        as_corrido      = sum(a for _, a, _ in corr_bars_bot)
        rho_corrido     = as_corrido / (b_cm * d_cm)
    else:
        phi_mn_corr_pos = 0.0
        eps_t_pos = 0.0
        as_corrido = 0.0
        rho_corrido = 0.0

    if corr_bars_top:
        res_corr_neg    = compute_section_mn(corr_bars_top, b_cm, h_cm, fc)
        phi_mn_corr_neg = res_corr_neg['negative']['phi_Mn']
        eps_t_neg       = res_corr_neg['negative']['epsilon_t']
    else:
        phi_mn_corr_neg = 0.0
        eps_t_neg = 0.0

    corr_bars = corr_bars_bot

    corrido_weight_kg = 0.0
    for _, area, dname in corr_bars:
        corrido_weight_kg += REBAR_WEIGHTS_KGM[_name_to_idx(dname)] * L
    corrido_weight_kg *= 2.0

    zone_info = _compute_zone_info(xs, ms,
                                   phi_mn_corr_pos, phi_mn_corr_neg,
                                   d_m, db_A_m, db_B_m, L,
                                   ms_max=ms_max, ms_min=ms_min)

    BLOCK = N_CAPAS_MAX * n_slots * 2
    for zone_ord, zone_id in enumerate(_ZONE_ORDER_INTERNAL):
        if not zone_info[zone_id]['exists']:
            bast_offset = 2 + BLOCK + zone_ord * BLOCK
            z_rep[bast_offset: bast_offset + BLOCK] = 0

    dec = decode(z_rep, n_slots)

    zone_results = {}
    baston_weight_kg = 0.0

    for zone_id in ZONE_IDS:
        zinfo = zone_info[zone_id]
        mu = zinfo['Mu']
        bast_len = (zinfo['x_end'] - zinfo['x_start']) if zinfo['exists'] else 0.0

        face = _ZONE_MAP[zone_id][1]
        bast_mat = dec['bastones'][zone_id]
        combined = get_combined_bars(dec['corrido'], bast_mat, face,
                                     diam_A, diam_B, h_cm)

        all_bars  = [(y, a, d) for y, a, d, *_ in combined]
        bast_only = [(d, a) for _, a, d, bt, *_ in combined if bt == 'baston']

        if all_bars:
            res_z = compute_section_mn(all_bars, b_cm, h_cm, fc)
            phi_mn_z = (res_z['positive']['phi_Mn'] if face == 'BOT'
                        else res_z['negative']['phi_Mn'])
        else:
            phi_mn_z = 0.0

        z_weight = 0.0
        for dname, _ in bast_only:
            z_weight += REBAR_WEIGHTS_KGM[_name_to_idx(dname)] * bast_len

        bast_slots_per_layer = []
        for k in range(N_CAPAS_MAX):
            n_on = int(bast_mat[k, :, 1].sum())
            bast_slots_per_layer.append(n_on)

        baston_weight_kg += z_weight
        zone_results[zone_id] = {
            'phi_Mn':    round(phi_mn_z, 4),
            'Mu':        round(mu, 4),
            'ok':        phi_mn_z >= mu,
            'weight_kg': round(z_weight, 4),
            'n_bast':    len(bast_only),
            'bast_slots_per_layer': bast_slots_per_layer,
        }

    total_weight_kg = corrido_weight_kg + baston_weight_kg

    rho_min_check = compute_rho_min(fc)
    feasible = (all(v['ok'] for v in zone_results.values())
                and rho_corrido >= rho_min_check)

    return {
        'total_weight_kg':    round(total_weight_kg, 3),
        'corrido_weight_kg':  round(corrido_weight_kg, 3),
        'baston_weight_kg':   round(baston_weight_kg, 3),
        'phi_mn_corrido':     {
            'positive': round(phi_mn_corr_pos, 4),
            'negative': round(phi_mn_corr_neg, 4),
        },
        'zone_results':       zone_results,
        'zone_extents':       zone_info,
        'violations':         {},
        'feasible':           feasible,
        'as_corrido':         round(as_corrido, 4),
        'rho_corrido':        round(rho_corrido, 6),
        'eps_t_corrido_pos':  round(eps_t_pos, 6),
        'eps_t_corrido_neg':  round(eps_t_neg, 6),
        'decoded':            dec,
        'z_repaired':         z_rep,
        'corr_bars':          corr_bars,
        'xs':                 xs,
        'ms':                 ms,
        'ms_max':             ms_max,
        'ms_min':             ms_min,
        'L':                  L,
    }


def compute_fitness(eval_result: dict, b_cm: float, d_cm: float,
                    as_corrido: float, fc: float) -> float:
    return fitness_components.compute_fitness(
        eval_result, b_cm, d_cm, as_corrido, fc,
        compute_rho_min, ET_MIN_DUCTILITY, N_CAPAS_MAX,
        LAMBDA_M, LAMBDA_N, LAMBDA_G, PENALTY_FIXED,
        LAMBDA_EXC, LAMBDA_CORR_EXC, LAMBDA_CONSTR_BAST
    )


def _name_to_idx(dname: str) -> int:
    for idx, info in REBAR_CATALOG.items():
        if info['name'] == dname:
            return idx
    return 0


# ---------------------------------------------------------------------------
# Wrapper combinado
# ---------------------------------------------------------------------------

def evaluate_and_fitness(z: np.ndarray, beam: dict, n_slots: int) -> tuple:
    """Retorna (eval_result, fitness_value)."""
    inp = beam['inputs']
    b   = round(float(inp['b_m']), 2)
    h   = float(inp['h_m'])
    fc  = float(inp['fc_kg_cm2'])
    b_cm = b * 100.0
    d_cm = ((h - R1) * 100.0)

    ev = eval_individual(z, beam, n_slots)
    as_c = ev['as_corrido']
    fit  = compute_fitness(ev, b_cm, d_cm, as_c, fc)
    return ev, fit
