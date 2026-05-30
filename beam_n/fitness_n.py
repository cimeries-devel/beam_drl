"""Funcion de fitness para GA de vigas de N tramos.

Evalua un cromosoma contra la envolvente de momentos de n tramos continuos.
Detecta zonas por tramo, fusiona barras en apoyos interiores, y calcula
peso + penalizaciones.

Cromosoma nuevo (sin genes globales diam_A/diam_B):
  - Cada slot: [dO, on_i]  con dO en REBAR_CATALOG_N (0=3/8" .. 5=1")
  - Penalizacion LAMBDA_DIAM_CONTIG si diametros en una zona no son contiguos
"""

import numpy as np

from config.config import VARILLAS_POR_ANCHO, R1
from ga.chromosome import block_size
from domain.section_calculator import compute_section_mn, compute_rho_min

from .config_n import (
    N_CAPAS_MAX,
    generate_zone_ids,
    parse_zone_id,
    LAMBDA_M, LAMBDA_N, LAMBDA_G, PENALTY_FIXED,
    LAMBDA_EXC, LAMBDA_CORR_EXC, LAMBDA_CONSTR_BAST,
    ET_MIN_DUCTILITY,
    LAMBDA_ANCHORAGE,
    ensure_joints,
    REBAR_CATALOG_N,
    REBAR_WEIGHTS_KGM_N,
    LAMBDA_DIAM_CONTIG,
)
from .anchorage import anchorage_length_m, classify_anchorage
from .chromosome_n import (
    decode_n, repair_n, chrom_length_n,
    get_active_bars_n, get_combined_bars_n,
)


def _dO_from_name(dname: str) -> int:
    """Indice dO en REBAR_CATALOG_N dado el nombre '3/8', '1/2', etc."""
    for dO, info in REBAR_CATALOG_N.items():
        if info['name'] == dname:
            return dO
    return 2  # fallback a 5/8"


# ---------------------------------------------------------------------------
# Penalizacion por diametros no contiguos
# ---------------------------------------------------------------------------

def _compute_p_diam_contig(dec: dict, n_slots: int) -> float:
    """Penaliza spread de dO > 1 en cada capa de cada zona activa.

    Para cada capa con >= 2 barras activas:
      spread = max(dO) - min(dO) entre slots activos
      P += max(0, spread - 1)
    """
    P = 0.0
    all_mats = [dec['corrido_top'], dec['corrido_bot']]
    all_mats += list(dec['bastones'].values())
    for mat in all_mats:
        for k in range(N_CAPAS_MAX):
            oni_k = mat[k, :, 1]
            if int(oni_k.sum()) < 2:
                continue
            active_dOs = mat[k, oni_k == 1, 0].astype(int)
            spread = int(active_dOs.max()) - int(active_dOs.min())
            P += max(0, spread - 1)
    return P


# ---------------------------------------------------------------------------
# Deteccion de zonas por tramo
# ---------------------------------------------------------------------------

def _find_threshold_crossings(xs, ms, threshold, face):
    """Encuentra x donde M_face(x) == threshold (interpolacion lineal)."""
    signed = threshold if face == 'BOT' else -threshold
    crossings = []
    for i in range(len(ms) - 1):
        di = ms[i] - signed
        dj = ms[i + 1] - signed
        if di * dj < 0:
            t = di / (di - dj)
            crossings.append(xs[i] + t * (xs[i + 1] - xs[i]))
    return sorted(crossings)


def _compute_zone_info_n(beam: dict,
                         phi_corr_pos: float,
                         phi_corr_neg: float,
                         d_m: float,
                         db_max_m: float) -> dict:
    """Determina zonas activas para todos los 6*n_spans tramos.

    Itera por tramo. Para cada tramo i con rango [x0, x1]:
    - Extrae sub-arrays de x, M_max, M_min dentro del tramo
    - Busca cruces con phi_corrido
    - LEFT = segmento que toca x0, RIGHT = que toca x1, MID = interior

    Returns
    -------
    dict: zone_id -> {exists, x_start, x_end, x_theor_start, x_theor_end, Mu}
    """
    spans = beam['spans']
    n_spans = beam['n_spans']
    xs_all = np.array(beam['outputs']['x_m'], dtype=float)

    outs = beam['outputs']
    ms_max_all = np.array(outs.get('M_tonf_m_max', outs.get('M_tonf_m', [])),
                          dtype=float)
    ms_min_all = np.array(outs.get('M_tonf_m_min', outs.get('M_tonf_m', [])),
                          dtype=float)

    extension = max(d_m, 12.0 * db_max_m)
    result = {}

    for span_i in range(n_spans):
        sp = spans[span_i]
        x0_span = sp['x0']
        x1_span = sp['x1']

        mask = (xs_all >= x0_span - 1e-6) & (xs_all <= x1_span + 1e-6)
        xs_span = xs_all[mask].tolist()
        ms_max_span = ms_max_all[mask].tolist()
        ms_min_span = ms_min_all[mask].tolist()

        if len(xs_span) < 2:
            for face in ('TOP', 'BOT'):
                for pos in ('LEFT', 'MID', 'RIGHT'):
                    zid = f'{pos}_{face}_T{span_i + 1}'
                    result[zid] = _empty_zone()
            continue

        for face in ('TOP', 'BOT'):
            phi = phi_corr_pos if face == 'BOT' else phi_corr_neg
            ms_face = ms_max_span if face == 'BOT' else ms_min_span

            mf = np.array(
                [max(m, 0.0) for m in ms_face] if face == 'BOT'
                else [max(-m, 0.0) for m in ms_face],
                dtype=float
            )

            crossings = _find_threshold_crossings(xs_span, ms_face, phi, face)

            x_events = sorted(set([xs_span[0]] + crossings + [xs_span[-1]]))
            segments = []
            xs_arr = np.array(xs_span, dtype=float)
            for i in range(len(x_events) - 1):
                xa, xb = x_events[i], x_events[i + 1]
                x_mid = (xa + xb) / 2.0
                mf_mid = float(np.interp(x_mid, xs_arr, mf))
                if mf_mid > phi:
                    segments.append((xa, xb))

            def _mu_range(xa, xb):
                pts = [float(mf[i]) for i in range(len(xs_span))
                       if xa <= xs_span[i] <= xb]
                return max(pts) if pts else 0.0

            left_seg = next(
                (s for s in segments if abs(s[0] - xs_span[0]) < 1e-6), None)
            zid_left = f'LEFT_{face}_T{span_i + 1}'
            if left_seg is not None:
                x_te = left_seg[1]
                result[zid_left] = {
                    'exists': True,
                    'x_start': xs_span[0],
                    'x_end': min(xs_span[-1], x_te + extension),
                    'x_theor_start': xs_span[0],
                    'x_theor_end': x_te,
                    'Mu': _mu_range(xs_span[0], x_te),
                }
            else:
                result[zid_left] = _empty_zone()

            right_seg = next(
                (s for s in reversed(segments)
                 if abs(s[1] - xs_span[-1]) < 1e-6
                 and abs(s[0] - xs_span[0]) > 1e-6),
                None
            )
            zid_right = f'RIGHT_{face}_T{span_i + 1}'
            if right_seg is not None:
                x_ts = right_seg[0]
                result[zid_right] = {
                    'exists': True,
                    'x_start': max(xs_span[0], x_ts - extension),
                    'x_end': xs_span[-1],
                    'x_theor_start': x_ts,
                    'x_theor_end': xs_span[-1],
                    'Mu': _mu_range(x_ts, xs_span[-1]),
                }
            else:
                result[zid_right] = _empty_zone()

            mid_segs = [
                s for s in segments
                if abs(s[0] - xs_span[0]) > 1e-6
                and abs(s[1] - xs_span[-1]) > 1e-6
            ]
            zid_mid = f'MID_{face}_T{span_i + 1}'
            if mid_segs:
                best = max(mid_segs, key=lambda s: _mu_range(s[0], s[1]))
                x_ts, x_te = best
                result[zid_mid] = {
                    'exists': True,
                    'x_start': max(xs_span[0], x_ts - extension),
                    'x_end': min(xs_span[-1], x_te + extension),
                    'x_theor_start': x_ts,
                    'x_theor_end': x_te,
                    'Mu': _mu_range(x_ts, x_te),
                }
            else:
                result[zid_mid] = _empty_zone()

    return result


def _empty_zone():
    return {
        'exists': False,
        'x_start': 0.0, 'x_end': 0.0,
        'x_theor_start': 0.0, 'x_theor_end': 0.0,
        'Mu': 0.0,
    }


# ---------------------------------------------------------------------------
# Evaluacion del individuo
# ---------------------------------------------------------------------------

def eval_individual_n(z: np.ndarray, beam: dict,
                      n_slots: int, n_spans: int) -> dict:
    """Evalua un cromosoma de n tramos y retorna metricas completas."""
    inp = beam['inputs']
    b   = round(float(inp['b_m']), 2)
    h   = float(inp['h_m'])
    fc  = float(inp['fc_kg_cm2'])

    b_cm = b * 100.0
    h_cm = h * 100.0
    d_cm = (h - R1) * 100.0
    d_m  = h - R1

    min_v, max_v = VARILLAS_POR_ANCHO[b]

    L_total = beam['spans'][-1]['x1'] - beam['spans'][0]['x0']

    # --- Reparar + decodificar ---
    z_rep = repair_n(z, n_slots, n_spans, min_v, max_v, beam=beam)
    dec = decode_n(z_rep, n_slots, n_spans)

    # --- Max diametro del corrido para calcular extension de corte ---
    def _max_dO_mat(mat):
        active = mat[:, :, 1].flatten()
        dOs = mat[:, :, 0].flatten()
        vals = dOs[active == 1]
        return int(vals.max()) if len(vals) > 0 else 2
    max_dO = max(_max_dO_mat(dec['corrido_top']), _max_dO_mat(dec['corrido_bot']))
    db_max_m = REBAR_CATALOG_N[max_dO]['diam_cm'] / 100.0

    # --- Barras del corrido (TOP y BOT por separado) ---
    corr_bars_top = get_active_bars_n(dec['corrido_top'], 'TOP', h_cm)
    corr_bars_bot = get_active_bars_n(dec['corrido_bot'], 'BOT', h_cm)

    # --- Capacidad del corrido ---
    if corr_bars_bot:
        # TOP bars act as compression steel for M+ (small y_from_top → near compression face)
        res_corr_pos = compute_section_mn(corr_bars_bot + corr_bars_top, b_cm, h_cm, fc)
        phi_mn_corr_pos = res_corr_pos['positive']['phi_Mn']
        eps_t_pos = res_corr_pos['positive']['epsilon_t']
        as_corrido_bot = sum(a for _, a, _ in corr_bars_bot)
    else:
        phi_mn_corr_pos = 0.0
        eps_t_pos = 0.0
        as_corrido_bot = 0.0

    if corr_bars_top:
        # BOT bars act as compression steel for M- (large y_from_top → near bottom compression face)
        res_corr_neg = compute_section_mn(corr_bars_top + corr_bars_bot, b_cm, h_cm, fc)
        phi_mn_corr_neg = res_corr_neg['negative']['phi_Mn']
        eps_t_neg = res_corr_neg['negative']['epsilon_t']
        as_corrido_top = sum(a for _, a, _ in corr_bars_top)
    else:
        phi_mn_corr_neg = 0.0
        eps_t_neg = 0.0
        as_corrido_top = 0.0

    rho_corrido_bot = as_corrido_bot / (b_cm * d_cm) if d_cm > 0 else 0.0
    rho_corrido_top = as_corrido_top / (b_cm * d_cm) if d_cm > 0 else 0.0

    # --- Peso corrido con anclaje en apoyos externos ---
    ensure_joints(beam)
    joints = beam['joints']
    h_col_L = float(joints[0]['h_col_m'])
    h_col_R = float(joints[-1]['h_col_m'])

    corrido_anchorage = {'TOP': [], 'BOT': []}

    def _corrido_weight_face(corrido_mat, face):
        w = 0.0
        infeasible = False
        for c in range(N_CAPAS_MAX):
            for s in range(n_slots):
                if int(corrido_mat[c, s, 1]) == 0:
                    continue
                dO = int(corrido_mat[c, s, 0])
                dname = REBAR_CATALOG_N[dO]['name']
                dec_L = classify_anchorage(dname, fc, face, h_col_L, c)
                dec_R = classify_anchorage(dname, fc, face, h_col_R, c)
                anc_L = anchorage_length_m(dname, fc, face, h_col_L, c)
                anc_R = anchorage_length_m(dname, fc, face, h_col_R, c)
                if anc_L is None or anc_R is None:
                    infeasible = True
                    Lbar = L_total + 1.5
                else:
                    Lbar = L_total + anc_L + anc_R
                w += REBAR_WEIGHTS_KGM_N[dO] * Lbar
                corrido_anchorage[face].append({
                    'capa': c, 'slot': s, 'diam': dname,
                    'left': dec_L, 'right': dec_R,
                })
        return w, infeasible

    corrido_top_weight, infeas_ext_top = _corrido_weight_face(
        dec['corrido_top'], 'TOP')
    corrido_bot_weight, infeas_ext_bot = _corrido_weight_face(
        dec['corrido_bot'], 'BOT')
    corrido_weight_kg = corrido_top_weight + corrido_bot_weight
    P_anclaje_ext = 100.0 if (infeas_ext_top or infeas_ext_bot) else 0.0

    # --- Zonas basadas en cruces M(x) con phi_corrido ---
    zone_info = _compute_zone_info_n(beam, phi_mn_corr_pos, phi_mn_corr_neg,
                                     d_m, db_max_m)

    # --- Enmascarar bastones inactivos ---
    zone_ids = generate_zone_ids(n_spans)
    BLOCK = block_size(n_slots)
    for zone_ord, zone_id in enumerate(zone_ids):
        if not zone_info[zone_id]['exists']:
            bast_offset = 2 * BLOCK + zone_ord * BLOCK
            z_rep[bast_offset: bast_offset + BLOCK] = 0
    dec = decode_n(z_rep, n_slots, n_spans)

    # --- Evaluar cada zona ---
    zone_results = {}
    baston_weight_kg = 0.0

    for zone_id in zone_ids:
        zinfo = zone_info[zone_id]
        mu = zinfo['Mu']
        bast_len = (zinfo['x_end'] - zinfo['x_start']) if zinfo['exists'] else 0.0

        parsed = parse_zone_id(zone_id)
        face = parsed['face']

        corrido_mat = dec['corrido_top'] if face == 'TOP' else dec['corrido_bot']
        bast_mat = dec['bastones'][zone_id]

        combined = get_combined_bars_n(corrido_mat, bast_mat, face, h_cm)
        tension_bars = [(y, a, d) for y, a, d, *_ in combined]
        bast_only = [(d, a) for _, a, d, bt, *_ in combined if bt == 'baston']

        # Add opposite-face corrido as compression steel (motor handles via bar position)
        opp_bars = corr_bars_bot if face == 'TOP' else corr_bars_top
        all_bars = tension_bars + [(y, a, d) for y, a, d in opp_bars] if tension_bars else []

        if all_bars:
            res_z = compute_section_mn(all_bars, b_cm, h_cm, fc)
            phi_mn_z = (res_z['positive']['phi_Mn'] if face == 'BOT'
                        else res_z['negative']['phi_Mn'])
        else:
            phi_mn_z = 0.0

        z_weight = 0.0
        for dname, _ in bast_only:
            dO = _dO_from_name(dname)
            z_weight += REBAR_WEIGHTS_KGM_N[dO] * bast_len

        bast_slots_per_layer = [
            int(bast_mat[k, :, 1].sum()) for k in range(N_CAPAS_MAX)
        ]

        baston_weight_kg += z_weight
        zone_results[zone_id] = {
            'phi_Mn': round(phi_mn_z, 4),
            'Mu': round(mu, 4),
            'ok': phi_mn_z >= mu,
            'weight_kg': round(z_weight, 4),
            'n_bast': len(bast_only),
            'bast_slots_per_layer': bast_slots_per_layer,
        }

    # --- Fusion parcial en apoyos interiores ---
    support_detail = {}
    support_cells = {}
    for k in range(n_spans - 1):
        h_col_int = float(joints[k + 1]['h_col_m'])
        for face in ('TOP', 'BOT'):
            right_id = f'RIGHT_{face}_T{k + 1}'
            left_id = f'LEFT_{face}_T{k + 2}'
            right_mat = dec['bastones'][right_id]
            left_mat = dec['bastones'][left_id]

            n_continue = 0
            n_term_r = 0
            n_term_l = 0
            cells = []
            for c in range(N_CAPAS_MAX):
                for s in range(n_slots):
                    r_on = int(right_mat[c, s, 1])
                    l_on = int(left_mat[c, s, 1])
                    if r_on == 0 and l_on == 0:
                        continue
                    if r_on == 1 and l_on == 1:
                        n_continue += 1
                        dO = int(right_mat[c, s, 0])
                        dname = REBAR_CATALOG_N[dO]['name']
                        cells.append({
                            'capa': c, 'slot': s, 'diam': dname,
                            'kind': 'continue', 'anchor': None,
                        })
                    elif r_on == 1 and l_on == 0:
                        n_term_r += 1
                        dO = int(right_mat[c, s, 0])
                        dname = REBAR_CATALOG_N[dO]['name']
                        dec_a = classify_anchorage(dname, fc, face, h_col_int, c)
                        anc = anchorage_length_m(dname, fc, face, h_col_int, c)
                        if anc is not None:
                            baston_weight_kg += REBAR_WEIGHTS_KGM_N[dO] * anc
                        cells.append({
                            'capa': c, 'slot': s, 'diam': dname,
                            'kind': 'term_R', 'anchor': dec_a,
                        })
                    else:
                        n_term_l += 1
                        dO = int(left_mat[c, s, 0])
                        dname = REBAR_CATALOG_N[dO]['name']
                        dec_a = classify_anchorage(dname, fc, face, h_col_int, c)
                        anc = anchorage_length_m(dname, fc, face, h_col_int, c)
                        if anc is not None:
                            baston_weight_kg += REBAR_WEIGHTS_KGM_N[dO] * anc
                        cells.append({
                            'capa': c, 'slot': s, 'diam': dname,
                            'kind': 'term_L', 'anchor': dec_a,
                        })

            r_zr = zone_results.get(right_id, {})
            l_zr = zone_results.get(left_id, {})
            key = f'{k + 1}_{face}'
            support_detail[key] = {
                'n_continue': n_continue,
                'n_terminate_right': n_term_r,
                'n_terminate_left': n_term_l,
                'phi_Mn_right': r_zr.get('phi_Mn', 0.0),
                'phi_Mn_left': l_zr.get('phi_Mn', 0.0),
                'Mu_right': r_zr.get('Mu', 0.0),
                'Mu_left': l_zr.get('Mu', 0.0),
                'ok': r_zr.get('ok', True) and l_zr.get('ok', True),
            }
            support_cells[key] = cells

    total_weight_kg = corrido_weight_kg + baston_weight_kg

    rho_min_check = compute_rho_min(fc)
    rho_corrido = max(rho_corrido_bot, rho_corrido_top)
    feasible = (all(v['ok'] for v in zone_results.values())
                and rho_corrido >= rho_min_check
                and P_anclaje_ext == 0.0)

    return {
        'total_weight_kg': round(total_weight_kg, 3),
        'corrido_weight_kg': round(corrido_weight_kg, 3),
        'baston_weight_kg': round(baston_weight_kg, 3),
        'phi_mn_corrido': {
            'positive': round(phi_mn_corr_pos, 4),
            'negative': round(phi_mn_corr_neg, 4),
        },
        'zone_results': zone_results,
        'zone_extents': zone_info,
        'support_detail': support_detail,
        'support_cells': support_cells,
        'corrido_anchorage': corrido_anchorage,
        'P_anclaje_ext': P_anclaje_ext,
        'violations': {},
        'feasible': feasible,
        'as_corrido_bot': round(as_corrido_bot, 4),
        'as_corrido_top': round(as_corrido_top, 4),
        'rho_corrido_bot': round(rho_corrido_bot, 6),
        'rho_corrido_top': round(rho_corrido_top, 6),
        'eps_t_corrido_pos': round(eps_t_pos, 6),
        'eps_t_corrido_neg': round(eps_t_neg, 6),
        'decoded': dec,
        'z_repaired': z_rep,
        'corr_bars_top': corr_bars_top,
        'corr_bars_bot': corr_bars_bot,
        'L_total': L_total,
    }


# ---------------------------------------------------------------------------
# Funcion de fitness
# ---------------------------------------------------------------------------

def compute_fitness_n(eval_result: dict, b_cm: float, d_cm: float,
                      fc: float, beam: dict) -> float:
    """Calcula fitness total para n tramos.

    fitness = W
            + LAMBDA_M * P_cap
            + LAMBDA_N * P_norm
            + LAMBDA_G * PENALTY_FIXED * P_constr
            + LAMBDA_EXC * P_exc
            + LAMBDA_CORR_EXC * P_exc_corr
            + LAMBDA_CONSTR_BAST * P_constr_bast
            + LAMBDA_ANCHORAGE * P_anclaje_ext
            + LAMBDA_DIAM_CONTIG * P_diam_contig
    """
    W = eval_result['total_weight_kg']
    n_slots = eval_result['decoded']['corrido_top'].shape[1]

    # P_capacidad: deficit en cada zona
    P_cap = 0.0
    for v in eval_result['zone_results'].values():
        mu = max(v['Mu'], 0.01)
        deficit = max(0.0, (mu - v['phi_Mn']) / mu * 100.0)
        P_cap += deficit

    # P_normativa
    rho_min = compute_rho_min(fc)
    rho_bot = eval_result['rho_corrido_bot']
    rho_top = eval_result['rho_corrido_top']
    eps_t_pos = eval_result['eps_t_corrido_pos']
    eps_t_neg = eval_result['eps_t_corrido_neg']
    fy = 4200.0
    Es = 2_000_000.0
    eps_y = fy / Es

    P_norm = 0.0
    for rho in (rho_bot, rho_top):
        if rho < rho_min:
            P_norm += (rho_min - rho) / max(rho_min, 1e-9) * 100.0
    for eps_t in (eps_t_pos, eps_t_neg):
        if eps_t < ET_MIN_DUCTILITY:
            P_norm += (ET_MIN_DUCTILITY - eps_t) / ET_MIN_DUCTILITY * 100.0
        if eps_t < eps_y:
            P_norm += (eps_y - eps_t) / eps_y * 100.0

    # P_constructiva
    P_constr = 0.0
    dec = eval_result.get('decoded', {})

    if not eval_result.get('corr_bars_bot'):
        P_constr += 10.0
    if not eval_result.get('corr_bars_top'):
        P_constr += 10.0

    for corr_key in ('corrido_top', 'corrido_bot'):
        corrido_mat = dec.get(corr_key)
        if corrido_mat is None:
            continue
        ns = corrido_mat.shape[1]
        for k in range(N_CAPAS_MAX):
            oni_k = corrido_mat[k, :, 1]
            n_on = int(oni_k.sum())
            if n_on == 0:
                continue
            if oni_k[0] == 0:
                P_constr += 1.0
            if oni_k[ns - 1] == 0:
                P_constr += 1.0
            if n_on == 1:
                P_constr += 1.0

    # P_exceso_bast
    P_exc = 0.0
    for v in eval_result['zone_results'].values():
        if v.get('n_bast', 0) > 0 and v['Mu'] > 0.01:
            exceso = max(0.0, (v['phi_Mn'] - v['Mu']) / v['Mu'] * 100.0)
            P_exc += exceso

    # P_exceso_corrido: por tercio de CADA tramo
    P_exc_corr = 0.0
    phi_corr_pos = eval_result['phi_mn_corrido']['positive']
    phi_corr_neg = eval_result['phi_mn_corrido']['negative']
    zone_extents = eval_result.get('zone_extents', {})

    xs_all = np.array(beam['outputs']['x_m'], dtype=float)
    outs = beam['outputs']
    ms_max_all = np.array(
        outs.get('M_tonf_m_max', outs.get('M_tonf_m', [])), dtype=float)
    ms_min_all = np.array(
        outs.get('M_tonf_m_min', outs.get('M_tonf_m', [])), dtype=float)

    for span_i, sp in enumerate(beam['spans']):
        x0 = sp['x0']
        x1 = sp['x1']
        L_span = sp['L_m']
        L3 = L_span / 3.0

        tercios_x = [
            (x0, x0 + L3),
            (x0 + L3, x0 + 2 * L3),
            (x0 + 2 * L3, x1),
        ]
        tercio_names = ['LEFT', 'MID', 'RIGHT']

        for face, phi_corr in (('BOT', phi_corr_pos), ('TOP', phi_corr_neg)):
            ms_face = ms_max_all if face == 'BOT' else ms_min_all
            for idx, (x_lo, x_hi) in enumerate(tercios_x):
                zone_id = f'{tercio_names[idx]}_{face}_T{span_i + 1}'
                if zone_extents.get(zone_id, {}).get('exists', False):
                    continue
                vals = [
                    max(float(m), 0.0) if face == 'BOT'
                    else max(-float(m), 0.0)
                    for x, m in zip(xs_all, ms_face)
                    if x_lo <= x <= x_hi
                ]
                mu_t = max(vals) if vals else 0.0
                if mu_t > 0.01 and phi_corr > mu_t:
                    P_exc_corr += (phi_corr - mu_t) / mu_t * 100.0

    # P_constr_bast
    P_constr_bast = 0.0
    for v in eval_result['zone_results'].values():
        slots = v.get('bast_slots_per_layer', [0, 0, 0])
        if len(slots) >= 2 and slots[1] > 2:
            P_constr_bast += (slots[1] - 2)
        if len(slots) >= 3:
            P_constr_bast += slots[2] * 2
        n_bast = v.get('n_bast', 0)
        if n_bast > 2:
            P_constr_bast += (n_bast - 2)

    P_anclaje_ext = eval_result.get('P_anclaje_ext', 0.0)

    # P_diam_contig: diametros no contiguos en cada zona
    P_diam = _compute_p_diam_contig(dec, n_slots)

    violations = {
        'P_capacidad': round(P_cap, 4),
        'P_normativa': round(P_norm, 4),
        'P_constructiva': round(P_constr, 4),
        'P_exceso_bast': round(P_exc, 4),
        'P_exceso_corrido': round(P_exc_corr, 4),
        'P_constr_bast': round(P_constr_bast, 4),
        'P_anclaje_ext': round(P_anclaje_ext, 4),
        'P_diam_contig': round(P_diam, 4),
    }
    eval_result['violations'] = violations

    fitness = (W
               + LAMBDA_M * P_cap
               + LAMBDA_N * P_norm
               + LAMBDA_G * P_constr * PENALTY_FIXED
               + LAMBDA_EXC * P_exc
               + LAMBDA_CORR_EXC * P_exc_corr
               + LAMBDA_CONSTR_BAST * P_constr_bast
               + LAMBDA_ANCHORAGE * P_anclaje_ext
               + LAMBDA_DIAM_CONTIG * P_diam)
    return fitness


# ---------------------------------------------------------------------------
# Wrapper combinado
# ---------------------------------------------------------------------------

def evaluate_and_fitness_n(z: np.ndarray, beam: dict,
                           n_slots: int, n_spans: int) -> tuple:
    """Retorna (eval_result, fitness_value)."""
    inp = beam['inputs']
    b = round(float(inp['b_m']), 2)
    h = float(inp['h_m'])
    fc = float(inp['fc_kg_cm2'])
    b_cm = b * 100.0
    d_cm = (h - R1) * 100.0

    ev = eval_individual_n(z, beam, n_slots, n_spans)
    fit = compute_fitness_n(ev, b_cm, d_cm, fc, beam)
    return ev, fit


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import json

    test_path = os.path.join(_HERE, 'test_beam.json')
    if not os.path.exists(test_path):
        print(f"No se encontro {test_path}")
        sys.exit(1)

    with open(test_path, 'r') as f:
        beam = json.load(f)

    from chromosome_n import n_slots_for_beam, chrom_length_n, repair_n

    b = round(beam['inputs']['b_m'], 2)
    n_s = n_slots_for_beam(b)
    n_spans = beam['n_spans']

    print(f"Viga: {beam['id']}, n_spans={n_spans}, b={b}, h={beam['inputs']['h_m']}")
    print(f"n_slots={n_s}, chrom_length={chrom_length_n(n_s, n_spans)}")

    np.random.seed(42)
    L = chrom_length_n(n_s, n_spans)
    z = np.random.randint(0, 6, L, dtype=np.int8)
    z[1::2] = np.random.randint(0, 2, L // 2, dtype=np.int8)

    ev, fit = evaluate_and_fitness_n(z, beam, n_s, n_spans)
    print(f"\nFitness: {fit:.2f}")
    print(f"Peso total: {ev['total_weight_kg']:.2f} kg")
    print(f"  Corrido: {ev['corrido_weight_kg']:.2f} kg")
    print(f"  Bastones: {ev['baston_weight_kg']:.2f} kg")
    print(f"phiMn corrido: pos={ev['phi_mn_corrido']['positive']:.3f}, "
          f"neg={ev['phi_mn_corrido']['negative']:.3f}")
    print(f"Factible: {ev['feasible']}")

    n_ok = sum(1 for v in ev['zone_results'].values() if v['ok'])
    n_total = len(ev['zone_results'])
    print(f"Zonas OK: {n_ok}/{n_total}")

    for zid, zr in ev['zone_results'].items():
        exists = ev['zone_extents'][zid]['exists']
        if exists:
            print(f"  {zid}: phiMn={zr['phi_Mn']:.3f}, Mu={zr['Mu']:.3f}, "
                  f"ok={zr['ok']}, n_bast={zr['n_bast']}")

    if ev['support_detail']:
        print("\nApoyos interiores (fusion parcial):")
        for key, val in ev['support_detail'].items():
            print(f"  {key}: cont={val['n_continue']}, "
                  f"term_R={val['n_terminate_right']}, "
                  f"term_L={val['n_terminate_left']}, ok={val['ok']}")

    print(f"\nViolaciones: {ev['violations']}")
    print("fitness_n.py: test OK")
