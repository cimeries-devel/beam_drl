"""CLI y visualizacion para GA de vigas de N tramos.

Uso:
  # Desde JSON pre-guardado
  python run_inference_n.py --source json --json_beam test_beam.json --fc 210 --save ./figs

  # Desde ETABS (en vivo)
  python run_inference_n.py --source etabs --fc 210 --save ./figs
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from config.config import R1
from config.config_ga import LD_INFERIOR, LD_SUPERIOR, R_LIBRE_CM, D_ESTRIBO_CM, HOOK_TAIL_CM

from beam_n.config_n import (
    generate_zone_ids, parse_zone_id, ensure_joints,
    REBAR_CATALOG_N,
)
from beam_n.chromosome_n import get_combined_bars_n


def _prompt_h_col(beam: dict) -> None:
    """Prompt interactivo para h_col en cada apoyo.

    Escribe beam['joints'] con n_spans+1 entradas {x_m, h_col_m, type}.
    """
    n_spans = int(beam['n_spans'])
    supports_x = list(beam['supports_x_m'])
    n_joints = n_spans + 1

    print("\n" + "=" * 60)
    print("INGRESO DE h_col (ancho de columna en cada apoyo)")
    print("=" * 60)
    print(f"Viga de {n_spans} tramos, {n_joints} apoyos.")
    print("Presione Enter para aceptar el default [0.30 m].\n")

    joints = []
    for i in range(n_joints):
        x = float(supports_x[i]) if i < len(supports_x) else 0.0
        jtype = 'exterior' if (i == 0 or i == n_spans) else 'interior'
        label = f"Apoyo {i} ({jtype:8s}, x={x:6.2f} m)"
        while True:
            raw = input(f"  {label}  h_col [m] (default 0.30): ").strip()
            if raw == '':
                h = 0.30
                break
            try:
                h = float(raw)
                if h <= 0:
                    print("    valor debe ser > 0")
                    continue
                if h < 0.15:
                    print(f"    advertencia: {h} m es muy chico, "
                          f"probablemente infactible")
                break
            except ValueError:
                print("    ingresar un numero valido")
        joints.append({'x_m': x, 'h_col_m': h, 'type': jtype})

    beam['joints'] = joints

    print("\nResumen de apoyos:")
    for j in joints:
        print(f"  x={j['x_m']:6.2f}  h_col={j['h_col_m']:.2f} m  ({j['type']})")

    resp = input("\nConfirmar y arrancar GA? [Y/n]: ").strip().lower()
    if resp == 'n':
        print("Abortado por el usuario.")
        sys.exit(0)

# ---------------------------------------------------------------------------
# Colores constantes
# ---------------------------------------------------------------------------
_CORRIDO_COLOR = '#f5c842'
_BASTON_COLOR = '#e040fb'
_BG_ELEV = '#1a1a2e'
_DIAM_COLORS = {
    '3/8': '#6a1b9a',
    '1/2': '#1565c0',
    '5/8': '#2e7d32',
    '3/4': '#e65100',
    '7/8': '#ad6c00',
    '1':   '#c62828',
}


def _dO_from_name(dname: str) -> int:
    for dO, info in REBAR_CATALOG_N.items():
        if info['name'] == dname:
            return dO
    return 2


def _compute_ld_m(diam_name, fc, face):
    fc_key = int(round(fc))
    table = LD_INFERIOR if face == 'BOT' else LD_SUPERIOR
    if diam_name in table and fc_key in table[diam_name]:
        return table[diam_name][fc_key] / 100.0
    return 0.40


def _bars_spec(bars):
    """Spec string desde lista de (y, area, dname): 'NodX"[+ModY"]'."""
    from collections import Counter
    cnt = Counter(d for _, _, d in bars)
    if not cnt:
        return ''
    parts = [f'{n}o{d}"' for d, n in sorted(cnt.items())]
    return '+'.join(parts)


def _mat_spec(mat):
    """Spec string desde matriz de slots con dO absoluto."""
    if mat is None:
        return ''
    from collections import Counter
    cnt = Counter()
    for c in range(mat.shape[0]):
        for s in range(mat.shape[1]):
            if int(mat[c, s, 1]) == 1:
                dO = int(mat[c, s, 0])
                cnt[REBAR_CATALOG_N[dO]['name']] += 1
    if not cnt:
        return ''
    parts = [f'{n}o{d}"' for d, n in sorted(cnt.items())]
    return '+'.join(parts)


# ---------------------------------------------------------------------------
# Diagrama de momentos
# ---------------------------------------------------------------------------

def _plot_moment_n(ax, beam, result, rank):
    xs = np.array(beam['outputs']['x_m'], dtype=float)
    outs = beam['outputs']
    L_total = beam['spans'][-1]['x1']
    fc = float(beam['inputs']['fc_kg_cm2'])
    supports = beam['supports_x_m']

    ms_max = np.array(outs.get('M_tonf_m_max', outs.get('M_tonf_m', [])))
    ms_min = np.array(outs.get('M_tonf_m_min', outs.get('M_tonf_m', [])))

    phi_pos = result['phi_mn_corrido']['positive']
    phi_neg = result['phi_mn_corrido']['negative']

    # Banda corrido
    ax.fill_between([0, L_total], 0, -phi_pos,
                    color=_CORRIDO_COLOR, alpha=0.20, zorder=1,
                    label=f'phiMn+ corr = {phi_pos:.2f}')
    ax.fill_between([0, L_total], 0, phi_neg,
                    color=_CORRIDO_COLOR, alpha=0.20, zorder=1,
                    label=f'phiMn- corr = {phi_neg:.2f}')
    ax.axhline(-phi_pos, color='#c8900a', lw=1.8, ls='--', alpha=0.9, zorder=3)
    ax.axhline(phi_neg, color='#c8900a', lw=1.8, ls='--', alpha=0.9, zorder=3)

    # Apoyos
    for sx in supports:
        ax.axvline(sx, color='#888', lw=1.0, ls=':', alpha=0.6, zorder=1)

    # Zonas baston
    zone_extents = result.get('zone_extents', {})
    zones_ok = result['zone_results']
    bast_label_added = False

    for zone_id, zinfo in zone_extents.items():
        if not zinfo['exists']:
            continue
        zr = zones_ok.get(zone_id, {})
        if zr.get('n_bast', 0) == 0:
            continue

        parsed = parse_zone_id(zone_id)
        face = parsed['face']
        phi_corr = phi_pos if face == 'BOT' else phi_neg
        phi_z = zr.get('phi_Mn', phi_corr)

        x_s = zinfo['x_start']
        x_e = zinfo['x_end']
        x_ts = zinfo['x_theor_start']
        x_te = zinfo['x_theor_end']

        sign = -1 if face == 'BOT' else 1
        y_base = sign * phi_corr
        y_top_ = sign * phi_z

        bast_mat_z = result['bastones_matrices'].get(zone_id)
        diam_bast = '5/8'
        if bast_mat_z is not None:
            active_dOs = bast_mat_z[:, :, 0][bast_mat_z[:, :, 1] == 1]
            if len(active_dOs) > 0:
                diam_bast = REBAR_CATALOG_N[int(active_dOs.max())]['name']
        ld = _compute_ld_m(diam_bast, fc, face)
        bar_len = max(x_e - x_s, 1e-6)
        ld_phys = min(ld, bar_len * 0.90)

        zone_type = parsed['pos']
        x_rL = x_s + ld_phys
        x_rR = x_e - ld_phys

        if zone_type == 'LEFT':
            xs_trap = np.array([x_s, x_rR, x_e])
            ys_trap = np.array([y_top_, y_top_, y_base])
        elif zone_type == 'RIGHT':
            xs_trap = np.array([x_s, x_rL, x_e])
            ys_trap = np.array([y_base, y_top_, y_top_])
        else:
            xs_trap = np.array([x_s, x_rL, x_rR, x_e])
            ys_trap = np.array([y_base, y_top_, y_top_, y_base])

        # Demanda
        xs_dem = np.linspace(max(x_s, x_ts), min(x_e, x_te), 80)
        ys_dem = np.interp(xs_dem, xs_trap, ys_trap)
        if len(xs_dem) > 1:
            lbl = 'Baston' if not bast_label_added else ''
            ax.fill_between(xs_dem, y_base, ys_dem,
                            color=_BASTON_COLOR, alpha=0.28, zorder=2, label=lbl)
            bast_label_added = True

        # Extension
        if x_ts > x_s + 1e-6:
            xs_ext = np.linspace(x_s, min(x_ts, x_e), 40)
            ys_ext = np.interp(xs_ext, xs_trap, ys_trap)
            ax.fill_between(xs_ext, y_base, ys_ext,
                            color=_BASTON_COLOR, alpha=0.10, zorder=2)
        if x_te < x_e - 1e-6:
            xs_ext = np.linspace(max(x_te, x_s), x_e, 40)
            ys_ext = np.interp(xs_ext, xs_trap, ys_trap)
            ax.fill_between(xs_ext, y_base, ys_ext,
                            color=_BASTON_COLOR, alpha=0.10, zorder=2)

        ax.plot(xs_trap, ys_trap, color=_BASTON_COLOR, lw=1.5, ls='-', zorder=5)

    # Curvas de momento
    ms_max_plot = -ms_max
    ms_min_plot = -ms_min
    ax.fill_between(xs, ms_max_plot, ms_min_plot,
                    color='#d0d0ff', alpha=0.20, zorder=0)
    ax.plot(xs, ms_max_plot, color='#1565c0', lw=1.8, zorder=6, label='Env. M+')
    ax.plot(xs, ms_min_plot, color='#c62828', lw=1.8, zorder=6, label='Env. M-')

    ax.axhline(0, color='#555', lw=0.8)
    # Extender xlim para incluir las caras exteriores de columnas extremas
    joints_local = beam.get('joints', [])
    hc_left = joints_local[0]['h_col_m'] if joints_local else 0.0
    hc_right = joints_local[-1]['h_col_m'] if joints_local else 0.0
    ax.set_xlim(-hc_left / 2.0 - 0.05, L_total + hc_right / 2.0 + 0.05)
    ax.set_xlabel('x (m)', fontsize=9)
    ax.set_ylabel('Momento (tonf-m)', fontsize=9)
    ax.set_title(f'Rank {rank} — Diagrama M(x) con capacidad', fontsize=10,
                 fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Elevacion esquematica
# ---------------------------------------------------------------------------

def _layer_y_m(mat, face, h_m):
    """Retorna dict {capa -> y_m desde el fondo} para barras activas.

    Usa REBAR_CATALOG_N con dO absoluto. Replica la formula de get_active_bars_n.
    """
    from config_ga import S_VERT_MIN_CM as _S
    h_cm = h_m * 100.0
    capas_info = []
    for k in range(mat.shape[0]):
        oni_k = mat[k, :, 1]
        if oni_k.sum() == 0:
            continue
        rep_dO = int(mat[k, oni_k == 1, 0].max())
        db_cm = REBAR_CATALOG_N[rep_dO]['diam_cm']
        capas_info.append((k, db_cm))
    if not capas_info:
        return {}

    y_from_face = []
    y_prev = db_prev = None
    for i, (_, db_cm) in enumerate(capas_info):
        if i == 0:
            y_k = R_LIBRE_CM + D_ESTRIBO_CM + db_cm / 2.0
        else:
            sep = max(db_prev, _S)
            y_k = y_prev + db_prev / 2.0 + sep + db_cm / 2.0
        y_from_face.append(y_k)
        y_prev = y_k
        db_prev = db_cm

    if face == 'TOP':
        y_from_top = y_from_face
    else:
        y_from_top = [h_cm - y for y in y_from_face]

    return {k: (h_cm - yt) / 100.0 for (k, _), yt in zip(capas_info, y_from_top)}


def _draw_anchor(ax, x0, y, face, anchor_decision, direction, color):
    """Dibuja anclaje desde (x0, y) dentro del nudo."""
    if anchor_decision is None or anchor_decision['type'] == 'infeasible':
        ax.plot(x0, y, 'x', color='#ff5252', markersize=10,
                markeredgewidth=2.4, zorder=8)
        return
    L = anchor_decision['length_cm'] / 100.0
    x_end = x0 + direction * L
    ax.plot([x0, x_end], [y, y], color=color, lw=2.5, zorder=8,
            solid_capstyle='round')
    if anchor_decision['type'] == 'hook90':
        tail = anchor_decision['tail_cm'] / 100.0
        y_end = y - tail if face == 'TOP' else y + tail
        ax.plot([x_end, x_end], [y, y_end], color=color, lw=2.8,
                zorder=9, solid_capstyle='round')


def _draw_cota(ax, x0, x1, y, text, color='#e0e0e0', fontsize=6.5,
                tick_size=0.06):
    """Dibuja cota horizontal con ticks y numero centrado."""
    ax.plot([x0, x1], [y, y], color=color, lw=0.6, zorder=4)
    ax.plot([x0, x0], [y - tick_size, y + tick_size],
            color=color, lw=0.6, zorder=4)
    ax.plot([x1, x1], [y - tick_size, y + tick_size],
            color=color, lw=0.6, zorder=4)
    ax.text((x0 + x1) / 2.0, y + tick_size * 1.2, text,
            color=color, fontsize=fontsize, ha='center', va='bottom', zorder=5)


def _draw_break_zigzag(ax, x_center, width, y_top_of_stub, color,
                        direction='up', zigzag_size=0.08):
    """Dibuja marca de break en zigzag, estilo plano de taller."""
    w = width
    z = zigzag_size
    x_l = x_center - w / 2.0
    x_r = x_center + w / 2.0
    sign = 1 if direction == 'up' else -1
    y0 = y_top_of_stub
    # zigzag: 3 picos
    n_peaks = 3
    xs = np.linspace(x_l, x_r, n_peaks * 2 + 1)
    ys = [y0] * len(xs)
    for i, x in enumerate(xs):
        if i == 0 or i == len(xs) - 1:
            ys[i] = y0
        elif i % 2 == 1:
            ys[i] = y0 + sign * z
        else:
            ys[i] = y0 - sign * z * 0.5
    ax.plot(xs, ys, color=color, lw=0.9, zorder=4)


def _plot_elevation_n(ax, beam, result, rank):
    """Elevacion estilo plano de taller.

    Fondo oscuro, lineas cian, cotas arriba/abajo, X marks en cortes,
    ganchos dibujados, aspect ratio 1:1.
    """
    L_total = beam['spans'][-1]['x1']
    supports = list(beam['supports_x_m'])
    joints = beam.get('joints', [])
    if not joints:
        joints = [{'x_m': x, 'h_col_m': 0.30} for x in supports]

    h_m = float(beam['inputs']['h_m'])
    n_spans = beam['n_spans']

    # Paleta estilo plano (colores por defecto matplotlib)
    BG = 'white'
    LINE = 'black'
    TEXT = 'black'
    X_MARK = '#c62828'
    RED = '#c62828'

    # Altura de stub de columna (porcion visible arriba/abajo de la viga).
    col_stub = max(h_m * 0.60, 0.25)
    cota_up_y = h_m + col_stub + 0.40
    cota_dn_y = -col_stub - 0.40

    ax.set_facecolor(BG)
    # Mismo rango x que el diagrama de momentos (ax0) — extendido para mostrar
    # caras exteriores y anclajes que se meten en las columnas extremas.
    hc_left = joints[0]['h_col_m']
    hc_right = joints[-1]['h_col_m']
    ax.set_xlim(-hc_left / 2.0 - 0.05, L_total + hc_right / 2.0 + 0.05)
    ax.set_ylim(cota_dn_y - 0.20, cota_up_y + 0.20)
    ax.axis('off')

    # --- Columnas: rectangulo vertical continuo que cruza la viga ---
    # (se dibujan primero para que la viga comparta sus caras laterales)
    for j in joints:
        x_c = j['x_m']
        hc = j['h_col_m']
        ax.add_patch(patches.Rectangle(
            (x_c - hc / 2, -col_stub), hc, h_m + 2 * col_stub,
            linewidth=1.2, edgecolor=LINE, facecolor='none', zorder=3))
        # Break zigzag al tope superior e inferior del stub
        _draw_break_zigzag(ax, x_c, hc, h_m + col_stub, LINE,
                            direction='up', zigzag_size=col_stub * 0.35)
        _draw_break_zigzag(ax, x_c, hc, -col_stub, LINE,
                            direction='down', zigzag_size=col_stub * 0.35)

    # --- Viga: tramos de cara interior a cara interior de las columnas ---
    for k in range(n_spans):
        x_left = joints[k]['x_m'] + joints[k]['h_col_m'] / 2.0
        x_right = joints[k + 1]['x_m'] - joints[k + 1]['h_col_m'] / 2.0
        # Solo borde superior e inferior (las verticales las da la columna)
        ax.plot([x_left, x_right], [0, 0],
                color=LINE, lw=1.2, zorder=3)
        ax.plot([x_left, x_right], [h_m, h_m],
                color=LINE, lw=1.2, zorder=3)

    # --- Estribos: lineas verticales densas ---
    stirrup_sp = max(0.12, h_m * 0.5)
    x = 0.0
    while x <= L_total + 1e-6:
        # saltar si dentro de columna
        in_col = any(
            abs(x - j['x_m']) <= j['h_col_m'] / 2.0 + 1e-6
            for j in joints
        )
        if not in_col and 1e-6 < x < L_total - 1e-6:
            ax.plot([x, x], [h_m * 0.08, h_m * 0.92],
                    color=LINE, lw=0.45, alpha=0.55, zorder=4)
        x += stirrup_sp

    # --- Datos del diseño ---
    corr_top_mat = result['corrido_top_matrix']
    corr_bot_mat = result['corrido_bot_matrix']
    bast_mats = result['bastones_matrices']
    corr_anc = result.get('corrido_anchorage', {'TOP': [], 'BOT': []})
    support_cells = result.get('support_cells', {})
    zone_extents = result.get('zone_extents', {})
    zone_results = result['zone_results']

    y_top_by_capa = _layer_y_m(corr_top_mat, 'TOP', h_m)
    y_bot_by_capa = _layer_y_m(corr_bot_mat, 'BOT', h_m)

    color_corr = LINE
    color_bast = LINE

    # --- Corrido: una linea por capa (con label 'NφX"') ---
    # La barra se traza de cara a cara de las columnas extremas.
    # El anclaje arranca en la cara y se mete HACIA ADENTRO de la columna.
    x_face_L0 = joints[0]['x_m'] + joints[0]['h_col_m'] / 2.0
    x_face_RN = joints[-1]['x_m'] - joints[-1]['h_col_m'] / 2.0

    def _draw_corrido_line(face, y_by_capa, corr_mat, anc_entries):
        by_capa = {}
        for entry in anc_entries:
            by_capa.setdefault(entry['capa'], []).append(entry)
        for c, entries in by_capa.items():
            y = y_by_capa.get(c)
            if y is None:
                continue
            ax.plot([x_face_L0, x_face_RN], [y, y],
                    color=color_corr, lw=1.4, zorder=6)
            _draw_anchor(ax, x_face_L0, y, face, entries[0]['left'], -1,
                         color_corr)
            _draw_anchor(ax, x_face_RN, y, face, entries[0]['right'], +1,
                         color_corr)
            # Label NφX" usando los nombres directos de cada barra
            from collections import Counter
            cnt = Counter(e['diam'] for e in entries)
            lbl = '+'.join(f'{n}φ{d}"' for d, n in sorted(cnt.items()))
            y_lbl = y + (0.025 if face == 'TOP' else -0.025)
            va = 'bottom' if face == 'TOP' else 'top'
            sp = beam['spans'][0]
            x_lbl = (sp['x0'] + sp['x1']) / 2.0
            ax.text(x_lbl, y_lbl, lbl, color=LINE, fontsize=7,
                    ha='center', va=va, zorder=10, fontweight='normal')

    _draw_corrido_line('TOP', y_top_by_capa, corr_top_mat, corr_anc['TOP'])
    _draw_corrido_line('BOT', y_bot_by_capa, corr_bot_mat, corr_anc['BOT'])

    # --- Bastones ---
    for zone_id in generate_zone_ids(n_spans):
        zr = zone_results.get(zone_id, {})
        if zr.get('n_bast', 0) == 0:
            continue
        parsed = parse_zone_id(zone_id)
        face = parsed['face']
        pos = parsed['pos']
        span = parsed['span']

        zinfo = zone_extents.get(zone_id, {})
        x0 = zinfo.get('x_start', 0.0)
        x1 = zinfo.get('x_end', L_total)
        x_ts = zinfo.get('x_theor_start', x0)
        x_te = zinfo.get('x_theor_end', x1)

        bast_mat = bast_mats.get(zone_id)
        if bast_mat is None:
            continue
        y_by_capa = (y_top_by_capa if face == 'TOP' else y_bot_by_capa)
        y_bast = _layer_y_m(bast_mat, face, h_m)
        y_by_capa = {**y_bast, **y_by_capa}

        # Ligero offset para no solapar con corrido
        offs = 0.015 if face == 'TOP' else -0.015

        for c in range(bast_mat.shape[0]):
            for s in range(bast_mat.shape[1]):
                if int(bast_mat[c, s, 1]) == 0:
                    continue
                y = y_by_capa.get(c)
                if y is None:
                    continue
                y_draw = y + offs

                # Por defecto, extremos del baston = zona
                x_bar_L = x0
                x_bar_R = x1

                # Si el baston toca un apoyo interior, chequear si
                # pasa o termina (y cobrar anclaje si termina)
                key_int = None  # (joint_index)
                if pos == 'LEFT' and span >= 2:
                    key_int = span - 1  # apoyo interior izq del tramo
                    side = 'L'
                elif pos == 'RIGHT' and span <= n_spans - 1:
                    key_int = span  # apoyo interior der del tramo
                    side = 'R'

                if key_int is not None:
                    support_key = f'{key_int}_{face}'
                    cells = support_cells.get(support_key, [])
                    cell = next((cc for cc in cells
                                 if cc['capa'] == c and cc['slot'] == s), None)
                    if cell is not None:
                        hc_int = float(joints[key_int]['h_col_m'])
                        x_joint = supports[key_int]
                        x_face_inner = (x_joint + hc_int / 2.0
                                        if side == 'L'
                                        else x_joint - hc_int / 2.0)
                        x_face_outer = (x_joint - hc_int / 2.0
                                        if side == 'L'
                                        else x_joint + hc_int / 2.0)
                        if cell['kind'] == 'continue':
                            # Pasante: la barra cruza el nudo entero.
                            # Extiende el extremo del baston hasta la cara
                            # opuesta de la columna (sale a tramo vecino).
                            if side == 'L':
                                x_bar_L = x_face_outer
                            else:
                                x_bar_R = x_face_outer
                        else:
                            # Termina dentro del nudo — anclaje desde la cara
                            # interior hacia adentro de la columna.
                            anc = cell.get('anchor')
                            if side == 'L':
                                x_bar_L = x_face_inner
                                _draw_anchor(ax, x_bar_L, y_draw, face,
                                             anc, -1, color_bast)
                            else:
                                x_bar_R = x_face_inner
                                _draw_anchor(ax, x_bar_R, y_draw, face,
                                             anc, +1, color_bast)

                ax.plot([x_bar_L, x_bar_R], [y_draw, y_draw],
                        color=color_bast, lw=1.6, zorder=6)

        # --- Marcas X grandes en cortes teoricos ---
        for x_cut in (x_ts, x_te):
            in_joint = any(
                abs(x_cut - j['x_m']) <= j['h_col_m'] / 2.0 + 1e-3
                for j in joints
            )
            if in_joint:
                continue
            y_mark = h_m + col_stub * 0.25 if face == 'TOP' else -col_stub * 0.25
            ax.text(x_cut, y_mark, 'X', color=X_MARK,
                    fontsize=12, fontweight='bold',
                    ha='center', va='center', zorder=10)

        # --- Label NφX" sobre bastones ---
        spec = _mat_spec(bast_mat)
        if spec:
            y_label = (h_m + col_stub * 0.6 if face == 'TOP'
                       else -col_stub * 0.6)
            va = 'center'
            ax.text((x_ts + x_te) / 2.0, y_label, spec,
                    color=LINE, fontsize=6.5, ha='center', va=va,
                    zorder=9)

    # --- Cotas arriba: longitudes de bastones (x_theor_start..x_theor_end) ---
    # Buscar todos los pares (x_ts, x_te) de zonas TOP activas y dibujar
    y_cota_top = h_m + col_stub + 0.22
    for zone_id in generate_zone_ids(n_spans):
        parsed = parse_zone_id(zone_id)
        if parsed['face'] != 'TOP':
            continue
        zr = zone_results.get(zone_id, {})
        if zr.get('n_bast', 0) == 0:
            continue
        zinfo = zone_extents.get(zone_id, {})
        x_ts = zinfo.get('x_theor_start', 0.0)
        x_te = zinfo.get('x_theor_end', 0.0)
        if x_te - x_ts < 1e-3:
            continue
        _draw_cota(ax, x_ts, x_te, y_cota_top, f'{x_te - x_ts:.2f}',
                   color=TEXT)

    # --- Cotas abajo: longitudes de tramos ---
    y_cota_bot = -col_stub - 0.22
    for sp in beam['spans']:
        _draw_cota(ax, sp['x0'], sp['x1'], y_cota_bot,
                   f'{sp["L_m"]:.2f}', color=TEXT)

    # --- h_col como cota corta junto a columnas ---
    for j in joints:
        x_c = j['x_m']
        hc = j['h_col_m']
        _draw_cota(ax, x_c - hc / 2, x_c + hc / 2,
                   y_cota_bot, f'{hc:.2f}', color=TEXT, fontsize=5.5,
                   tick_size=col_stub * 0.08)


def _spec_table_axes(ax, beam, result):
    """Tabla de spec por tramo (corrido + bastones por cara)."""
    ax.axis('off')
    ax.set_title('Spec por tramo', fontsize=8, pad=2, loc='left')
    corr_top_bars = result.get('corrido_top_bars', [])
    corr_bot_bars = result.get('corrido_bot_bars', [])
    spec_corr_top = _bars_spec(corr_top_bars)
    spec_corr_bot = _bars_spec(corr_bot_bars)
    bast_mats = result['bastones_matrices']

    n_spans = beam['n_spans']
    lines = [f'CORRIDO  TOP: {spec_corr_top}    BOT: {spec_corr_bot}']
    for i in range(n_spans):
        parts = []
        for pos in ('LEFT', 'MID', 'RIGHT'):
            for face in ('TOP', 'BOT'):
                zid = f'{pos}_{face}_T{i + 1}'
                sp = _mat_spec(bast_mats.get(zid))
                if sp:
                    parts.append(f'{pos[0]}{face[0]}:{sp}')
        txt = f'T{i+1}  ' + ('  |  '.join(parts) if parts else '(solo corrido)')
        lines.append(txt)

    ax.text(0.01, 0.95, '\n'.join(lines),
            family='monospace', fontsize=7, va='top', ha='left',
            transform=ax.transAxes, color='#222')


# ---------------------------------------------------------------------------
# Seccion transversal
# ---------------------------------------------------------------------------

def _draw_section_n(ax, beam, result, span_idx, tercio_idx):
    """Dibuja seccion transversal para un tramo y posicion (LEFT/MID/RIGHT)."""
    sec_names = ['LEFT', 'MID', 'RIGHT']
    sec_name = sec_names[tercio_idx]
    zone_suffix = f'_T{span_idx + 1}'
    n_spans = beam['n_spans']

    sp = beam['spans'][span_idx]
    b = round(float(sp.get('b_m', beam['inputs']['b_m'])), 2)
    h = float(sp.get('h_m', beam['inputs']['h_m']))
    h_cm = h * 100.0
    b_cm = b * 100.0

    corrido_top_mat = result['corrido_top_matrix']
    corrido_bot_mat = result['corrido_bot_matrix']
    bast_mats = result['bastones_matrices']
    n_slots = corrido_top_mat.shape[1]

    # Corner radius from capa 0 representative bar (largest active dO in capa 0)
    capa0_active = corrido_top_mat[0, corrido_top_mat[0, :, 1] == 1, 0]
    dO_corner = int(capa0_active.max()) if len(capa0_active) > 0 else 2
    r_corner = REBAR_CATALOG_N[dO_corner]['diam_cm'] / 2.0
    margin = R_LIBRE_CM + D_ESTRIBO_CM + r_corner
    slot_xs = np.linspace(margin, b_cm - margin, n_slots)

    ax.set_facecolor('white')
    ax.add_patch(patches.Rectangle(
        (0, 0), b_cm, h_cm,
        linewidth=1.5, edgecolor='#333', facecolor='#f8f8f8'))

    s_off = R_LIBRE_CM + D_ESTRIBO_CM / 2.0
    ax.add_patch(patches.Rectangle(
        (s_off, s_off), b_cm - 2 * s_off, h_cm - 2 * s_off,
        linewidth=1.8, edgecolor='#444', facecolor='none',
        joinstyle='round', zorder=4))

    legend_keys = {}

    def _draw_face(face):
        corrido_mat = corrido_top_mat if face == 'TOP' else corrido_bot_mat
        bast_key = f'{sec_name}_{face}{zone_suffix}'
        bast_mat = bast_mats.get(bast_key, np.zeros_like(corrido_mat))

        combined = get_combined_bars_n(corrido_mat, bast_mat, face, h_cm)
        for y_top, area, dname, bar_type, capa_k, slot_idx in combined:
            db_cm = REBAR_CATALOG_N[_dO_from_name(dname)]['diam_cm']
            r = db_cm / 2.0
            x = slot_xs[slot_idx]
            y_plot = h_cm - y_top
            color = _DIAM_COLORS.get(dname, '#aaa')
            filled = (bar_type == 'corrido')

            if filled:
                ax.add_patch(plt.Circle((x, y_plot), r,
                                        color=color, zorder=5, alpha=0.92))
            else:
                ax.add_patch(plt.Circle((x, y_plot), r,
                                        fill=False, edgecolor=color,
                                        lw=2.2, zorder=5))
            legend_keys.setdefault((dname, filled), (color, filled))

    _draw_face('TOP')
    _draw_face('BOT')

    ax.set_xlim(-2, b_cm + 2)
    ax.set_ylim(-2, h_cm + 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Legend
    if legend_keys:
        handles = []
        for (dname, filled), (color, _) in sorted(legend_keys.items()):
            tipo = 'corr' if filled else 'bast'
            h_ = patches.Patch(
                facecolor=color if filled else 'none',
                edgecolor=color, linewidth=1.5,
                label=f'o{dname}" {tipo}')
            handles.append(h_)
        ax.legend(handles=handles, fontsize=5,
                  loc='upper center', bbox_to_anchor=(0.5, -0.02),
                  ncol=min(len(handles), 4), framealpha=0.75)

    zbot_id = f'{sec_name}_BOT{zone_suffix}'
    ztop_id = f'{sec_name}_TOP{zone_suffix}'
    zbot_res = result['zone_results'].get(zbot_id, {})
    ztop_res = result['zone_results'].get(ztop_id, {})
    ok_bot = 'SI' if zbot_res.get('ok') else 'NO'
    ok_top = 'SI' if ztop_res.get('ok') else 'NO'
    title = (f'T{span_idx+1} {sec_name} | B:{ok_bot} T:{ok_top}\n'
             f'+{zbot_res.get("phi_Mn",0):.2f}/-{ztop_res.get("phi_Mn",0):.2f}')
    ax.set_title(title, fontsize=6.5, pad=3)


# ---------------------------------------------------------------------------
# Generar figura completa
# ---------------------------------------------------------------------------

def generate_figure_n(beam, result, rank, save_dir=None):
    n_spans = beam['n_spans']
    n_rows = 3 + n_spans
    # Escalar ancho en funcion del largo de la viga para mantener 1:1
    L_total = beam['spans'][-1]['x1']
    fig_w = max(16, min(30, L_total * 1.8))
    spec_h = 0.35 + 0.08 * n_spans
    height_ratios = [1.2, 1.4, spec_h] + [1.0] * n_spans
    fig_h = 4.5 + 2.5 * n_spans

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(n_rows, 3, figure=fig,
                  height_ratios=height_ratios,
                  hspace=0.55, wspace=0.3)

    # Fila 0: momentos
    ax0 = fig.add_subplot(gs[0, :])
    _plot_moment_n(ax0, beam, result, rank)

    # Fila 1: elevacion detallada
    ax1 = fig.add_subplot(gs[1, :])
    _plot_elevation_n(ax1, beam, result, rank)

    # Fila 2: tabla spec por tramo
    ax2 = fig.add_subplot(gs[2, :])
    _spec_table_axes(ax2, beam, result)

    # Filas 3+: secciones por tramo
    for si in range(n_spans):
        for col in range(3):
            ax_s = fig.add_subplot(gs[3 + si, col])
            _draw_section_n(ax_s, beam, result, span_idx=si, tercio_idx=col)

    # Titulo
    beam_id = beam['id']
    b = beam['inputs']['b_m']
    h = beam['inputs']['h_m']
    fc = beam['inputs']['fc_kg_cm2']
    L_total = beam['spans'][-1]['x1']

    zones_ok = result['zones_ok']
    n_zones = len(result['zone_results'])
    feasible = 'Factible' if result['feasible'] else 'NO factible'
    fig.suptitle(
        f'GA N-Tramos — Viga {beam_id} | '
        f'b={b}m h={h}m fc={fc} L={L_total:.2f}m | '
        f'{n_spans} tramos | Rank {rank} | '
        f'W={result["total_weight_kg"]:.2f}kg | '
        f'{zones_ok}/{n_zones} OK | {feasible}',
        fontsize=11, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f'ga_ntramos_{beam_id}_rank{rank}.png'
        path = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  Figura guardada: {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura de convergencia
# ---------------------------------------------------------------------------

def plot_convergence(result_ga, beam_id, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    gens = range(1, len(result_ga['history_best']) + 1)
    ax.plot(gens, result_ga['history_best'], 'b-', lw=1.5, label='Best')
    ax.plot(gens, result_ga['history_mean'], 'g--', lw=1.0, label='Mean')
    ax.set_xlabel('Generacion')
    ax.set_ylabel('Fitness')
    ax.set_title(f'Convergencia — Viga {beam_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'convergencia_{beam_id}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  Convergencia guardada: {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(beam, result_ga, path):
    data = {
        'beam_id': beam['id'],
        'n_spans': beam['n_spans'],
        'b': beam['inputs']['b_m'],
        'h': beam['inputs']['h_m'],
        'fc': beam['inputs']['fc_kg_cm2'],
        'supports_x_m': beam['supports_x_m'],
        'spans': beam['spans'],
        'top3': [],
        'generations_run': result_ga['generations_run'],
        'elapsed_s': result_ga['elapsed_s'],
    }

    for r in result_ga['top3']:
        entry = {
            'rank': r['rank'],
            'corrido_top': _bars_spec(r['corrido_top_bars']),
            'corrido_bot': _bars_spec(r['corrido_bot_bars']),
            'total_weight_kg': r['total_weight_kg'],
            'corrido_weight_kg': r['corrido_weight_kg'],
            'baston_weight_kg': r['baston_weight_kg'],
            'feasible': r['feasible'],
            'zones_ok': r['zones_ok'],
            'zones_total': len(r['zone_results']),
        }
        # Zonas activas
        active_zones = {}
        for zid, zr in r['zone_results'].items():
            ze = r['zone_extents'].get(zid, {})
            if ze.get('exists', False):
                active_zones[zid] = {
                    'phi_Mn': zr['phi_Mn'],
                    'Mu': zr['Mu'],
                    'ok': zr['ok'],
                    'n_bast': zr['n_bast'],
                    'weight_kg': zr['weight_kg'],
                }
        entry['active_zones'] = active_zones
        data['top3'].append(entry)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f'JSON exportado: {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='GA vigas N tramos')
    parser.add_argument('--source', choices=['etabs', 'json'], default='json')
    parser.add_argument('--json_beam', type=str, default=None,
                        help='Path al JSON del beam dict')
    parser.add_argument('--fc', type=float, default=210.0)
    parser.add_argument('--combo', type=str, default='ENV')
    parser.add_argument('--item_type_elm', type=int, default=0)
    parser.add_argument('--pop', type=int, default=None)
    parser.add_argument('--gen', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--corrido_simetrico', action='store_true', default=True)
    parser.add_argument('--corrido_asimetrico', action='store_true', default=False)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--json_out', type=str, default=None)
    parser.add_argument('--h_col_default', action='store_true', default=False,
                        help='Usa h_col=0.30 en todos los apoyos sin prompt')
    args = parser.parse_args()

    corrido_sim = not args.corrido_asimetrico

    # Cargar beam
    if args.source == 'etabs':
        from beam_n.etabs_extractor_n import extract_multispan_beam
        beam = extract_multispan_beam(
            fc_kg_cm2=args.fc,
            item_type_elm=args.item_type_elm,
            combo=args.combo,
        )
        print(f"Extraido de ETABS: {beam['id']}, {beam['n_spans']} tramos")
    else:
        if not args.json_beam:
            # Default al test_beam.json
            args.json_beam = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_beam.json')
        with open(args.json_beam, 'r', encoding='utf-8') as f:
            beam = json.load(f)
        print(f"Cargado: {beam['id']}, {beam['n_spans']} tramos")

    # Joints / h_col
    ensure_joints(beam)
    if args.h_col_default:
        print("h_col_default activo -> 0.30 m en todos los apoyos")
    else:
        _prompt_h_col(beam)

    # Info
    print(f"Seccion: b={beam['inputs']['b_m']}m x h={beam['inputs']['h_m']}m")
    print(f"fc: {beam['inputs']['fc_kg_cm2']} kg/cm2")
    print(f"Apoyos: {beam['supports_x_m']}")
    for i, sp in enumerate(beam['spans']):
        print(f"  T{i+1}: frame={sp['frame']}, L={sp['L_m']:.3f}m")

    # GA
    from beam_n.beam_ga_n import BeamGANSpans
    ga = BeamGANSpans(
        beam,
        pop_size=args.pop,
        n_gen=args.gen,
        seed=args.seed,
        corrido_simetrico=corrido_sim,
    )
    print(f"\nGA: pop={ga.pop_size}, gen={ga.n_gen}, "
          f"chrom={ga.chrom_L} genes, {ga.n_zones} zonas")
    print(f"Corrido simetrico: {corrido_sim}")
    print()

    result_ga = ga.run()

    # Resultados
    print(f"\n{'='*60}")
    print(f"Generaciones: {result_ga['generations_run']}, "
          f"tiempo: {result_ga['elapsed_s']}s")
    for r in result_ga['top3']:
        print(f"\nRank {r['rank']}: W={r['total_weight_kg']:.2f}kg, "
              f"feasible={r['feasible']}, "
              f"{r['zones_ok']}/{len(r['zone_results'])} zonas OK")
        print(f"  Corrido TOP: {_bars_spec(r['corrido_top_bars'])}  "
              f"BOT: {_bars_spec(r['corrido_bot_bars'])}")

    # Figuras
    if args.save:
        for r in result_ga['top3']:
            generate_figure_n(beam, r, r['rank'], save_dir=args.save)
        plot_convergence(result_ga, beam['id'], save_dir=args.save)

    # JSON
    if args.json_out:
        export_json(beam, result_ga, args.json_out)


if __name__ == '__main__':
    main()
