"""Funciones de visualizacion para el GA de vigas de N tramos."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from config import REBAR_CATALOG
from config_ga import R_LIBRE_CM, D_ESTRIBO_CM
from chromosome import get_combined_bars, bar_y_positions
from config_n import parse_zone_id

# ---------------------------------------------------------------------------
# Colores constantes
# ---------------------------------------------------------------------------
_CORRIDO_COLOR = '#f5c842'
_BASTON_COLOR = '#e040fb'
_BG_ELEV = '#1a1a2e'
_DIAM_COLORS = {
    '1/2': '#1565c0',
    '5/8': '#2e7d32',
    '3/4': '#e65100',
    '1':   '#c62828',
}

def _name2idx(dname: str) -> int:
    for idx, info in REBAR_CATALOG.items():
        if info['name'] == dname:
            return idx
    return 0

def _compute_ld_m(diam_name, fc, face):
    from config_compat import LD_INFERIOR, LD_SUPERIOR
    fc_key = int(round(fc))
    table = LD_INFERIOR if face == 'BOT' else LD_SUPERIOR
    if diam_name in table and fc_key in table[diam_name]:
        return table[diam_name][fc_key] / 100.0
    return 0.40

def _corr_spec(bars, dA_name, dB_name):
    n_A = sum(1 for _, _, d in bars if d == dA_name)
    n_B = sum(1 for _, _, d in bars if d == dB_name)
    if dA_name == dB_name or n_B == 0:
        return f'{len(bars)}o{dA_name}"'
    if n_A == 0:
        return f'{len(bars)}o{dB_name}"'
    return f'{n_A}o{dA_name}"+{n_B}o{dB_name}"'

def _bast_spec(mat, dA_name, dB_name):
    if mat is None:
        return ''
    oni = mat[:, :, 1]
    cho = mat[:, :, 0]
    n_A = int((oni * (1 - cho)).sum())
    n_B = int((oni * cho).sum())
    if n_A > 0 and n_B > 0:
        return f'{n_A}o{dA_name}"+{n_B}o{dB_name}"'
    elif n_B > 0:
        return f'{n_B}o{dB_name}"'
    elif n_A > 0:
        return f'{n_A}o{dA_name}"'
    return ''

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

        diam_bast = result.get('diam_B', '5/8')
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
                            color=_BASTON_COLOR, alpha=0.35, zorder=2, label=lbl)
            bast_label_added = True

    # Momentos actuantes
    ax.plot(xs, ms_max, color='#3498db', lw=1.3, label='M max', zorder=5)
    ax.plot(xs, ms_min, color='#e74c3c', lw=1.3, label='M min', zorder=5)

    ax.set_title(f"Rank {rank} - Capacidad vs Demanda (tonf-m)", fontsize=10, color='white')
    ax.set_facecolor('#0f0f1b')
    ax.grid(True, color='#444', alpha=0.3)
    ax.legend(fontsize=7, loc='lower right', facecolor='#1a1a2e', labelcolor='white')
    ax.tick_params(colors='white', labelsize=8)

def _layer_y_m(mat, face, dA_idx, dB_idx, h_m):
    """Calcula la posicion Y real de cada capa en metros para visualizacion."""
    n_capas, n_slots, _ = mat.shape
    r_libre_m = R_LIBRE_CM / 100.0
    d_est_m = D_ESTRIBO_CM / 100.0

    diam_A_m = REBAR_CATALOG[dA_idx]['diam_cm'] / 100.0
    diam_B_m = REBAR_CATALOG[dB_idx]['diam_cm'] / 100.0

    y_pos = []
    current_y = r_libre_m + d_est_m
    for c in range(n_capas):
        # Diametro dominante de la capa
        oni = mat[c, :, 1]
        cho = mat[c, :, 0]
        if oni.sum() == 0:
            y_pos.append(0)
            continue

        has_B = (oni * cho).sum() > 0
        d_capa = diam_B_m if has_B else diam_A_m
        y_center = current_y + d_capa / 2.0
        y_pos.append(y_center if face == 'BOT' else h_m - y_center)
        current_y += d_capa + 2.54 / 100.0  # 1 inch de separacion

    return np.array(y_pos)

def _draw_anchor(ax, x0, y, face, anchor_decision, direction, color):
    """Dibuja el gancho de anclaje (hook)."""
    from config_ga import HOOK_TAIL_CM
    tail = HOOK_TAIL_CM / 100.0
    if anchor_decision == 1:  # HOOK
        sign = 1 if face == 'BOT' else -1
        dx = 0.05 * direction
        ax.plot([x0, x0, x0 + dx], [y, y + sign * tail, y + sign * tail],
                color=color, lw=1.5, solid_capstyle='round')

def _draw_cota(ax, x0, x1, y, text, color='#e0e0e0', fontsize=6.5, tick_size=0.06):
    ax.plot([x0, x1], [y, y], color=color, lw=0.6, alpha=0.8)
    ax.plot([x0, x0], [y - tick_size/2, y + tick_size/2], color=color, lw=0.6)
    ax.plot([x1, x1], [y - tick_size/2, y + tick_size/2], color=color, lw=0.6)
    xm = (x0 + x1) / 2
    ax.text(xm, y + 0.02, text, color=color, fontsize=fontsize,
            ha='center', va='bottom', fontweight='light')

def _draw_break_zigzag(ax, x_center, width, y_top_of_stub, color, direction='up', zigzag_size=0.08):
    """Dibuja un corte zigzag para columnas."""
    x_left = x_center - width / 2
    x_right = x_center + width / 2
    y_base = y_top_of_stub
    y_peak = y_base + (zigzag_size if direction == 'up' else -zigzag_size)

    pts_x = [x_left, x_left + width * 0.25, x_left + width * 0.75, x_right]
    pts_y = [y_base, y_peak, y_base - (y_peak - y_base), y_base]

    ax.plot(pts_x, pts_y, color=color, lw=0.8, alpha=0.7)

def _plot_elevation_n(ax, beam, result, rank):
    h_m = beam['inputs']['h_m']
    L_total = beam['spans'][-1]['x1']
    spans = beam['spans']
    n_spans = len(spans)
    joints = beam.get('joints', [])

    # Dibujar vigas y columnas
    for i, sp in enumerate(spans):
        rect = patches.Rectangle((sp['x0'], 0), sp['L_m'], h_m,
                                 linewidth=1, edgecolor='#444', facecolor='#252540', alpha=0.3)
        ax.add_patch(rect)

    for j in joints:
        x_c = j['x_m']
        w_c = j['h_col_m']
        # Stub inferior
        ax.add_patch(patches.Rectangle((x_c - w_c / 2, -0.6), w_c, 0.6,
                                       facecolor='#252540', alpha=0.2))
        _draw_break_zigzag(ax, x_c, w_c, -0.6, '#444', 'down')
        # Stub superior
        ax.add_patch(patches.Rectangle((x_c - w_c / 2, h_m), w_c, 0.6,
                                       facecolor='#252540', alpha=0.2))
        _draw_break_zigzag(ax, x_c, w_c, h_m + 0.6, '#444', 'up')

    # Barras Corridas
    dA_idx = _name2idx(result['diam_A'])
    dB_idx = _name2idx(result['diam_B'])

    def _draw_corrido_line(face, y_by_capa, corr_mat, anc_entries):
        for c in range(corr_mat.shape[0]):
            oni = corr_mat[c, :, 1]
            if oni.sum() == 0: continue
            y = y_by_capa[c]
            # Simplificado: asume una sola barra larga
            x0, x1 = 0, L_total
            ax.plot([x0, x1], [y, y], color=_CORRIDO_COLOR, lw=1.8, zorder=10)
            # Anclajes extremos
            _draw_anchor(ax, x0, y, face, anc_entries[0], 1, _CORRIDO_COLOR)
            _draw_anchor(ax, x1, y, face, anc_entries[-1], -1, _CORRIDO_COLOR)

    y_bot = _layer_y_m(result['mat_bot_corr'], 'BOT', dA_idx, dB_idx, h_m)
    _draw_corrido_line('BOT', y_bot, result['mat_bot_corr'], result['anchors_bot'])

    y_top = _layer_y_m(result['mat_top_corr'], 'TOP', dA_idx, dB_idx, h_m)
    _draw_corrido_line('TOP', y_top, result['mat_top_corr'], result['anchors_top'])

    # Bastones
    zone_extents = result.get('zone_extents', {})
    for zone_id, zinfo in zone_extents.items():
        if not zinfo['exists']: continue
        zr = result['zone_results'].get(zone_id, {})
        if zr.get('n_bast', 0) == 0: continue

        parsed = parse_zone_id(zone_id)
        face = parsed['face']
        mat = zr['mat_bast']
        y_capas = _layer_y_m(mat, face, dA_idx, dB_idx, h_m)

        for c in range(mat.shape[0]):
            oni = mat[c, :, 1]
            if oni.sum() == 0: continue
            y = y_capas[c]
            ax.plot([zinfo['x_start'], zinfo['x_end']], [y, y],
                    color=_BASTON_COLOR, lw=1.8, zorder=11)

    ax.set_xlim(-0.5, L_total + 0.5)
    ax.set_ylim(-1.0, h_m + 1.0)
    ax.set_aspect('equal')
    ax.set_facecolor(_BG_ELEV)
    ax.set_title(f"Elevacion - Viga {beam['id']}", color='white', fontsize=10)
    ax.axis('off')

def _spec_table_axes(ax, beam, result):
    ax.axis('off')
    dA, dB = result['diam_A'], result['diam_B']
    n_spans = beam['n_spans']

    # Header
    props = dict(boxstyle='round', facecolor='#252540', alpha=0.5)
    txt = f"RESUMEN DE ARMADO (Rank {result['rank']})\n"
    txt += f"Diametros: A={dA}\", B={dB}\"\n"
    txt += f"Peso Total: {result['total_weight_kg']:.2f} kg\n"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, color='white',
            fontsize=9, verticalalignment='top', bbox=props)

    # Detalle por tramo
    y_cursor = 0.70
    for i in range(n_spans):
        ax.text(0.05, y_cursor, f"TRAMO {i+1}:", transform=ax.transAxes,
                color=_CORRIDO_COLOR, fontsize=8, fontweight='bold')
        y_cursor -= 0.06
        # Info de bastones... (podria expandirse)
        y_cursor -= 0.05

def _draw_section_n(ax, beam, result, span_idx, tercio_idx):
    """Dibuja una seccion transversal en un punto especifico."""
    b_m = beam['inputs']['b_m']
    h_m = beam['inputs']['h_m']
    b_cm = b_m * 100
    h_cm = h_m * 100

    ax.add_patch(patches.Rectangle((0, 0), b_cm, h_cm, facecolor='#2c2c3e', edgecolor='#666', lw=2))

    # Estribo
    r = R_LIBRE_CM
    ax.add_patch(patches.Rectangle((r, r), b_cm - 2*r, h_cm - 2*r,
                                   fill=False, edgecolor='#888', lw=1.5, ls='--'))

    # Reutilizar logica de chromosome para obtener posiciones de barras
    # (Simplificado para este ejemplo de refactorizacion)
    ax.set_xlim(-5, b_cm + 5)
    ax.set_ylim(-5, h_cm + 5)
    ax.set_aspect('equal')
    ax.axis('off')
    label = ["Izquierda", "Centro", "Derecha"][tercio_idx]
    ax.set_title(f"T{span_idx+1} {label}", color='white', fontsize=8)

def generate_figure_n(beam, result, rank, save_dir=None):
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    fig.patch.set_facecolor('#0a0a12')
    gs = GridSpec(3, 4, figure=fig)

    ax_mom = fig.add_subplot(gs[0, :3])
    _plot_moment_n(ax_mom, beam, result, rank)

    ax_table = fig.add_subplot(gs[0, 3])
    _spec_table_axes(ax_table, beam, result)

    ax_elev = fig.add_subplot(gs[1, :])
    _plot_elevation_n(ax_elev, beam, result, rank)

    # Secciones representativas (Tramo 1 y Tramo N)
    for i, s_idx in enumerate([0, beam['n_spans'] - 1]):
        if s_idx < 0: continue
        for t_idx in range(3):
            ax_sec = fig.add_subplot(gs[2, i*3 + t_idx if i*3+t_idx < 4 else 3])
            _draw_section_n(ax_sec, beam, result, s_idx, t_idx)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"result_{beam['id']}_rank{rank}.png"
        path = os.path.join(save_dir, fname)
        plt.savefig(path, dpi=130, facecolor=fig.get_facecolor())
        print(f"  -> Figura guardada: {path}")
    plt.close(fig)

def plot_convergence(result_ga, beam_id, save_dir=None):
    hist = result_ga.get('history', [])
    if not hist: return

    gens = [h['gen'] for h in hist]
    best = [h['best_fit'] for h in hist]
    avg = [h['avg_fit'] for h in hist]

    plt.figure(figsize=(8, 5))
    plt.plot(gens, best, label='Mejor Fitness')
    plt.plot(gens, avg, label='Promedio Fitness', alpha=0.6)
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.title(f"Convergencia GA - {beam_id}")
    plt.xlabel("Generacion")
    plt.ylabel("Fitness (log scale)")
    plt.legend()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"convergence_{beam_id}.png")
        plt.savefig(path)
        print(f"  -> Convergencia guardada: {path}")
    plt.close()
