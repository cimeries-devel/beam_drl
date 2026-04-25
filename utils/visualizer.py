import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from config.config import REBAR_CATALOG
from ga.chromosome import get_combined_bars
from config.config_ga import ZONE_IDS, R_LIBRE_CM, D_ESTRIBO_CM, LD_INFERIOR, LD_SUPERIOR

# ---------------------------------------------------------------------------
# Paletas
# ---------------------------------------------------------------------------
_ZONE_COLORS = {
    'LEFT_TOP': '#6baed6', 'MID_TOP': '#fc8d59', 'RIGHT_TOP': '#74c476',
    'LEFT_BOT': '#08519c', 'MID_BOT': '#b30000', 'RIGHT_BOT': '#006d2c',
}
_CORRIDO_COLOR = '#f5c842'   # dorado
_BASTON_COLOR  = '#e040fb'   # magenta
_BG_ELEV       = '#23243a'   # fondo oscuro elevación

# Color por diámetro (para secciones transversales)
_DIAM_COLORS = {
    '1/2': '#42a5f5',   # azul
    '5/8': '#66bb6a',   # verde
    '3/4': '#ffa726',   # naranja
    '1':   '#ef5350',   # rojo
}

def _compute_ld_m(diam_name: str, fc_kgcm2: float, face: str = 'BOT') -> float:
    """Longitud de desarrollo E.060 2009 (Tablas 21-2 / 21-3) en metros.

    Para fc no coincidente con columna de tabla: interpolación lineal entre
    los dos fc más cercanos.  Mínimo reglamentario 0.30 m.

    Parámetros
    ----------
    diam_name : '1/2' | '5/8' | '3/4' | '1'  (nombre de varilla)
    fc_kgcm2  : resistencia del concreto (kg/cm²)
    face      : 'BOT' → Tabla 21-2 (inferior); 'TOP' → Tabla 21-3 (superior)
    """
    table = LD_SUPERIOR if face == 'TOP' else LD_INFERIOR
    row   = table.get(diam_name, table.get('5/8'))   # fallback a 5/8

    fcs   = sorted(row.keys())
    fc    = float(fc_kgcm2)

    # Clamp al rango de la tabla
    if fc <= fcs[0]:
        ld_cm = row[fcs[0]]
    elif fc >= fcs[-1]:
        ld_cm = row[fcs[-1]]
    else:
        # Interpolación lineal entre los dos fc más cercanos
        for i in range(len(fcs) - 1):
            if fcs[i] <= fc <= fcs[i + 1]:
                t = (fc - fcs[i]) / (fcs[i + 1] - fcs[i])
                ld_cm = row[fcs[i]] + t * (row[fcs[i + 1]] - row[fcs[i]])
                break

    return max(ld_cm / 100.0, 0.30)   # convertir cm → m, mínimo 0.30 m


# ---------------------------------------------------------------------------
# Fila 0: Diagrama de momentos con banda de capacidad
# ---------------------------------------------------------------------------

def _plot_moment(ax, beam, result, rank):
    xs     = beam['outputs']['x_m']
    outs   = beam['outputs']
    L      = xs[-1]
    fc     = float(beam['inputs']['fc_kg_cm2'])

    # Detectar envolvente (ETABS) vs curva única (dataset)
    _has_envelope = ('M_tonf_m_max' in outs and 'M_tonf_m_min' in outs)
    if _has_envelope:
        ms_max_raw = outs['M_tonf_m_max']
        ms_min_raw = outs['M_tonf_m_min']
        ms_raw = ms_max_raw  # fallback para anotaciones BOT
    else:
        ms_raw = outs['M_tonf_m']

    # Convención: negamos M para que hogging (negativo) aparezca arriba
    ms     = [-m for m in ms_raw]
    xs_arr = np.array(xs)
    ms_arr = np.array(ms)

    phi_pos = result['phi_mn_corrido']['positive']
    phi_neg = result['phi_mn_corrido']['negative']

    # Banda corrido (dorada)
    ax.fill_between([0, L], 0, -phi_pos,
                    color=_CORRIDO_COLOR, alpha=0.20, zorder=1,
                    label=f'φMn+ corrido = {phi_pos:.2f} t·m')
    ax.fill_between([0, L], 0,  phi_neg,
                    color=_CORRIDO_COLOR, alpha=0.20, zorder=1,
                    label=f'φMn− corrido = {phi_neg:.2f} t·m')
    ax.axhline(-phi_pos, color='#c8900a', lw=1.8, ls='--', alpha=0.9, zorder=3)
    ax.axhline( phi_neg, color='#c8900a', lw=1.8, ls='--', alpha=0.9, zorder=3)

    # Zonas bastones: extents reales + pendiente de desarrollo (Ottazzi Cap.10)
    zone_extents = result.get('zone_extents', {})
    zones_ok     = result['zone_results']
    bast_label_added = False
    ext_label_added  = False
    grad_label_added = False

    for zone_id, zinfo in zone_extents.items():
        if not zinfo['exists']:
            continue
        zr = zones_ok.get(zone_id, {})
        if zr.get('n_bast', 0) == 0:
            continue

        face     = 'BOT' if zone_id.endswith('_BOT') else 'TOP'
        phi_corr = phi_pos if face == 'BOT' else phi_neg
        phi_z    = zr.get('phi_Mn', phi_corr)

        x_s  = zinfo['x_start']
        x_e  = zinfo['x_end']
        x_ts = zinfo['x_theor_start']
        x_te = zinfo['x_theor_end']

        # Signo: BOT → zona está debajo del eje; TOP → encima
        sign   = -1 if face == 'BOT' else 1
        y_base = sign * phi_corr   # capacidad corrido (base de la zona bastón)
        y_top_ = sign * phi_z      # capacidad total (corrido + bastón)

        # ---------- Pendiente de desarrollo (Ld) ----------
        diam_bast = result.get('diam_B', '5/8')
        ld = _compute_ld_m(diam_bast, fc, face)
        bar_len  = max(x_e - x_s, 1e-6)
        ld_phys  = min(ld, bar_len * 0.90)   # cap p/ barras muy cortas

        zone_type = zone_id.split('_')[0]    # 'LEFT', 'MID' o 'RIGHT'
        x_rL = x_s + ld_phys
        x_rR = x_e - ld_phys

        if zone_type == 'LEFT':
            xs_trap = np.array([x_s,    x_rR,   x_e   ])
            ys_trap = np.array([y_top_, y_top_, y_base ])
        elif zone_type == 'RIGHT':
            xs_trap = np.array([x_s,    x_rL,   x_e   ])
            ys_trap = np.array([y_base, y_top_, y_top_])
        else:
            xs_trap = np.array([x_s,    x_rL,   x_rR,   x_e   ])
            ys_trap = np.array([y_base, y_top_, y_top_, y_base ])

        # ---------- Fills ------------------------------------------------
        lbl_b = 'Bastón (demanda)' if not bast_label_added else ''
        lbl_e = 'Extensión E.060'  if not ext_label_added  else ''

        # Demanda real [x_ts, x_te]
        xs_dem = np.linspace(max(x_s, x_ts), min(x_e, x_te), 80)
        ys_dem = np.interp(xs_dem, xs_trap, ys_trap)
        if len(xs_dem) > 1:
            ax.fill_between(xs_dem, y_base, ys_dem,
                            color=_BASTON_COLOR, alpha=0.28, zorder=2,
                            label=lbl_b)
            bast_label_added = True

        # Extensión izquierda [x_s, x_ts]
        if x_ts > x_s + 1e-6:
            xs_ext = np.linspace(x_s, min(x_ts, x_e), 40)
            ys_ext = np.interp(xs_ext, xs_trap, ys_trap)
            ax.fill_between(xs_ext, y_base, ys_ext,
                            color=_BASTON_COLOR, alpha=0.10, zorder=2,
                            label=lbl_e)
            ax.axvline(x_ts, color=_BASTON_COLOR, lw=1.2, ls='--',
                       alpha=0.65, zorder=4)
            ext_label_added = True

        # Extensión derecha [x_te, x_e]
        if x_te < x_e - 1e-6:
            xs_ext = np.linspace(max(x_te, x_s), x_e, 40)
            ys_ext = np.interp(xs_ext, xs_trap, ys_trap)
            ax.fill_between(xs_ext, y_base, ys_ext,
                            color=_BASTON_COLOR, alpha=0.10, zorder=2,
                            label=lbl_e if not ext_label_added else '')
            ax.axvline(x_te, color=_BASTON_COLOR, lw=1.2, ls='--',
                       alpha=0.65, zorder=4)
            ext_label_added = True

        # Línea trapezoidal de capacidad
        ax.plot(xs_trap, ys_trap,
                color=_BASTON_COLOR, lw=2.0, ls='-', zorder=5,
                label='φMn bastón (c/ Ld)' if not grad_label_added else '')
        grad_label_added = True

        if ld_phys > 0.05 * L:
            y_mid   = (y_base + y_top_) / 2
            ax.annotate(
                f'Ld={ld:.2f}m',
                xy=(x_s + ld_phys / 2, y_mid),
                fontsize=6, color=_BASTON_COLOR,
                ha='center', va='center', alpha=0.85,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6, lw=0))

    # Curva de momentos
    if _has_envelope:
        ms_max_plot = np.array([-m for m in ms_max_raw])
        ms_min_plot = np.array([-m for m in ms_min_raw])
        ax.fill_between(xs_arr, ms_max_plot, ms_min_plot,
                        color='#d0d0ff', alpha=0.20, zorder=0)
        ax.plot(xs_arr, ms_max_plot, color='#1565c0', lw=1.8, zorder=6,
                label='Env. M+ (max)')
        ax.plot(xs_arr, ms_min_plot, color='#c62828', lw=1.8, zorder=6,
                label='Env. M− (min)')
    else:
        ax.plot(xs_arr, ms_arr, color='#1a1a2e', lw=2.0, zorder=6, label='M(x)')
        ax.fill_between(xs_arr, 0, ms_arr, where=ms_arr > 0,
                        color='#e8e8ff', alpha=0.35, zorder=0)
        ax.fill_between(xs_arr, 0, ms_arr, where=ms_arr < 0,
                        color='#ffe8e8', alpha=0.35, zorder=0)

    # Marcadores de demanda (Mu) por zona activa
    for zone_id, zinfo in zone_extents.items():
        if not zinfo['exists'] or zinfo['Mu'] < 0.01:
            continue
        face = 'BOT' if zone_id.endswith('_BOT') else 'TOP'
        x_ts, x_te = zinfo['x_theor_start'], zinfo['x_theor_end']
        x_lo = min(x_ts, x_te)
        x_hi = max(x_ts, x_te) if x_te > x_ts else x_lo + (L * 0.01)

        zr    = zones_ok.get(zone_id, {})
        color = '#27ae60' if zr.get('ok') else '#e74c3c'

        if _has_envelope:
            ms_annot = ms_max_raw if face == 'BOT' else ms_min_raw
        else:
            ms_annot = ms_raw

        if face == 'BOT':
            pts = [(xs[i], ms_annot[i])
                   for i in range(len(xs)) if x_lo <= xs[i] <= x_hi and ms_annot[i] > 0]
            if pts:
                x_mx, m_mx = max(pts, key=lambda p: p[1])
                ax.annotate(f'{m_mx:.1f}', (x_mx, -m_mx),
                            xytext=(0, -14), textcoords='offset points',
                            ha='center', fontsize=7, color=color, fontweight='bold')
        else:
            pts = [(xs[i], ms_annot[i])
                   for i in range(len(xs)) if x_lo <= xs[i] <= x_hi and ms_annot[i] < 0]
            if pts:
                x_mx, m_mx = min(pts, key=lambda p: p[1])
                ax.annotate(f'{abs(m_mx):.1f}', (x_mx, abs(m_mx)),
                            xytext=(0, 6), textcoords='offset points',
                            ha='center', fontsize=7, color=color, fontweight='bold')

    ax.axhline(0, color='#555', lw=0.8)
    ax.set_xlim(0, L)
    ax.set_xlabel('x (m)', fontsize=9)
    ax.set_ylabel('Momento (tonf·m)', fontsize=9)
    ax.set_title(f'Rank {rank} — Diagrama M(x) con capacidad',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Fila 1: Elevación esquemática
# ---------------------------------------------------------------------------

def _plot_elevation(ax, beam, result, rank):
    L   = float(beam['outputs']['x_m'][-1])
    h_m = float(beam['inputs']['h_m'])

    ax.set_facecolor(_BG_ELEV)
    ax.set_xlim(-0.05, L + 0.05)
    ax.set_ylim(-0.15, 1.15)
    ax.axis('off')
    ax.set_title('Elevación esquemática', fontsize=9, color='white', pad=4)

    # Contorno de la viga
    rect = patches.FancyBboxPatch((0, 0), L, 1.0,
                                   boxstyle='round,pad=0.01',
                                   linewidth=1.5, edgecolor='#ccc',
                                   facecolor='#2d2e4a')
    ax.add_patch(rect)

    # Corrido superior e inferior
    y_top = 0.90
    y_bot = 0.10
    ax.plot([0, L], [y_top, y_top], color=_CORRIDO_COLOR,
            lw=3, solid_capstyle='butt', label='Corrido')
    ax.plot([0, L], [y_bot, y_bot], color=_CORRIDO_COLOR,
            lw=3, solid_capstyle='butt')

    def _bast_spec(zone_id):
        mat = result['bastones_matrices'].get(zone_id)
        if mat is None:
            return ''
        dA = result['diam_A']
        dB = result['diam_B']
        oni  = mat[:, :, 1]
        cho  = mat[:, :, 0]
        n_A  = int((oni * (1 - cho)).sum())
        n_B  = int((oni * cho).sum())
        if n_A > 0 and n_B > 0:
            return f'{n_A}o{dA}"+{n_B}o{dB}"'
        elif n_B > 0:
            return f'{n_B}o{dB}"'
        elif n_A > 0:
            return f'{n_A}o{dA}"'
        return ''

    zones_ok     = result['zone_results']
    zone_extents = result.get('zone_extents', {})
    bast_legend_added = False

    for zone_id in ZONE_IDS:
        zr = zones_ok.get(zone_id, {})
        if zr.get('n_bast', 0) == 0:
            continue
        face  = 'BOT' if zone_id.endswith('_BOT') else 'TOP'
        zinfo = zone_extents.get(zone_id, {})
        x0    = zinfo.get('x_start', 0.0)
        x1    = zinfo.get('x_end',   L)
        x_ts  = zinfo.get('x_theor_start', x0)
        x_te  = zinfo.get('x_theor_end',   x1)

        y_line = y_top - 0.06 if face == 'TOP' else y_bot + 0.06
        y_text = y_top + 0.04 if face == 'TOP' else y_bot - 0.08
        va_txt = 'bottom'    if face == 'TOP' else 'top'

        ax.plot([x0, x1], [y_line, y_line],
                color=_BASTON_COLOR, lw=2.5, solid_capstyle='butt',
                alpha=0.5, zorder=4)
        ax.plot([x_ts, x_te], [y_line, y_line],
                color=_BASTON_COLOR, lw=2.5, solid_capstyle='butt',
                alpha=1.0, zorder=5,
                label='Bastón' if not bast_legend_added else '')
        bast_legend_added = True

        for x_cut in [x_ts, x_te]:
            if 1e-6 < x_cut < L - 1e-6:
                ax.plot([x_cut, x_cut], [0, 1],
                        color=_BASTON_COLOR, lw=0.9, ls=':', alpha=0.55)

        ax.text((x0 + x1) / 2, y_text,
                _bast_spec(zone_id), color=_BASTON_COLOR,
                fontsize=7, ha='center', va=va_txt, fontweight='bold')

    dA = result['diam_A']
    dB = result['diam_B']
    corr_bars = result['corrido_bars']
    n_A = sum(1 for _, _, d in corr_bars if d == dA)
    n_B = sum(1 for _, _, d in corr_bars if d == dB)
    if dA == dB or n_B == 0:
        spec = f'{len(corr_bars)}ø{dA}"'
    else:
        spec = f'{n_A}ø{dA}" + {n_B}ø{dB}"'
    ax.text(L / 2, 0.50, f'CORRIDO: {spec}',
            color=_CORRIDO_COLOR, fontsize=8,
            ha='center', va='center', fontweight='bold')

    ax.text(0,   -0.08, '0',         color='#aaa', fontsize=7, ha='center')
    ax.text(L/4, -0.08, f'{L/4:.1f}m', color='#aaa', fontsize=7, ha='center')
    ax.text(L/2, -0.08, f'{L/2:.1f}m', color='#aaa', fontsize=7, ha='center')
    ax.text(3*L/4, -0.08, f'{3*L/4:.1f}m', color='#aaa', fontsize=7, ha='center')
    ax.text(L,   -0.08, f'{L:.1f}m', color='#aaa', fontsize=7, ha='center')

    ax.legend(loc='upper left', fontsize=7, facecolor='#2d2e4a',
              labelcolor='white', framealpha=0.7)


# ---------------------------------------------------------------------------
# Fila 2: Secciones transversales (T1, T2, T3) — TOP + BOT unificado
# ---------------------------------------------------------------------------

_SEC_NAMES = ['LEFT', 'MID', 'RIGHT']

def _draw_section(ax, beam, result, tercio_idx):
    sec_name = _SEC_NAMES[tercio_idx]
    b    = round(float(beam['inputs']['b_m']), 2)
    h    = float(beam['inputs']['h_m'])
    h_cm = h * 100.0
    b_cm = b * 100.0

    dA_idx = result['diam_A_idx']
    dB_idx = result['diam_B_idx']
    n_slots = result['corrido_matrix'].shape[1]
    bast_mats = result['bastones_matrices']
    empty = np.zeros_like(result['corrido_matrix'])

    r_corner = REBAR_CATALOG[dA_idx]['diam_cm'] / 2.0
    margin   = R_LIBRE_CM + D_ESTRIBO_CM + r_corner
    slot_xs  = np.linspace(margin, b_cm - margin, n_slots)

    ax.set_facecolor('white')
    ax.add_patch(patches.Rectangle(
        (0, 0), b_cm, h_cm,
        linewidth=1.5, edgecolor='#333', facecolor='#f8f8f8'))

    s_off = R_LIBRE_CM + D_ESTRIBO_CM / 2.0
    ax.add_patch(patches.Rectangle(
        (s_off, s_off),
        b_cm - 2.0 * s_off,
        h_cm - 2.0 * s_off,
        linewidth=1.8, edgecolor='#444', facecolor='none',
        joinstyle='round', zorder=4))

    legend_keys = {}

    def _draw_face(face):
        bast_key = f'{sec_name}_{face}'
        bast_mat = bast_mats.get(bast_key, empty)
        combined = get_combined_bars(result['corrido_matrix'], bast_mat,
                                     face, dA_idx, dB_idx, h_cm)
        for y_top, area, dname, bar_type, capa_k, slot_idx in combined:
            db_cm  = REBAR_CATALOG[_name2idx(dname)]['diam_cm']
            r      = db_cm / 2.0
            x      = slot_xs[slot_idx]
            y_plot = h_cm - y_top
            color  = _DIAM_COLORS.get(dname, '#aaaaaa')
            filled = (bar_type == 'corrido')

            if filled:
                ax.add_patch(plt.Circle((x, y_plot), r,
                                        color=color, zorder=5, alpha=0.92))
                legend_keys.setdefault((dname, True), (color, True))
            else:
                ax.add_patch(plt.Circle((x, y_plot), r,
                                        fill=False, edgecolor=color,
                                        lw=2.2, zorder=5))
                legend_keys.setdefault((dname, False), (color, False))

    _draw_face('TOP')
    _draw_face('BOT')

    ax.set_xlim(-2, b_cm + 2)
    ax.set_ylim(-2, h_cm + 4)
    ax.set_aspect('equal')
    ax.axis('off')

    if legend_keys:
        legend_handles = []
        for (dname, filled), (color, _) in sorted(legend_keys.items()):
            tipo = 'corrido' if filled else 'baston'
            h = patches.Patch(
                facecolor=color if filled else 'none',
                edgecolor=color, linewidth=1.8,
                label=f'o{dname}" {tipo}')
            legend_handles.append(h)
        ax.legend(handles=legend_handles, fontsize=6,
                  loc='upper center', bbox_to_anchor=(0.5, -0.02),
                  ncol=len(legend_handles), framealpha=0.75,
                  handlelength=1.2, handleheight=1.2)

    zbot_res = result['zone_results'].get(f'{sec_name}_BOT', {})
    ztop_res = result['zone_results'].get(f'{sec_name}_TOP', {})
    ok_bot = 'SI' if zbot_res.get('ok') else 'NO'
    ok_top = 'SI' if ztop_res.get('ok') else 'NO'
    title = (f'{sec_name} | BOT:{ok_bot} TOP:{ok_top}\n'
             f'φMn+={zbot_res.get("phi_Mn", 0):.2f} / '
             f'φMn-={ztop_res.get("phi_Mn", 0):.2f} t-m')
    ax.set_title(title, fontsize=7.5, pad=3)

def _name2idx(dname: str) -> int:
    for idx, info in REBAR_CATALOG.items():
        if info['name'] == dname:
            return idx
    return 0

# ---------------------------------------------------------------------------
# Generar figura para un diseño
# ---------------------------------------------------------------------------

def generate_figure(beam, result, rank, save_dir=None):
    fig = plt.figure(figsize=(18, 13), constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig,
                  height_ratios=[1.4, 0.7, 1.2],
                  hspace=0.45, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, :])
    _plot_moment(ax0, beam, result, rank)

    ax1 = fig.add_subplot(gs[1, :])
    _plot_elevation(ax1, beam, result, rank)

    for col in range(3):
        ax_s = fig.add_subplot(gs[2, col])
        _draw_section(ax_s, beam, result, tercio_idx=col)

    beam_id = beam['id']
    b = round(float(beam['inputs']['b_m']), 2)
    h = float(beam['inputs']['h_m'])
    fc = float(beam['inputs']['fc_kg_cm2'])
    L = float(beam['outputs']['x_m'][-1])

    zones_ok = result['zones_ok']
    feasible = 'Factible' if result['feasible'] else 'NO factible'
    subtitle = (f'GA Viga Completa — Viga #{beam_id} | '
                f'b={b}m h={h}m fc={fc:.0f}kg/cm² L={L}m | '
                f'Rank {rank} | W={result["total_weight_kg"]:.2f}kg | '
                f'{zones_ok}/6 zonas OK | {feasible}')
    fig.suptitle(subtitle, fontsize=9.5, fontweight='bold', y=0.98)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f'ga_completa_viga_{beam_id}_rank{rank}.png')
        fig.savefig(fname, dpi=140, bbox_inches='tight')
        print(f'  Figura guardada: {fname}')

    plt.close(fig)

def generate_convergence_figure(ga_result, beam_id, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    gens = list(range(1, len(ga_result['history_best']) + 1))

    if ga_result.get('history_worst'):
        ax.plot(gens, ga_result['history_worst'],
                label='Peor fitness', color='#7b1fa2', lw=1.2, ls='-.')
    ax.plot(gens, ga_result['history_mean'],
            label='Fitness promedio', color='#e74c3c', lw=1.5, ls='--')
    ax.plot(gens, ga_result['history_best'],
            label='Mejor fitness', color='#1565c0', lw=2)
    if ga_result.get('early_stop'):
        ax.axvline(ga_result['generations_run'],
                   color='#555', ls=':', lw=1.5, label='Early stop')
    ax.set_xlabel('Generación')
    ax.set_ylabel('Fitness')
    ax.set_title(f'Convergencia GA — Viga #{beam_id}')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f'ga_completa_viga_{beam_id}_convergencia.png')
        fig.savefig(fname, dpi=120, bbox_inches='tight')
        print(f'  Convergencia guardada: {fname}')

    plt.close(fig)
