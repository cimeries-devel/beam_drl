"""Cromosoma parametrico para GA de vigas de N tramos.

Layout (flat int8 array):
  [0]              diam_A_idx  (0-3)
  [1]              diam_B_idx  ∈ {diam_A, diam_A+1}
  [2..2+BLOCK)     CORRIDO_TOP  → [N_CAPAS_MAX, n_slots, 2]
  [2+BLOCK..2+2*BLOCK) CORRIDO_BOT  → [N_CAPAS_MAX, n_slots, 2]
  [2+2*BLOCK..]    6*n_spans bloques de bastones en orden zone_ids

  Total genes = 2 + (2 + 6*n_spans) * BLOCK
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'GA_viga_completa'))
sys.path.insert(0, os.path.join(_HERE, '..', 'mejora del modelo'))
sys.path.insert(0, os.path.join(_HERE, '..', 'GA beam'))

# Reutilizar funciones del cromosoma original (span-independent)
from chromosome import (
    bar_y_positions,
    get_active_bars,
    get_combined_bars,
    _repair_matrix,
    n_slots_for_beam,
    block_size,
)
from config import REBAR_CATALOG, VARILLAS_POR_ANCHO
from config_n import (
    N_CAPAS_MAX,
    generate_zone_ids,
    parse_zone_id,
    CORRIDO_SIMETRICO,
    ensure_joints,
)
from anchorage import classify_anchorage


# ---------------------------------------------------------------------------
# Utilidades de tamano
# ---------------------------------------------------------------------------

def chrom_length_n(n_slots: int, n_spans: int) -> int:
    """Longitud total del cromosoma para n tramos.

    2 (diametros) + 2 corridos (TOP+BOT) + 6*n_spans bastones.
    """
    BLOCK = block_size(n_slots)
    return 2 + (2 + 6 * n_spans) * BLOCK


# ---------------------------------------------------------------------------
# Decode / Encode
# ---------------------------------------------------------------------------

def decode_n(z: np.ndarray, n_slots: int, n_spans: int) -> dict:
    """Decodifica cromosoma plano a dict estructurado.

    Returns
    -------
    {
        'diam_A': int, 'diam_B': int,
        'corrido_top': ndarray [N_CAPAS_MAX, n_slots, 2],
        'corrido_bot': ndarray [N_CAPAS_MAX, n_slots, 2],
        'bastones': {zone_id: ndarray [N_CAPAS_MAX, n_slots, 2], ...}
    }
    """
    BLOCK = block_size(n_slots)
    diam_A = int(z[0])
    diam_B = int(z[1])

    offset = 2
    corrido_top = z[offset: offset + BLOCK].reshape(N_CAPAS_MAX, n_slots, 2).copy()
    offset += BLOCK
    corrido_bot = z[offset: offset + BLOCK].reshape(N_CAPAS_MAX, n_slots, 2).copy()
    offset += BLOCK

    zone_ids = generate_zone_ids(n_spans)
    bastones = {}
    for zone_id in zone_ids:
        bastones[zone_id] = z[offset: offset + BLOCK].reshape(
            N_CAPAS_MAX, n_slots, 2).copy()
        offset += BLOCK

    return {
        'diam_A': diam_A,
        'diam_B': diam_B,
        'corrido_top': corrido_top,
        'corrido_bot': corrido_bot,
        'bastones': bastones,
    }


def encode_n(decoded: dict, n_slots: int, n_spans: int) -> np.ndarray:
    """Codifica dict estructurado a cromosoma plano."""
    BLOCK = block_size(n_slots)
    L = chrom_length_n(n_slots, n_spans)
    z = np.zeros(L, dtype=np.int8)
    z[0] = decoded['diam_A']
    z[1] = decoded['diam_B']

    offset = 2
    z[offset: offset + BLOCK] = decoded['corrido_top'].flatten()
    offset += BLOCK
    z[offset: offset + BLOCK] = decoded['corrido_bot'].flatten()
    offset += BLOCK

    zone_ids = generate_zone_ids(n_spans)
    for zone_id in zone_ids:
        z[offset: offset + BLOCK] = decoded['bastones'][zone_id].flatten()
        offset += BLOCK

    return z


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def repair_n(z: np.ndarray, n_slots: int, n_spans: int,
             min_v: int, max_v: int,
             corrido_simetrico: bool = CORRIDO_SIMETRICO,
             beam: dict = None) -> np.ndarray:
    """Repara cromosoma de n tramos.

    Pasos:
    1. Clip + orden + contiguidad de diametros
    2. Reparar corrido_top y corrido_bot
    3. Si corrido_simetrico: copiar corrido_top -> corrido_bot
    4. Bastones: oni no puede solapar con corrido de su cara
    5. Soporte compartido: OR de oni entre RIGHT_*_Tk y LEFT_*_T{k+1}
    """
    z = z.copy()
    BLOCK = block_size(n_slots)
    n_total_blocks = 2 + 6 * n_spans  # corrido_top + corrido_bot + bastones

    # --- 1. Diametros ---
    z[0] = int(np.clip(z[0], 0, 3))
    z[1] = int(np.clip(z[1], 0, 3))
    if z[0] > z[1]:
        z[0], z[1] = z[1], z[0]
        # Intercambiar diam_choice en todos los bloques
        offset = 2
        for _ in range(n_total_blocks):
            block_view = z[offset: offset + BLOCK].reshape(
                N_CAPAS_MAX, n_slots, 2)
            oni = block_view[:, :, 1]
            choice = block_view[:, :, 0]
            mask = oni == 1
            choice[mask] = 1 - choice[mask]
            z[offset: offset + BLOCK] = block_view.flatten()
            offset += BLOCK
    z[1] = min(int(z[1]), int(z[0]) + 1)

    # --- 2. Reparar corrido_top ---
    off_top = 2
    corrido_top = z[off_top: off_top + BLOCK].reshape(
        N_CAPAS_MAX, n_slots, 2).copy()
    _repair_matrix(corrido_top, n_slots, min_v, max_v)
    _enforce_corrido_diam_rule(corrido_top, n_slots)
    z[off_top: off_top + BLOCK] = corrido_top.flatten()

    # --- 3. Reparar corrido_bot ---
    off_bot = 2 + BLOCK
    if corrido_simetrico:
        # Copiar corrido_top -> corrido_bot
        z[off_bot: off_bot + BLOCK] = z[off_top: off_top + BLOCK].copy()
        corrido_bot = corrido_top.copy()
    else:
        corrido_bot = z[off_bot: off_bot + BLOCK].reshape(
            N_CAPAS_MAX, n_slots, 2).copy()
        _repair_matrix(corrido_bot, n_slots, min_v, max_v)
        _enforce_corrido_diam_rule(corrido_bot, n_slots)
        z[off_bot: off_bot + BLOCK] = corrido_bot.flatten()

    # --- 4. Bastones: no solapar con corrido de su cara ---
    corrido_top_oni = corrido_top[:, :, 1].astype(bool)
    corrido_bot_oni = corrido_bot[:, :, 1].astype(bool)

    zone_ids = generate_zone_ids(n_spans)
    offset = 2 + 2 * BLOCK  # despues de ambos corridos

    for zone_id in zone_ids:
        parsed = parse_zone_id(zone_id)
        face = parsed['face']
        locked = corrido_top_oni if face == 'TOP' else corrido_bot_oni

        bast_view = z[offset: offset + BLOCK].reshape(
            N_CAPAS_MAX, n_slots, 2).copy()
        bast_view[:, :, 1][locked] = 0
        _repair_matrix(bast_view, n_slots, min_v, max_v,
                       allow_zero_capas=True, locked=locked)
        z[offset: offset + BLOCK] = bast_view.flatten()
        offset += BLOCK

    # --- 5. Soporte compartido con fusion parcial ---
    # En cada apoyo interior k, clasificacion celda a celda:
    # barras que pueden anclar terminan; las que no, se fuerzan a continuar.
    _enforce_support_sharing(z, n_slots, n_spans, BLOCK, beam)

    return z


def _enforce_corrido_diam_rule(corrido_view: np.ndarray, n_slots: int):
    """Fuerza minimo 2 barras del mismo diametro por capa del corrido.

    Si una capa tiene barras con diam_choice mixto y ninguno tiene >=2,
    fuerza todas al diametro menor (diam_A).
    """
    for k in range(N_CAPAS_MAX):
        oni_k = corrido_view[k, :, 1]
        n_on = int(oni_k.sum())
        if n_on < 2:
            continue
        on_slots = np.where(oni_k == 1)[0]
        choices = corrido_view[k, on_slots, 0]
        n_A = int((choices == 0).sum())
        n_B = int((choices == 1).sum())
        if n_A >= 2 or n_B >= 2:
            continue
        target = 0 if n_A >= n_B else 1
        corrido_view[k, on_slots, 0] = target


def _classify_and_write_cell(right_view, left_view, c, s,
                              dA_name, dB_name, fc, face, h_col_m):
    """Decide celda (capa c, slot s) entre continuar, terminar o conflicto.

    - Ambos OFF: nada.
    - Ambos ON mismo diam: pasante (ok).
    - Ambos ON distinto diam: conflicto -> forzar al mayor (diam_B=1).
    - Uno ON: intentar terminar con anclaje recto/gancho segun h_col.
      Si infactible, forzar continuidad (copiar al otro lado).
    """
    r_ch = int(right_view[c, s, 0])
    r_on = int(right_view[c, s, 1])
    l_ch = int(left_view[c, s, 0])
    l_on = int(left_view[c, s, 1])

    if r_on == 0 and l_on == 0:
        return

    if r_on == 1 and l_on == 1:
        if r_ch == l_ch:
            return
        target_ch = max(r_ch, l_ch)
        right_view[c, s, 0] = target_ch
        left_view[c, s, 0] = target_ch
        return

    # Solo uno ON -- intentar terminar
    active_ch = r_ch if r_on == 1 else l_ch
    diam_name = dA_name if active_ch == 0 else dB_name
    decision = classify_anchorage(diam_name, fc, face, h_col_m, layer_idx=c)

    if decision['type'] != 'infeasible':
        # Termina dentro del nudo -- asimetria permitida, no tocar
        return

    # Infactible -> forzar continuidad (copiar al lado apagado)
    if r_on == 1:
        left_view[c, s, 0] = r_ch
        left_view[c, s, 1] = 1
    else:
        right_view[c, s, 0] = l_ch
        right_view[c, s, 1] = 1


def _enforce_support_sharing(z: np.ndarray, n_slots: int,
                             n_spans: int, BLOCK: int,
                             beam: dict = None):
    """Fusion parcial celda por celda en apoyos interiores.

    Una barra pasa al otro lado si ambos lados la tienen activa en la misma
    celda (capa, slot). Si solo un lado la tiene, termina dentro del nudo
    con anclaje recto/gancho si el h_col lo permite; si no, se fuerza a
    continuar. Conflictos de diametro en celda compartida -> el mayor gana.
    """
    if n_spans < 2:
        return

    zone_ids = generate_zone_ids(n_spans)

    # Si no hay beam (e.g. test aislado), usar fallback: h_col grande,
    # fc=210, diam names neutros -- todo cabe recto, equivale al OR anterior
    # pero respetando el layout.
    if beam is None:
        fc = 210.0
        dA_name = '5/8'
        dB_name = '3/4'
        joints_h = [999.0] * (n_spans + 1)
    else:
        ensure_joints(beam)
        fc = float(beam['inputs']['fc_kg_cm2'])
        dA_idx = int(z[0])
        dB_idx = int(z[1])
        dA_name = REBAR_CATALOG[dA_idx]['name']
        dB_name = REBAR_CATALOG[dB_idx]['name']
        joints_h = [float(j['h_col_m']) for j in beam['joints']]

    for k in range(n_spans - 1):
        h_col_m = joints_h[k + 1]  # apoyo interior k+1 (0-indexed joint list)
        for face in ('TOP', 'BOT'):
            right_id = f'RIGHT_{face}_T{k + 1}'
            left_id = f'LEFT_{face}_T{k + 2}'

            right_zone_idx = zone_ids.index(right_id)
            left_zone_idx = zone_ids.index(left_id)

            right_off = 2 + 2 * BLOCK + right_zone_idx * BLOCK
            left_off = 2 + 2 * BLOCK + left_zone_idx * BLOCK

            right_view = z[right_off: right_off + BLOCK].reshape(
                N_CAPAS_MAX, n_slots, 2)
            left_view = z[left_off: left_off + BLOCK].reshape(
                N_CAPAS_MAX, n_slots, 2)

            for c in range(N_CAPAS_MAX):
                for s in range(n_slots):
                    _classify_and_write_cell(
                        right_view, left_view, c, s,
                        dA_name, dB_name, fc, face, h_col_m,
                    )

            z[right_off: right_off + BLOCK] = right_view.flatten()
            z[left_off: left_off + BLOCK] = left_view.flatten()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    np.random.seed(42)
    b = 0.25
    n_s = n_slots_for_beam(b)
    n_spans = 3

    L = chrom_length_n(n_s, n_spans)
    BLOCK = block_size(n_s)
    print(f"b={b}, n_slots={n_s}, n_spans={n_spans}")
    print(f"chrom_length={L}, block={BLOCK}")
    print(f"zone_ids ({6*n_spans}):", generate_zone_ids(n_spans))

    # Cromosoma aleatorio
    z = np.random.randint(0, 2, L, dtype=np.int8)
    z[0] = 1  # diam_A = 5/8"
    z[1] = 2  # diam_B = 3/4"

    min_v, max_v = VARILLAS_POR_ANCHO[round(b, 2)]
    z_rep = repair_n(z, n_s, n_spans, min_v, max_v, corrido_simetrico=True)
    dec = decode_n(z_rep, n_s, n_spans)

    print(f"\ndiam_A={dec['diam_A']}, diam_B={dec['diam_B']}")
    print(f"corrido_top capa0 oni: {dec['corrido_top'][0, :, 1]}")
    print(f"corrido_bot capa0 oni: {dec['corrido_bot'][0, :, 1]}")
    assert np.array_equal(dec['corrido_top'], dec['corrido_bot']), \
        "Simetria de corrido FAIL"
    print("Corrido simetrico: OK")

    # Con fusion parcial, RIGHT/LEFT ya no deben ser identicos en general.
    # Sin beam -> fallback h_col=999, equivalente a permitir asimetria libre.
    print("Fusion parcial aplicada (RIGHT/LEFT pueden diferir)")

    # Round-trip
    enc = encode_n(dec, n_s, n_spans)
    dec2 = decode_n(enc, n_s, n_spans)
    assert dec2['diam_A'] == dec['diam_A']
    assert dec2['diam_B'] == dec['diam_B']
    assert np.array_equal(dec2['corrido_top'], dec['corrido_top'])
    assert np.array_equal(dec2['corrido_bot'], dec['corrido_bot'])
    for zid in dec['bastones']:
        assert np.array_equal(dec2['bastones'][zid], dec['bastones'][zid])
    print("Round-trip encode/decode: OK")

    # Barras activas
    bars_top = get_active_bars(dec['corrido_top'], 'TOP',
                               dec['diam_A'], dec['diam_B'], 40.0)
    bars_bot = get_active_bars(dec['corrido_bot'], 'BOT',
                               dec['diam_A'], dec['diam_B'], 40.0)
    print(f"\nBarras corrido TOP: {len(bars_top)}")
    for b_ in bars_top:
        print(f"  y={b_[0]:.2f}cm, As={b_[1]}cm2, o{b_[2]}")
    print(f"Barras corrido BOT: {len(bars_bot)}")
    for b_ in bars_bot:
        print(f"  y={b_[0]:.2f}cm, As={b_[1]}cm2, o{b_[2]}")
