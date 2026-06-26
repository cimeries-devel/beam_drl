"""Cromosoma parametrico para GA de vigas de N tramos.

Layout (flat int8 array) — SIN genes globales de diametro:
  [0..BLOCK)              CORRIDO_TOP  → [N_CAPAS_MAX, n_slots, 2]
  [BLOCK..2*BLOCK)        CORRIDO_BOT  → [N_CAPAS_MAX, n_slots, 2]
  [2*BLOCK..]             6*n_spans bloques de bastones en orden zone_ids

  Por cada slot (capa c, posicion s):
    mat[c, s, 0] = dO    → indice absoluto en REBAR_CATALOG_N (0=3/8" .. 5=1")
    mat[c, s, 1] = on_i  → 0=inactivo, 1=activo

  Total genes = (2 + 6·N) · BLOCK
  BLOCK = N_CAPAS_MAX × n_slots × 2

Cambio respecto al sistema anterior:
  - Eliminados genes globales diam_A/diam_B (posiciones 0 y 1).
  - dO ahora es indice absoluto (0-5) en lugar de choice relativo (0/1).
  - Regla de diametros contiguos → penalizacion en fitness (no repair).
  - Anclaje exterior: repair obligatorio. Anclaje interior: GA decide via on_i.
"""

import numpy as np

from ga.chromosome import (
    _repair_matrix,
    n_slots_for_beam,
    block_size,
)
from config.config import VARILLAS_POR_ANCHO

from .config_n import (
    N_CAPAS_MAX,
    S_VERT_MIN_CM,
    R_LIBRE_CM,
    D_ESTRIBO_CM,
    REBAR_CATALOG_N,
    N_DIAM,
    generate_zone_ids,
    parse_zone_id,
    CORRIDO_SIMETRICO,
    ensure_joints,
)
from .anchorage import classify_anchorage


# ---------------------------------------------------------------------------
# Utilidades de tamano
# ---------------------------------------------------------------------------

def chrom_length_n(n_slots: int, n_spans: int) -> int:
    """Longitud total del cromosoma para n tramos (sin genes globales).

    (2 corridos + 6*n_spans bastones) * BLOCK
    """
    return (2 + 6 * n_spans) * block_size(n_slots)


# ---------------------------------------------------------------------------
# Decode / Encode
# ---------------------------------------------------------------------------

def decode_n(z: np.ndarray, n_slots: int, n_spans: int) -> dict:
    """Decodifica cromosoma plano a dict estructurado.

    Returns
    -------
    {
        'corrido_top': ndarray [N_CAPAS_MAX, n_slots, 2],
        'corrido_bot': ndarray [N_CAPAS_MAX, n_slots, 2],
        'bastones': {zone_id: ndarray [N_CAPAS_MAX, n_slots, 2], ...}
    }
    Nota: ya no hay claves 'diam_A' ni 'diam_B' — cada slot tiene su propio dO.
    """
    BLOCK = block_size(n_slots)

    offset = 0
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
        'corrido_top': corrido_top,
        'corrido_bot': corrido_bot,
        'bastones': bastones,
    }


def encode_n(decoded: dict, n_slots: int, n_spans: int) -> np.ndarray:
    """Codifica dict estructurado a cromosoma plano."""
    BLOCK = block_size(n_slots)
    L = chrom_length_n(n_slots, n_spans)
    z = np.zeros(L, dtype=np.int8)

    offset = 0
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

def _repair_matrix_n(mat: np.ndarray, n_slots: int, min_v: int, max_v: int,
                     allow_zero_capas: bool = False,
                     locked: np.ndarray = None):
    """Repara in-place una matriz [N_CAPAS, n_slots, 2] con dO absoluto.

    _repair_matrix de chromosome.py clipa mat[:,:,0] a [0,1], destruyendo
    nuestros valores dO en [0,5]. Este wrapper salva y restaura dO alrededor
    de esa llamada, dejando que _repair_matrix solo toque las reglas de on_i.
    """
    dO_saved = mat[:, :, 0].copy()
    _repair_matrix(mat, n_slots, min_v, max_v,
                   allow_zero_capas=allow_zero_capas, locked=locked)
    # Restaurar dO (repair_matrix los habia clipado a [0,1])
    mat[:, :, 0] = np.clip(dO_saved, 0, N_DIAM - 1)


def repair_n(z: np.ndarray, n_slots: int, n_spans: int,
             min_v: int, max_v: int,
             corrido_simetrico: bool = CORRIDO_SIMETRICO,
             beam: dict = None) -> np.ndarray:
    """Repara cromosoma de n tramos.

    Pasos:
    1. Clip dO a [0, N_DIAM-1] y on_i a {0,1} en todos los bloques
    2. Reparar corrido_top (reglas de capas, esquinas, max barras)
    3. Reparar corrido_bot (o copiar si corrido_simetrico)
    4. Bastones: on_i no puede solapar con corrido de su cara
    5. Soporte compartido: decision celda por celda en apoyos interiores
    """
    z = z.copy()
    BLOCK = block_size(n_slots)
    n_total_blocks = 2 + 6 * n_spans

    # --- 1. Clip dO y on_i en todos los bloques ---
    for bi in range(n_total_blocks):
        off = bi * BLOCK
        blk = z[off: off + BLOCK].reshape(N_CAPAS_MAX, n_slots, 2)
        blk[:, :, 0] = np.clip(blk[:, :, 0], 0, N_DIAM - 1)
        blk[:, :, 1] = np.clip(blk[:, :, 1], 0, 1)
        z[off: off + BLOCK] = blk.flatten()

    # --- 2. Reparar corrido_top ---
    off_top = 0
    corrido_top = z[off_top: off_top + BLOCK].reshape(
        N_CAPAS_MAX, n_slots, 2).copy()
    # Garantizar esquinas ON en capa_0 para que _repair_matrix active su logica
    corrido_top[0, 0, 1] = 1
    corrido_top[0, n_slots - 1, 1] = 1
    _repair_matrix_n(corrido_top, n_slots, min_v, max_v)
    z[off_top: off_top + BLOCK] = corrido_top.flatten()

    # --- 3. Reparar corrido_bot ---
    off_bot = BLOCK
    if corrido_simetrico:
        z[off_bot: off_bot + BLOCK] = z[off_top: off_top + BLOCK].copy()
        corrido_bot = corrido_top.copy()
    else:
        corrido_bot = z[off_bot: off_bot + BLOCK].reshape(
            N_CAPAS_MAX, n_slots, 2).copy()
        corrido_bot[0, 0, 1] = 1
        corrido_bot[0, n_slots - 1, 1] = 1
        _repair_matrix_n(corrido_bot, n_slots, min_v, max_v)
        z[off_bot: off_bot + BLOCK] = corrido_bot.flatten()

    # --- 4. Bastones: no solapar con corrido de su cara ---
    corrido_top_oni = corrido_top[:, :, 1].astype(bool)
    corrido_bot_oni = corrido_bot[:, :, 1].astype(bool)

    zone_ids = generate_zone_ids(n_spans)
    offset = 2 * BLOCK  # despues de ambos corridos

    for zone_id in zone_ids:
        parsed = parse_zone_id(zone_id)
        face = parsed['face']
        locked = corrido_top_oni if face == 'TOP' else corrido_bot_oni

        bast_view = z[offset: offset + BLOCK].reshape(
            N_CAPAS_MAX, n_slots, 2).copy()
        bast_view[:, :, 1][locked] = 0
        _repair_matrix_n(bast_view, n_slots, min_v, max_v,
                         allow_zero_capas=True, locked=locked)
        z[offset: offset + BLOCK] = bast_view.flatten()
        offset += BLOCK

    # --- 5. Soporte compartido con fusion parcial ---
    _enforce_support_sharing(z, n_slots, n_spans, BLOCK, beam)

    return z


def _classify_and_write_cell(right_view, left_view, c, s, fc, face, h_col_m):
    """Decide celda (capa c, slot s) en el apoyo interior: pasante o termina.

    dO es ahora absoluto (0-5), resuelto directamente desde REBAR_CATALOG_N.

    - Ambos OFF: nada.
    - Ambos ON, mismo dO: pasante (ok, misma barra).
    - Ambos ON, distinto dO: conflicto → mayor dO gana, barra pasante.
    - Solo uno ON: intentar terminar con gancho/recto segun h_col.
      Si infactible (capa>=1 sin recto), forzar continuidad.
    """
    r_dO = int(right_view[c, s, 0])
    r_on = int(right_view[c, s, 1])
    l_dO = int(left_view[c, s, 0])
    l_on = int(left_view[c, s, 1])

    if r_on == 0 and l_on == 0:
        return

    if r_on == 1 and l_on == 1:
        # Mismo dO: pasante sin cambio
        if r_dO == l_dO:
            return
        # Distinto dO: conflicto → mayor diametro, barra pasante
        target_dO = max(r_dO, l_dO)
        right_view[c, s, 0] = target_dO
        left_view[c, s, 0] = target_dO
        return

    # Solo un lado activo → intentar terminar con anclaje
    active_dO = r_dO if r_on == 1 else l_dO
    diam_name = REBAR_CATALOG_N[active_dO]['name']
    decision = classify_anchorage(diam_name, fc, face, h_col_m, layer_idx=c)

    if decision['type'] != 'infeasible':
        # Cabe (recto, gancho, o gancho constructivo) → dejar terminar
        return

    # Infactible (capa>=1 sin espacio recto) → forzar continuidad
    if r_on == 1:
        left_view[c, s, 0] = r_dO
        left_view[c, s, 1] = 1
    else:
        right_view[c, s, 0] = l_dO
        right_view[c, s, 1] = 1


def _enforce_support_sharing(z: np.ndarray, n_slots: int,
                             n_spans: int, BLOCK: int,
                             beam: dict = None):
    """Fusion parcial celda por celda en apoyos interiores.

    Offsets actualizados: sin los 2 genes globales al inicio.
    Los bastones empiezan en posicion 2*BLOCK (despues de los 2 corridos).
    """
    if n_spans < 2:
        return

    zone_ids = generate_zone_ids(n_spans)

    if beam is None:
        fc = 210.0
        joints_h = [999.0] * (n_spans + 1)
    else:
        ensure_joints(beam)
        fc = float(beam['inputs']['fc_kg_cm2'])
        joints_h = [float(j['h_col_m']) for j in beam['joints']]

    for k in range(n_spans - 1):
        h_col_m = joints_h[k + 1]
        for face in ('TOP', 'BOT'):
            right_id = f'RIGHT_{face}_T{k + 1}'
            left_id = f'LEFT_{face}_T{k + 2}'

            right_zone_idx = zone_ids.index(right_id)
            left_zone_idx = zone_ids.index(left_id)

            # Offset: 2 corridos + posicion del bloque de baston
            right_off = 2 * BLOCK + right_zone_idx * BLOCK
            left_off = 2 * BLOCK + left_zone_idx * BLOCK

            right_view = z[right_off: right_off + BLOCK].reshape(
                N_CAPAS_MAX, n_slots, 2)
            left_view = z[left_off: left_off + BLOCK].reshape(
                N_CAPAS_MAX, n_slots, 2)

            for c in range(N_CAPAS_MAX):
                for s in range(n_slots):
                    _classify_and_write_cell(
                        right_view, left_view, c, s,
                        fc, face, h_col_m,
                    )

            z[right_off: right_off + BLOCK] = right_view.flatten()
            z[left_off: left_off + BLOCK] = left_view.flatten()


# ---------------------------------------------------------------------------
# Barras activas — versiones para dO absoluto (REBAR_CATALOG_N)
# ---------------------------------------------------------------------------

def get_active_bars_n(matrix: np.ndarray, face: str, h_cm: float) -> list:
    """Barras activas de una matriz de slots con dO absoluto.

    Parameters
    ----------
    matrix : [N_CAPAS_MAX, n_slots, 2]  donde dim2 = [dO(0-5), on_i(0/1)]
    face   : 'TOP' o 'BOT'
    h_cm   : altura total de la seccion (cm)

    Returns
    -------
    list de (y_from_top_cm, area_cm2, diam_name)
    """
    capas_info = []
    for k in range(N_CAPAS_MAX):
        oni_k = matrix[k, :, 1]
        if oni_k.sum() == 0:
            continue
        # Diametro representativo por capa = mayor dO entre slots activos
        rep_dO = int(matrix[k, oni_k == 1, 0].max())
        db_cm = REBAR_CATALOG_N[rep_dO]['diam_cm']
        capas_info.append((k, db_cm))

    if not capas_info:
        return []

    # Posicion Y desde la cara (borde exterior)
    y_from_face = []
    y_prev = db_prev = None
    for k_order, (_, db_cm) in enumerate(capas_info):
        if k_order == 0:
            y_k = R_LIBRE_CM + D_ESTRIBO_CM + db_cm / 2.0
        else:
            sep = max(db_prev, S_VERT_MIN_CM)
            y_k = y_prev + db_prev / 2.0 + sep + db_cm / 2.0
        y_from_face.append(y_k)
        y_prev = y_k
        db_prev = db_cm

    if face == 'TOP':
        y_list = y_from_face
    else:
        y_list = [h_cm - y for y in y_from_face]

    bars = []
    for k_order, (k, _) in enumerate(capas_info):
        y_top = y_list[k_order]
        for s in range(matrix.shape[1]):
            if matrix[k, s, 1] == 1:
                dO = int(matrix[k, s, 0])
                info = REBAR_CATALOG_N[dO]
                bars.append((y_top, info['area_cm2'], info['name']))
    return bars


def get_combined_bars_n(corrido_mat: np.ndarray, baston_mat: np.ndarray,
                         face: str, h_cm: float) -> list:
    """Barras unificadas corrido + baston con dO absoluto.

    Returns
    -------
    list de (y_from_top_cm, area_cm2, diam_name, bar_type, capa_k, slot_idx)
    bar_type ∈ {'corrido', 'baston'}
    """
    n_slots = corrido_mat.shape[1]

    # Capas con alguna barra (corrido o baston)
    capas_info = []
    for k in range(N_CAPAS_MAX):
        c_oni = corrido_mat[k, :, 1]
        b_oni = baston_mat[k, :, 1]
        if c_oni.sum() == 0 and b_oni.sum() == 0:
            continue
        # Diametro maximo en esta capa (corrido y/o baston)
        diam_cms = []
        for s in range(n_slots):
            if c_oni[s] == 1:
                diam_cms.append(REBAR_CATALOG_N[int(corrido_mat[k, s, 0])]['diam_cm'])
            if b_oni[s] == 1:
                diam_cms.append(REBAR_CATALOG_N[int(baston_mat[k, s, 0])]['diam_cm'])
        capas_info.append((k, max(diam_cms)))

    if not capas_info:
        return []

    # Posicion Y desde la cara
    y_from_face = []
    y_prev = db_prev = None
    for k_order, (_, max_db) in enumerate(capas_info):
        if k_order == 0:
            y_k = R_LIBRE_CM + D_ESTRIBO_CM + max_db / 2.0
        else:
            sep = max(db_prev, S_VERT_MIN_CM)
            y_k = y_prev + db_prev / 2.0 + sep + max_db / 2.0
        y_from_face.append(y_k)
        y_prev = y_k
        db_prev = max_db

    if face == 'TOP':
        y_list = y_from_face
    else:
        y_list = [h_cm - y for y in y_from_face]

    bars = []
    for k_order, (k, _) in enumerate(capas_info):
        y_top = y_list[k_order]
        for s in range(n_slots):
            c_on = int(corrido_mat[k, s, 1])
            b_on = int(baston_mat[k, s, 1])
            if c_on == 1:
                dO = int(corrido_mat[k, s, 0])
                info = REBAR_CATALOG_N[dO]
                bars.append((y_top, info['area_cm2'], info['name'], 'corrido', k, s))
            elif b_on == 1:
                dO = int(baston_mat[k, s, 0])
                info = REBAR_CATALOG_N[dO]
                bars.append((y_top, info['area_cm2'], info['name'], 'baston', k, s))
    return bars


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
    print(f"chrom_length={L} (antes era {2 + (2+6*n_spans)*BLOCK}), block={BLOCK}")
    print(f"zone_ids ({6*n_spans}):", generate_zone_ids(n_spans))

    # --- Test 1: longitud correcta ---
    expected_L = (2 + 6 * n_spans) * BLOCK
    assert L == expected_L, f"Longitud incorrecta: {L} != {expected_L}"
    print("Test 1 PASS: chrom_length correcto (sin genes globales)")

    # --- Test 2: clip dO y round-trip encode/decode ---
    z = np.random.randint(0, 10, L, dtype=np.int8)  # dO fuera de rango a proposito
    z[1::2] = np.random.randint(0, 2, L // 2, dtype=np.int8)  # on_i
    min_v, max_v = VARILLAS_POR_ANCHO[round(b, 2)]
    z_rep = repair_n(z, n_s, n_spans, min_v, max_v, corrido_simetrico=True)
    dec = decode_n(z_rep, n_s, n_spans)

    # No deben existir claves diam_A/diam_B
    assert 'diam_A' not in dec, "diam_A no debe existir en decode_n"
    assert 'diam_B' not in dec, "diam_B no debe existir en decode_n"
    # Todo dO en rango
    assert dec['corrido_top'][:, :, 0].max() <= N_DIAM - 1
    assert dec['corrido_top'][:, :, 0].min() >= 0
    print("Test 2 PASS: clip dO y sin genes globales")

    # --- Test 3: corrido simetrico ---
    assert np.array_equal(dec['corrido_top'], dec['corrido_bot']), \
        "Simetria de corrido FAIL"
    print("Test 3 PASS: corrido simetrico")

    # --- Test 4: round-trip encode/decode ---
    enc = encode_n(dec, n_s, n_spans)
    dec2 = decode_n(enc, n_s, n_spans)
    assert np.array_equal(dec2['corrido_top'], dec['corrido_top'])
    assert np.array_equal(dec2['corrido_bot'], dec['corrido_bot'])
    for zid in dec['bastones']:
        assert np.array_equal(dec2['bastones'][zid], dec['bastones'][zid])
    print("Test 4 PASS: round-trip encode/decode")

    # --- Test 5: esquinas capa 0 activas ---
    assert dec['corrido_top'][0, 0, 1] == 1, "Esquina izquierda capa0 debe estar ON"
    assert dec['corrido_top'][0, n_s - 1, 1] == 1, "Esquina derecha capa0 debe estar ON"
    print("Test 5 PASS: esquinas capa 0 activas")

    # --- Test 6: fusion parcial (sin beam → h_col=999, todo termina) ---
    print("Test 6: fusion parcial aplicada (h_col=999, barras pueden terminar)")

    # --- Test 7: get_active_bars_n ---
    bars_top = get_active_bars_n(dec['corrido_top'], 'TOP', 40.0)
    bars_bot = get_active_bars_n(dec['corrido_bot'], 'BOT', 40.0)
    print(f"\nBarras corrido TOP ({len(bars_top)}):")
    for b_ in bars_top:
        print(f"  y={b_[0]:.2f}cm, As={b_[1]}cm2, {b_[2]}\"")
    print(f"Barras corrido BOT ({len(bars_bot)}):")
    for b_ in bars_bot:
        print(f"  y={b_[0]:.2f}cm, As={b_[1]}cm2, {b_[2]}\"")
    print("Test 7 PASS: get_active_bars_n")

    # --- Test 8: get_combined_bars_n ---
    zone_ids = generate_zone_ids(n_spans)
    first_zone = zone_ids[0]
    bast_mat = dec['bastones'][first_zone]
    combined = get_combined_bars_n(dec['corrido_top'], bast_mat, 'TOP', 40.0)
    print(f"\nCombinado corrido+baston ({first_zone}): {len(combined)} barras")
    for row in combined:
        print(f"  y={row[0]:.2f}cm, As={row[1]}cm2, {row[2]}\", {row[3]}")
    print("Test 8 PASS: get_combined_bars_n")

    print("\nTodos los tests OK")
