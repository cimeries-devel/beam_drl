"""Encode / decode / repair / posiciones Y del cromosoma para GA_viga_completa.

Chromosome layout (flat int array):
  [0]          diam_A_idx  (0-3)
  [1]          diam_B_idx  ∈ {diam_A, diam_A+1}  (contiguos o iguales)
  [2..2+BLOCK] corrido     → reshape [N_CAPAS_MAX, n_slots, 2]
                              dim-2: [diam_choice (0=A,1=B), oni (0/1)]
  Luego 6 bloques de bastones en el mismo formato:
    bast_LEFT_TOP, bast_MID_TOP, bast_RIGHT_TOP,
    bast_LEFT_BOT, bast_MID_BOT, bast_RIGHT_BOT

  Definición de zonas:
    LEFT_{face}:  M_face(x=0) > φMn_corrido  (bastón desde x=0 hasta primer cruce)
    RIGHT_{face}: M_face(x=L) > φMn_corrido  (bastón desde último cruce hasta x=L)
    MID_{face}:   región interior donde M_face > φMn (no toca x=0 ni x=L)

BLOCK = N_CAPAS_MAX * n_slots * 2
Total genes = 2 + 7 * BLOCK
"""

import os
import sys
import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(str(_HERE), '..', 'mejora del modelo'))
sys.path.insert(0, os.path.join(str(_HERE), '..', 'GA beam'))

from config.config import REBAR_CATALOG, VARILLAS_POR_ANCHO
from config import config_ga
from config.config_ga import (
    N_CAPAS_MAX, S_VERT_MIN_CM, R_LIBRE_CM, D_ESTRIBO_CM, ZONE_IDS, N_ZONES
)

# Orden de zonas bastón en el cromosoma (debe coincidir con ZONE_IDS en config_ga.py)
_ZONE_ORDER = ['LEFT_TOP', 'MID_TOP', 'RIGHT_TOP', 'LEFT_BOT', 'MID_BOT', 'RIGHT_BOT']


# ---------------------------------------------------------------------------
# Utilidades de tamaño
# ---------------------------------------------------------------------------

def n_slots_for_beam(b: float) -> int:
    """Número de slots por capa (= max_varillas para el ancho dado)."""
    return VARILLAS_POR_ANCHO[round(b, 2)][1]


def block_size(n_slots: int) -> int:
    """Genes por bloque (corrido o un bastón)."""
    return N_CAPAS_MAX * n_slots * 2


def chrom_length(n_slots: int) -> int:
    """Longitud total del cromosoma."""
    return 2 + (1 + N_ZONES) * block_size(n_slots)


# ---------------------------------------------------------------------------
# Posiciones Y de las barras
# ---------------------------------------------------------------------------

def bar_y_positions(face: str, capas_activas: list, diam_indices: list,
                    h_cm: float) -> list:
    """Posición Y de cada barra (cm) medida desde el borde superior.

    Parameters
    ----------
    face : 'TOP' o 'BOT'
    capas_activas : lista de índices de capa activos (0 = capa exterior)
    diam_indices  : lista de índices de diámetro por capa activa
    h_cm          : altura total de la sección (cm)

    Returns
    -------
    list of float, uno por capa activa, en cm desde borde superior.
    """
    if not capas_activas:
        return []

    # Calcular posiciones desde el borde de referencia (exterior de la cara)
    y_from_face = []
    y_prev = None
    db_prev = None

    for k_order, (capa_k, diam_k) in enumerate(zip(capas_activas, diam_indices)):
        db_k = REBAR_CATALOG[diam_k]['diam_cm']
        if k_order == 0:
            y_k = R_LIBRE_CM + D_ESTRIBO_CM + db_k / 2.0
        else:
            sep = max(db_prev, S_VERT_MIN_CM)
            y_k = y_prev + db_prev / 2.0 + sep + db_k / 2.0
        y_from_face.append(y_k)
        y_prev = y_k
        db_prev = db_k

    # Convertir a y_from_top
    if face == 'TOP':
        return y_from_face
    else:
        return [h_cm - y for y in y_from_face]


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode(z: np.ndarray, n_slots: int) -> dict:
    """Decodifica el cromosoma a un dict estructurado.

    Returns
    -------
    {
      'diam_A': int, 'diam_B': int,
      'corrido':  ndarray [N_CAPAS_MAX, n_slots, 2],
      'bastones': {'Z1_TOP': ndarray, ..., 'Z3_BOT': ndarray}
    }
    """
    BLOCK = block_size(n_slots)
    diam_A = int(z[0])
    diam_B = int(z[1])

    offset = 2
    corrido = z[offset: offset + BLOCK].reshape(N_CAPAS_MAX, n_slots, 2).copy()
    offset += BLOCK

    bastones = {}
    for zone_id in _ZONE_ORDER:
        bastones[zone_id] = z[offset: offset + BLOCK].reshape(N_CAPAS_MAX, n_slots, 2).copy()
        offset += BLOCK

    return {'diam_A': diam_A, 'diam_B': diam_B,
            'corrido': corrido, 'bastones': bastones}


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def encode(decoded: dict, n_slots: int) -> np.ndarray:
    """Codifica el dict a un cromosoma flat."""
    BLOCK = block_size(n_slots)
    z = np.zeros(chrom_length(n_slots), dtype=np.int8)
    z[0] = decoded['diam_A']
    z[1] = decoded['diam_B']
    offset = 2
    z[offset: offset + BLOCK] = decoded['corrido'].flatten()
    offset += BLOCK
    for zone_id in _ZONE_ORDER:
        z[offset: offset + BLOCK] = decoded['bastones'][zone_id].flatten()
        offset += BLOCK
    return z


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def repair(z: np.ndarray, n_slots: int,
           min_v: int, max_v: int) -> np.ndarray:
    """Repara el cromosoma para que cumpla restricciones constructivas.

    Pasos:
    1. Clip diam_A y diam_B a [0,3]; si diam_A > diam_B → swap + actualizar choices
    2. Bastón no puede estar ON en slot donde corrido ya está ON (misma capa)
    3. Si capa_0 tiene ≥1 barra ON → activar slots de esquina (0 y n_slots-1)
    4. Si capa tiene exactamente 1 barra ON → activar segunda barra (esquina)
    5. Capa interior activa requiere capa exterior con ≥ min_v barras
    """
    z = z.copy()
    BLOCK = block_size(n_slots)

    # 1. Clip, ordenar y forzar contiguidad de diámetros
    z[0] = int(np.clip(z[0], 0, 3))
    z[1] = int(np.clip(z[1], 0, 3))
    if z[0] > z[1]:
        z[0], z[1] = z[1], z[0]
        # Intercambiar diam_choice en todos los slots (0↔1)
        offset = 2
        for _ in range(1 + N_ZONES):
            block_view = z[offset: offset + BLOCK].reshape(N_CAPAS_MAX, n_slots, 2)
            oni = block_view[:, :, 1]
            choice = block_view[:, :, 0]
            mask = oni == 1
            choice[mask] = 1 - choice[mask]
            z[offset: offset + BLOCK] = block_view.flatten()
            offset += BLOCK
    # Forzar contiguidad: diam_B ∈ {diam_A, diam_A+1}
    z[1] = min(int(z[1]), int(z[0]) + 1)

    # Procesar cada bloque
    offset = 2
    corrido_view = z[offset: offset + BLOCK].reshape(N_CAPAS_MAX, n_slots, 2).copy()
    _repair_matrix(corrido_view, n_slots, min_v, max_v)

    # Regla corrido: mínimo 2 barras del mismo diámetro.
    # Si en alguna capa solo hay 2 barras y tienen diam_choice distinto (1+1),
    # se fuerzan ambas al diam_choice mayoritario (o al menor si empatan).
    for k in range(N_CAPAS_MAX):
        oni_k = corrido_view[k, :, 1]
        n_on = int(oni_k.sum())
        if n_on < 2:
            continue
        on_slots = np.where(oni_k == 1)[0]
        choices = corrido_view[k, on_slots, 0]
        n_A = int((choices == 0).sum())
        n_B = int((choices == 1).sum())
        # Solo intervenir si ningún diámetro tiene ≥2 barras
        if n_A >= 2 or n_B >= 2:
            continue
        # Forzar todas al diámetro con más barras; si empatan, usar diam_A (menor)
        target = 0 if n_A >= n_B else 1
        corrido_view[k, on_slots, 0] = target

    z[offset: offset + BLOCK] = corrido_view.flatten()
    offset += BLOCK

    # Bastones: oni no puede solapar con corrido
    corrido_oni = corrido_view[:, :, 1]  # [N_CAPAS, n_slots]

    locked_by_corrido = corrido_oni.astype(bool)

    for _ in range(N_ZONES):
        bast_view = z[offset: offset + BLOCK].reshape(N_CAPAS_MAX, n_slots, 2).copy()
        # Apagar slots donde corrido ya está ON (lo reitera _repair_matrix via locked)
        bast_view[:, :, 1][corrido_oni == 1] = 0
        _repair_matrix(bast_view, n_slots, min_v, max_v,
                       allow_zero_capas=True, locked=locked_by_corrido)
        z[offset: offset + BLOCK] = bast_view.flatten()
        offset += BLOCK

    return z


def _repair_matrix(mat: np.ndarray, n_slots: int, min_v: int, max_v: int,
                   allow_zero_capas: bool = False,
                   locked: np.ndarray = None):
    """Repara in-place una matriz [N_CAPAS, n_slots, 2].

    Restricciones aplicadas:
    - Clip oni a {0,1}; los slots bloqueados se mantienen a 0
    - Capa_0: si tiene barras ON → slots de esquina (no bloqueados) deben estar ON
    - Capa con exactamente 1 barra ON → activar segunda esquina disponible
    - Capa_k>0 activa requiere capa_{k-1} COMPLETAMENTE llena (todos los slots
      disponibles ocupados) antes de permitir barras en capa_k
    - Máximo max_v barras por capa

    Parameters
    ----------
    locked : ndarray bool [N_CAPAS, n_slots], opcional
        Slots que no se pueden activar (ej. ocupados por el corrido en bastones).
    """
    if locked is None:
        locked = np.zeros((N_CAPAS_MAX, n_slots), dtype=bool)

    mat[:, :, 1] = np.clip(mat[:, :, 1], 0, 1)
    mat[:, :, 0] = np.clip(mat[:, :, 0], 0, 1)
    mat[:, :, 1][locked] = 0  # garantizar slots bloqueados a 0

    for k in range(N_CAPAS_MAX):
        oni_k = mat[k, :, 1]
        n_on = int(oni_k.sum())

        if n_on == 0:
            continue

        # Slots disponibles en esta capa (no bloqueados)
        avail = [s for s in range(n_slots) if not locked[k, s]]
        n_avail = len(avail)
        if n_avail == 0:
            continue

        # Límite máximo
        cap = min(max_v, n_avail)
        if n_on > cap:
            on_idx = np.where(oni_k == 1)[0].tolist()
            corners = {avail[0], avail[-1]}
            interior = [i for i in on_idx if i not in corners]
            to_off = n_on - cap
            for idx in interior[-to_off:]:
                oni_k[idx] = 0
            n_on = int(oni_k.sum())

        # Regla de esquinas: activar ambas esquinas disponibles
        if n_on >= 1:
            oni_k[avail[0]] = 1
            oni_k[avail[-1]] = 1
            n_on = int(oni_k.sum())

        # Si sólo 1 barra: activar segunda esquina disponible
        if n_on == 1 and n_avail >= 2:
            oni_k[avail[-1]] = 1
            n_on = 2

        # Regla de capa interior: capa_{k-1} debe estar COMPLETAMENTE llena
        # antes de que capa_k tenga barras.
        if k > 0:
            avail_prev = [s for s in range(n_slots) if not locked[k - 1, s]]
            n_prev_on  = int(mat[k - 1, :, 1].sum())
            if n_prev_on < len(avail_prev):
                # Llenar todos los slots disponibles de capa_{k-1}
                for slot in avail_prev:
                    mat[k - 1, slot, 1] = 1


# ---------------------------------------------------------------------------
# Obtener barras activas
# ---------------------------------------------------------------------------

def get_active_bars(matrix: np.ndarray, face: str,
                    diam_A: int, diam_B: int, h_cm: float) -> list:
    """Retorna lista de (y_from_top, area_cm2, diam_name) para slots ON.

    Parameters
    ----------
    matrix : ndarray [N_CAPAS_MAX, n_slots, 2]
             dim-2: [diam_choice (0=A,1=B), oni]
    """
    bars = []
    capas_activas = []
    diam_indices_por_capa = []

    # Determinar diámetro representativo de cada capa (el más frecuente)
    for k in range(N_CAPAS_MAX):
        oni_k = matrix[k, :, 1]
        if oni_k.sum() == 0:
            continue

        choices_on = matrix[k, matrix[k, :, 1] == 1, 0]
        # Diámetro dominante en la capa (para Y)
        n_B = int((choices_on == 1).sum())
        n_A = int((choices_on == 0).sum())
        diam_rep = diam_B if n_B >= n_A else diam_A

        capas_activas.append(k)
        diam_indices_por_capa.append(diam_rep)

    if not capas_activas:
        return []

    y_list = bar_y_positions(face, capas_activas, diam_indices_por_capa, h_cm)

    bar_idx = 0
    for k_order, k in enumerate(capas_activas):
        y_k = y_list[k_order]
        for slot in range(matrix.shape[1]):
            if matrix[k, slot, 1] == 1:
                choice = int(matrix[k, slot, 0])
                diam_idx = diam_B if choice == 1 else diam_A
                info = REBAR_CATALOG[diam_idx]
                bars.append((y_k, info['area_cm2'], info['name']))
        bar_idx += 1

    return bars


# ---------------------------------------------------------------------------
# Layout combinado: corrido + bastones con Y compartida por capa
# ---------------------------------------------------------------------------

def get_combined_bars(corrido_matrix: np.ndarray,
                      baston_matrix: np.ndarray,
                      face: str,
                      diam_A: int, diam_B: int,
                      h_cm: float) -> list:
    """Retorna el layout unificado de barras (corrido + bastones) para una cara.

    La Y de cada capa se calcula UNA SOLA VEZ usando el diámetro máximo presente
    en esa capa (corrido o bastón incluidos). Así todas las barras de la misma
    capa comparten la misma posición vertical.

    Los bastones ocupan los slots que el corrido dejó libres en cada capa.

    Parameters
    ----------
    corrido_matrix : ndarray [N_CAPAS_MAX, n_slots, 2]
    baston_matrix  : ndarray [N_CAPAS_MAX, n_slots, 2]
    face           : 'TOP' o 'BOT'
    diam_A, diam_B : índices de diámetro global
    h_cm           : altura total de la sección (cm)

    Returns
    -------
    list of (y_from_top, area_cm2, diam_name, bar_type, capa_k, slot_idx)
        bar_type: 'corrido' o 'baston'
        Ordenada por capa (exterior primero) y slot (izquierda a derecha).
    """
    n_slots = corrido_matrix.shape[1]
    bars = []

    # --- Paso 1: identificar capas activas y su diámetro máximo ---
    capas_with_bars = []   # [(capa_k, max_diam_cm), ...]

    for k in range(N_CAPAS_MAX):
        corr_oni = corrido_matrix[k, :, 1]
        bast_oni  = baston_matrix[k, :, 1]

        all_diams = []
        for slot in range(n_slots):
            if corr_oni[slot] == 1:
                choice = int(corrido_matrix[k, slot, 0])
                d_idx  = diam_B if choice == 1 else diam_A
                all_diams.append(REBAR_CATALOG[d_idx]['diam_cm'])
            if bast_oni[slot] == 1:
                choice = int(baston_matrix[k, slot, 0])
                d_idx  = diam_B if choice == 1 else diam_A
                all_diams.append(REBAR_CATALOG[d_idx]['diam_cm'])

        if not all_diams:
            continue

        capas_with_bars.append((k, max(all_diams)))

    if not capas_with_bars:
        return []

    # --- Paso 2: calcular Y desde la cara de referencia para cada capa ---
    y_per_capa = {}
    y_prev = None
    db_prev = None

    for k_order, (capa_k, max_diam_cm) in enumerate(capas_with_bars):
        if k_order == 0:
            y_face = R_LIBRE_CM + D_ESTRIBO_CM + max_diam_cm / 2.0
        else:
            sep    = max(db_prev, S_VERT_MIN_CM)
            y_face = y_prev + db_prev / 2.0 + sep + max_diam_cm / 2.0
        y_per_capa[capa_k] = y_face
        y_prev  = y_face
        db_prev = max_diam_cm

    # --- Paso 3: convertir a y_from_top y recolectar barras ---
    def _to_ytop(y_face: float) -> float:
        return y_face if face == 'TOP' else (h_cm - y_face)

    for capa_k, _ in capas_with_bars:
        y_top    = _to_ytop(y_per_capa[capa_k])
        corr_oni = corrido_matrix[capa_k, :, 1]
        bast_oni  = baston_matrix[capa_k, :, 1]

        for slot in range(n_slots):
            if corr_oni[slot] == 1:
                choice = int(corrido_matrix[capa_k, slot, 0])
                d_idx  = diam_B if choice == 1 else diam_A
                info   = REBAR_CATALOG[d_idx]
                bars.append((y_top, info['area_cm2'], info['name'],
                             'corrido', capa_k, slot))
            elif bast_oni[slot] == 1:
                choice = int(baston_matrix[capa_k, slot, 0])
                d_idx  = diam_B if choice == 1 else diam_A
                info   = REBAR_CATALOG[d_idx]
                bars.append((y_top, info['area_cm2'], info['name'],
                             'baston', capa_k, slot))

    return bars