"""Constantes y configuracion para el GA de vigas de N tramos."""

from config.config_ga import (
    N_CAPAS_MAX,
    S_VERT_MIN_CM,
    R_LIBRE_CM,
    D_ESTRIBO_CM,
    # Lambdas de penalizacion
    LAMBDA_M,
    LAMBDA_N,
    LAMBDA_G,
    PENALTY_FIXED,
    LAMBDA_EXC,
    LAMBDA_CORR_EXC,
    LAMBDA_CONSTR_BAST,
    ET_MIN_DUCTILITY,
    # GA base
    TOURNAMENT_K,
    N_ELITE,
    P_CROSS,
    P_MUT_ONI,
    P_MUT_RESET,
    HOF_POOL_SIZE,
    RESTART_PAT,
    RESTART_RATIO,
    # Longitudes de anclaje
    LD_INFERIOR,
    LD_SUPERIOR,
    LDG,
    HOOK_TAIL_CM,
    H_COL_DEFAULT_M,
    LAMBDA_ANCHORAGE,
)

# ---------------------------------------------------------------------------
# Catalogo de varillas propio del sistema N-tramos (6 diametros: 3/8" a 1")
# No modifica config.py del sistema PPO (que solo tiene 4 entradas).
# ---------------------------------------------------------------------------
REBAR_CATALOG_N = {
    0: {"name": "1/2", "area_cm2": 1.29,  "diam_cm": 1.27},
    1: {"name": "5/8", "area_cm2": 2.00,  "diam_cm": 1.59},
    2: {"name": "3/4", "area_cm2": 2.84,  "diam_cm": 1.91},
    3: {"name": "1",   "area_cm2": 5.10,  "diam_cm": 2.54},
}

# Peso lineal del acero (kg/m) por indice dO
REBAR_WEIGHTS_KGM_N = {
    0: 0.994,   # 1/2"
    1: 1.552,   # 5/8"
    2: 2.235,   # 3/4"
    3: 3.973,   # 1"
}

# Numero de diametros disponibles (rango de dO: 0..N_DIAM-1)
N_DIAM = 4

# ---------------------------------------------------------------------------
# Penalizaciones adicionales del sistema N-tramos
# ---------------------------------------------------------------------------
# Penaliza diametros no contiguos dentro de una zona (max(dO)-min(dO) > 1)
LAMBDA_DIAM_CONTIG = 200.0

# El corrido recorre toda la viga y naturalmente "sobra" en zonas de baja
# demanda — ese exceso es inherente al diseño y ya lo penaliza el peso.
# Anular aqui para no distorsionar el fitness en el sistema N-tramos.
LAMBDA_CORR_EXC = 0.0

# ---------------------------------------------------------------------------
# Parametros de mutacion propios
# ---------------------------------------------------------------------------
# Probabilidad de mutar el diametro dO de un slot activo (±1 en catalogo)
P_MUT_DO = 0.06

# ---------------------------------------------------------------------------
# Opciones de corrido
# ---------------------------------------------------------------------------
CORRIDO_SIMETRICO = True   # True: corrido_top == corrido_bot

# ---------------------------------------------------------------------------
# Maximo de tramos soportado
# ---------------------------------------------------------------------------
MAX_SPANS = 10

# ---------------------------------------------------------------------------
# Posiciones dentro de un tramo y caras
# ---------------------------------------------------------------------------
_POSITIONS = ('LEFT', 'MID', 'RIGHT')
_FACES = ('TOP', 'BOT')


def generate_zone_ids(n_spans: int) -> list:
    """Genera 6*n zone IDs ordenados.

    Orden: por tramo, luego LEFT/MID/RIGHT x TOP/BOT.
    Ej n=2: ['LEFT_TOP_T1', 'MID_TOP_T1', 'RIGHT_TOP_T1',
             'LEFT_BOT_T1', 'MID_BOT_T1', 'RIGHT_BOT_T1',
             'LEFT_TOP_T2', ...]
    """
    ids = []
    for i in range(n_spans):
        for face in _FACES:
            for pos in _POSITIONS:
                ids.append(f'{pos}_{face}_T{i + 1}')
    return ids


def zone_ids_for_span(n_spans: int, span_idx: int) -> list:
    """Retorna los 6 zone IDs para un tramo especifico (0-indexed)."""
    all_ids = generate_zone_ids(n_spans)
    return all_ids[span_idx * 6: (span_idx + 1) * 6]


def parse_zone_id(zone_id: str) -> dict:
    """Parsea un zone ID en sus componentes.

    'LEFT_TOP_T2' -> {'pos': 'LEFT', 'face': 'TOP', 'span': 2}
    """
    parts = zone_id.rsplit('_T', 1)
    pos_face = parts[0]
    span = int(parts[1])
    pos, face = pos_face.rsplit('_', 1)
    return {'pos': pos, 'face': face, 'span': span}


def default_joints(n_spans: int, supports_x_m: list) -> list:
    """Crea N+1 joints default con h_col = H_COL_DEFAULT_M.

    Marca 'exterior' en los extremos y 'interior' en el resto.
    """
    n_joints = n_spans + 1
    joints = []
    for i in range(n_joints):
        x = float(supports_x_m[i]) if i < len(supports_x_m) else 0.0
        jtype = 'exterior' if (i == 0 or i == n_spans) else 'interior'
        joints.append({
            'x_m': x,
            'h_col_m': float(H_COL_DEFAULT_M),
            'type': jtype,
        })
    return joints


def ensure_joints(beam: dict) -> None:
    """Mutador: si el beam no trae 'joints', los agrega por default.

    Esto permite cargar beams antiguos (test_beam.json sin joints) sin romper.
    El valor default luego puede ser sobrescrito por el prompt interactivo.
    """
    if 'joints' in beam and beam['joints']:
        return
    n_spans = int(beam['n_spans'])
    supports = beam.get('supports_x_m', [])
    beam['joints'] = default_joints(n_spans, supports)


def scaled_ga_params(n_spans: int) -> dict:
    """Retorna pop_size y n_gen escalados segun n_spans.

    Para problemas mas grandes se necesita mas exploracion.
    """
    if n_spans <= 3:
        return {'pop_size': 150, 'n_gen': 200, 'early_stop_pat': 60}
    elif n_spans <= 6:
        return {'pop_size': 250, 'n_gen': 350, 'early_stop_pat': 80}
    else:
        return {'pop_size': 400, 'n_gen': 500, 'early_stop_pat': 100}
