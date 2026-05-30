"""Configuracion centralizada para el sistema PPO de diseno de vigas."""

# --- Constantes materiales E.060 ---
FY = 4200        # kg/cm2 - fluencia del acero
ES = 2_000_000   # kg/cm2 - modulo de elasticidad del acero
ECU = 0.003      # deformacion ultima del concreto
PHI_FLEXION = 0.9  # factor de reduccion por flexion

# --- Recubrimientos (en metros) ---
R1 = 0.06  # recubrimiento 1 capa
R2 = 0.08  # recubrimiento 2 capas

# --- Catalogo de varillas ---
REBAR_CATALOG = {
    0: {"name": "1/2", "area_cm2": 1.29, "diam_cm": 1.27},
    1: {"name": "5/8", "area_cm2": 2.00, "diam_cm": 1.59},
    2: {"name": "3/4", "area_cm2": 2.84, "diam_cm": 1.91},
    3: {"name": "1",   "area_cm2": 5.10, "diam_cm": 2.54},
}
NUM_REBAR_TYPES = len(REBAR_CATALOG)

# --- Varillas permitidas por ancho de viga ---
# (min_varillas, max_varillas) por capa
VARILLAS_POR_ANCHO = {
    0.15: (2, 2),
    0.20: (2, 2),
    0.25: (2, 3),
    0.30: (2, 4),
    0.35: (2, 4),
    0.40: (3, 5),
    0.45: (3, 5),
    0.50: (3, 6),
}

# --- Hiperparametros PPO ---
PPO_CONFIG = {
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# Learning rate con decay lineal: empieza en 3e-4 y baja a 1e-5
LR_START = 3e-4
LR_END = 1e-5

# --- Arquitectura de red ---
POLICY_KWARGS = dict(net_arch=[256, 256, 128])

# --- Constantes de entrenamiento ---
MAX_STEPS_PER_EPISODE = 5
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 50_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 100

# --- Porcentaje del dataset a usar (del total de 100k vigas) ---
TRAIN_PCT = 0.40
TEST_PCT = 0.05
