"""Constantes GA + parámetros del cromosoma para GA_viga_completa."""

# ---------------------------------------------------------------------------
# Geometría de sección
# ---------------------------------------------------------------------------
N_CAPAS_MAX   = 3          # capas por cara
S_VERT_MIN_CM = 2.54       # 1" separación mínima vertical entre capas
R_LIBRE_CM    = 4.0        # recubrimiento libre al estribo (cm)
D_ESTRIBO_CM  = 0.95       # diámetro estribo 3/8" (9.5 mm)

# ---------------------------------------------------------------------------
# Zonas de bastones (6 total): 3 tercios × TOP + BOT
# ---------------------------------------------------------------------------
ZONE_IDS = ['LEFT_TOP', 'MID_TOP', 'RIGHT_TOP', 'LEFT_BOT', 'MID_BOT', 'RIGHT_BOT']

# ---------------------------------------------------------------------------
# Hiperparámetros GA
# ---------------------------------------------------------------------------
POP_SIZE        = 150
N_GEN           = 200
TOURNAMENT_K    = 3
N_ELITE         = 5
P_CROSS         = 0.80
P_MUT_ONI       = 0.12       # por gen (flip oni)
P_MUT_CHOICE    = 0.08       # por gen (swap diam_choice A↔B)
P_MUT_DIAM      = 0.04       # por cromosoma (muta diam_A o diam_B global)
P_MUT_RESET     = 0.01       # por bloque bastón (reset a todo-cero)
EARLY_STOP_PAT  = 60         # generaciones sin mejora para early stop
HOF_POOL_SIZE   = 10         # tamaño del pool diverso de HoF
RESTART_PAT     = 25         # generaciones sin mejora para reinicio parcial
RESTART_RATIO   = 0.50       # fracción de la población que se reemplaza (peores)

# ---------------------------------------------------------------------------
# Lambdas de penalización
# ---------------------------------------------------------------------------
LAMBDA_M      = 1000.0     # déficit de capacidad φMn
LAMBDA_N      = 500.0      # ρ_min, ε_t, ρ_max
LAMBDA_G      = 100.0      # geometría, esquinas, capas
PENALTY_FIXED = 50.0       # unidad base para penalizaciones fijas
LAMBDA_EXC      = 3.0        # exceso de capacidad en bastones (% sobre Mu)
LAMBDA_CORR_EXC = 20        # exceso de capacidad del corrido respecto a demanda por tercio
LAMBDA_CONSTR_BAST = 50   # constructabilidad de bastones (penaliza capas densas y exceso de barras)

# ---------------------------------------------------------------------------
# Mínimo ε_t ACI-02+
# ---------------------------------------------------------------------------
ET_MIN_DUCTILITY = 0.004

# ---------------------------------------------------------------------------
# Longitudes de anclaje rectas — Norma E.060 2009, fy = 4 200 kg/cm²
# ---------------------------------------------------------------------------
# Tabla 21-2  Barras Inferiores (tracción)
# Condiciones: espaciamiento libre ≥ db, recubrimiento ≥ db, estribos mínimos
# Mínimo Ld ≥ 0.30 m  (Art. 12.2.1)
# Valores en cm; columnas = f'c (kg/cm²)
LD_INFERIOR = {
    #           210   280   350   420   550
    '3/8': {210: 34, 280: 29, 350: 26, 420: 24, 550: 21},
    '1/2': {210: 45, 280: 39, 350: 35, 420: 32, 550: 28},
    '5/8': {210: 56, 280: 49, 350: 43, 420: 40, 550: 35},
    '3/4': {210: 67, 280: 58, 350: 52, 420: 48, 550: 42},
    '7/8': {210: 98, 280: 85, 350: 76, 420: 69, 550: 60},
    '1':   {210: 112, 280: 97, 350: 86, 420: 79, 550: 69},
}

# Tabla 21-3  Barras Superiores (tracción)  — L'd ≥ 1.3 × Ld inferior
# Mínimo L'd ≥ 0.30 m
LD_SUPERIOR = {
    #           210   280   350   420   550
    '3/8': {210: 44, 280: 38, 350: 34, 420: 31, 550: 27},
    '1/2': {210: 58, 280: 51, 350: 45, 420: 41, 550: 36},
    '5/8': {210: 73, 280: 63, 350: 57, 420: 52, 550: 45},
    '3/4': {210: 88, 280: 76, 350: 68, 420: 62, 550: 54},
    '7/8': {210: 127, 280: 110, 350: 98, 420: 90, 550: 78},
    '1':   {210: 145, 280: 126, 350: 112, 420: 103, 550: 90},
}

# ---------------------------------------------------------------------------
# Longitudes de anclaje con gancho estándar 90° — E.060 Art. 12.5
# Ldg = max(318·db/√fc, 8·db, 15 cm). Valores redondeados a 5 cm hacia arriba.
# Se asume fy = 4200 kg/cm² (coeficiente 318 ya lo incluye).
# ---------------------------------------------------------------------------
LDG = {
    #           210   280   350   420
    '3/8': {210: 25, 280: 20, 350: 20, 420: 15},
    '1/2': {210: 30, 280: 25, 350: 25, 420: 20},
    '5/8': {210: 35, 280: 30, 350: 30, 420: 25},
    '3/4': {210: 45, 280: 40, 350: 35, 420: 30},
    '7/8': {210: 50, 280: 45, 350: 40, 420: 35},
    '1':   {210: 60, 280: 50, 350: 45, 420: 40},
}

# Cola vertical del gancho 90° = 16·db redondeada a 5 cm hacia arriba (cm).
HOOK_TAIL_CM = {
    '3/8': 20,
    '1/2': 25,
    '5/8': 30,
    '3/4': 35,
    '7/8': 40,
    '1':   45,
}

# Default de ancho de columna cuando el usuario no especifica (metros).
H_COL_DEFAULT_M = 0.30

# Lambda para penalizar infactibilidad de anclaje en apoyos externos.
# Severa pero menor que LAMBDA_M (capacidad) — el corrido siempre debería
# ser anclable si h_col es razonable; si no, el diseño es directamente
# inaceptable pero no tanto como una sección sin capacidad suficiente.
LAMBDA_ANCHORAGE = 800.0
