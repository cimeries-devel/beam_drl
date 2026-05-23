"""Constantes específicas del sistema GA_beam_compatibilizado.

Tablas de longitud de desarrollo extraídas de beam_functions.py (sin dependencia
de pandas). Solo contiene los 4 diámetros del catálogo del proyecto.
"""

# ---------------------------------------------------------------------------
# Tablas de longitud de desarrollo ld (metros)
# ---------------------------------------------------------------------------

# Barras SUPERIORES en tracción (momento negativo → acero en la parte alta)
LD_SUPERIOR = {
    '1/2': {210: 0.60, 280: 0.55, 350: 0.45},
    '5/8': {210: 0.75, 280: 0.65, 350: 0.60},
    '3/4': {210: 0.90, 280: 0.80, 350: 0.70},
    '1':   {210: 1.45, 280: 1.30, 350: 1.15},
}

# Barras INFERIORES en tracción (momento positivo → acero en la parte baja)
LD_INFERIOR = {
    '1/2': {210: 0.45, 280: 0.40, 350: 0.35},
    '5/8': {210: 0.60, 280: 0.50, 350: 0.45},
    '3/4': {210: 0.70, 280: 0.60, 350: 0.55},
    '1':   {210: 1.15, 280: 1.00, 350: 0.90},
}

# Valores de fc disponibles en las tablas
FC_TABLE_VALUES = [210, 280, 350]


def snap_fc(fc: float) -> int:
    """Retorna el valor de fc más cercano disponible en las tablas ld."""
    return min(FC_TABLE_VALUES, key=lambda v: abs(v - fc))
