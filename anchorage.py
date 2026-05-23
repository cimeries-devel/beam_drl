"""Decisiones de anclaje en nudos — recto vs gancho 90° vs infactible.

Modulo puro (sin numpy ni dependencias del GA). Lo consultan `chromosome_n.py`
y `fitness_n.py` para decidir si una barra cabe anclada en un nudo dado el
`h_col` (ancho de columna en direccion de la viga).

Reglas:
    espacio = h_col*100 - cover_joint
    1) espacio >= Ld(diam, face)          -> 'straight'
    2) espacio >= Ldg(diam)  Y capa == 0  -> 'hook90'
    3) sino                                -> 'infeasible'

Capas internas (capa >= 1) NO admiten gancho: el rabo de 16*db chocaria con
las barras de capa 0 al bajar dentro del nudo.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'GA_viga_completa'))

from config_ga import (
    LD_INFERIOR,
    LD_SUPERIOR,
    LDG,
    HOOK_TAIL_CM,
    R_LIBRE_CM,
    D_ESTRIBO_CM,
)

# Espacio ocupado por recubrimiento + estribo dentro del nudo (cm).
COVER_JOINT_CM = R_LIBRE_CM + D_ESTRIBO_CM

# Valor conservador para lookups fuera de tabla (fallback).
_LD_FALLBACK_CM = 60.0
_LDG_FALLBACK_CM = 50.0


def clear_space_cm(h_col_m):
    """Espacio util dentro del nudo para anclaje, en cm."""
    return h_col_m * 100.0 - COVER_JOINT_CM


def ld_cm(diam_name, fc_kg_cm2, face):
    """Longitud de anclaje recto (cm) segun E.060 Tabla 21-2/21-3.

    face: 'TOP' usa L'd (superior, factor 1.3); 'BOT' usa Ld inferior.
    """
    table = LD_SUPERIOR if face == 'TOP' else LD_INFERIOR
    fc_key = int(round(fc_kg_cm2))
    if diam_name in table and fc_key in table[diam_name]:
        return float(table[diam_name][fc_key])
    return _LD_FALLBACK_CM


def ldg_cm(diam_name, fc_kg_cm2):
    """Longitud de anclaje con gancho 90° (cm) segun E.060 Art. 12.5."""
    fc_key = int(round(fc_kg_cm2))
    if diam_name in LDG and fc_key in LDG[diam_name]:
        return float(LDG[diam_name][fc_key])
    return _LDG_FALLBACK_CM


def hook_tail_cm(diam_name):
    """Cola vertical del gancho 90° = 16*db, redondeada a 5 cm (cm)."""
    return float(HOOK_TAIL_CM.get(diam_name, 45.0))


def classify_anchorage(diam_name, fc_kg_cm2, face, h_col_m, layer_idx):
    """Decide el tipo de anclaje para una barra individual en un nudo.

    Args:
        diam_name: nombre de diametro ('1/2', '5/8', '3/4', '1', etc.)
        fc_kg_cm2: resistencia del concreto
        face: 'TOP' o 'BOT'
        h_col_m: ancho de columna en direccion de la viga (metros)
        layer_idx: indice de capa (0 = exterior, >=1 = interior)

    Returns:
        dict con:
            'type': 'straight' | 'hook90' | 'infeasible'
            'length_cm': longitud de anclaje horizontal (cm) | None
            'tail_cm': cola vertical adicional si gancho (cm) | 0
            'reason': descripcion breve
    """
    space = clear_space_cm(h_col_m)
    ld = ld_cm(diam_name, fc_kg_cm2, face)
    ldg = ldg_cm(diam_name, fc_kg_cm2)

    if space >= ld:
        return {
            'type': 'straight',
            'length_cm': ld,
            'tail_cm': 0.0,
            'reason': f'recto {face} cabe (espacio={space:.0f} >= Ld={ld:.0f})',
        }

    if layer_idx == 0 and space >= ldg:
        return {
            'type': 'hook90',
            'length_cm': ldg,
            'tail_cm': hook_tail_cm(diam_name),
            'reason': f'gancho 90° (espacio={space:.0f} >= Ldg={ldg:.0f})',
        }

    if layer_idx >= 1 and space < ld:
        return {
            'type': 'infeasible',
            'length_cm': None,
            'tail_cm': 0.0,
            'reason': (
                f'capa interior {layer_idx} sin espacio recto '
                f'(espacio={space:.0f} < Ld={ld:.0f}); gancho no permitido en capa>=1'
            ),
        }

    return {
        'type': 'infeasible',
        'length_cm': None,
        'tail_cm': 0.0,
        'reason': (
            f'ni recto ni gancho caben (espacio={space:.0f} < '
            f'Ld={ld:.0f}, Ldg={ldg:.0f})'
        ),
    }


def anchorage_length_m(diam_name, fc_kg_cm2, face, h_col_m, layer_idx):
    """Longitud total de anclaje en metros (horizontal + cola si gancho).

    Returns None si infactible.
    """
    dec = classify_anchorage(diam_name, fc_kg_cm2, face, h_col_m, layer_idx)
    if dec['type'] == 'infeasible':
        return None
    return (dec['length_cm'] + dec['tail_cm']) / 100.0


def is_feasible(diam_name, fc_kg_cm2, face, h_col_m, layer_idx):
    """True si hay algun anclaje factible para esta barra."""
    return classify_anchorage(
        diam_name, fc_kg_cm2, face, h_col_m, layer_idx
    )['type'] != 'infeasible'


# ---------------------------------------------------------------------------
# Self-tests (correr con: python anchorage.py)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # --- Caso 1: h_col 30 cm, fc 210, capa 0 ---
    # Espacio = 25 cm.
    # 1/2" BOT: Ld=45 cm no cabe, Ldg=30 cm no cabe (25<30) -> infactible
    d = classify_anchorage('1/2', 210, 'BOT', 0.30, 0)
    assert d['type'] == 'infeasible', f"1/2 BOT 30cm: {d}"

    # 5/8" TOP: Ld=73 no cabe, Ldg=35 no cabe -> infactible
    d = classify_anchorage('5/8', 210, 'TOP', 0.30, 0)
    assert d['type'] == 'infeasible', f"5/8 TOP 30cm: {d}"

    # --- Caso 2: h_col 35 cm, fc 210, capa 0 ---
    # Espacio = 30 cm.
    # 1/2" BOT: Ldg=30 cm cabe exacto -> gancho
    d = classify_anchorage('1/2', 210, 'BOT', 0.35, 0)
    assert d['type'] == 'hook90', f"1/2 BOT 35cm: {d}"
    assert d['length_cm'] == 30.0

    # --- Caso 3: h_col 40 cm, fc 210, capa 0 ---
    # Espacio = 35 cm.
    # 5/8" BOT: Ld=56 no cabe, Ldg=35 cabe -> gancho
    d = classify_anchorage('5/8', 210, 'BOT', 0.40, 0)
    assert d['type'] == 'hook90', f"5/8 BOT 40cm: {d}"

    # 3/4" BOT: Ld=67 no cabe, Ldg=45 no cabe (35<45) -> infactible
    d = classify_anchorage('3/4', 210, 'BOT', 0.40, 0)
    assert d['type'] == 'infeasible', f"3/4 BOT 40cm: {d}"

    # --- Caso 4: h_col 1.00 m (columna grande) ---
    # Espacio = 95 cm.
    # 5/8" TOP: Ld=73 cm cabe -> recto
    d = classify_anchorage('5/8', 210, 'TOP', 1.00, 0)
    assert d['type'] == 'straight', f"5/8 TOP 1m: {d}"
    # 1" BOT: Ld=112 cm NO cabe en 95 -> gancho (Ldg=60)
    d = classify_anchorage('1', 210, 'BOT', 1.00, 0)
    assert d['type'] == 'hook90', f"1 BOT 1m: {d}"
    # 1" BOT con h_col 1.20 m (espacio 115) -> recto (112<=115)
    d = classify_anchorage('1', 210, 'BOT', 1.20, 0)
    assert d['type'] == 'straight', f"1 BOT 1.2m: {d}"

    # --- Caso 5: capa interior sin espacio recto -> infactible (no hay gancho) ---
    # 5/8" BOT capa 1 con h_col 40 cm: Ld=56 no cabe -> infactible (no gancho)
    d = classify_anchorage('5/8', 210, 'BOT', 0.40, 1)
    assert d['type'] == 'infeasible', f"5/8 BOT capa1 40cm: {d}"
    assert 'capa interior' in d['reason']

    # --- Caso 6: anchorage_length_m ---
    # 5/8" BOT 40 cm gancho: 35 + 30 = 65 cm = 0.65 m
    L = anchorage_length_m('5/8', 210, 'BOT', 0.40, 0)
    assert abs(L - 0.65) < 1e-6, f"expected 0.65, got {L}"

    # 1/2" BOT 1 m recto: 45 cm + 0 = 0.45 m
    L = anchorage_length_m('1/2', 210, 'BOT', 1.00, 0)
    assert abs(L - 0.45) < 1e-6, f"expected 0.45, got {L}"

    # Infactible -> None
    L = anchorage_length_m('3/4', 210, 'BOT', 0.25, 0)
    assert L is None

    print("anchorage.py: todos los tests OK")
