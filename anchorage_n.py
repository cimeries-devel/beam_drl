"""Utilidades de anclaje y calculo de longitud de desarrollo."""

from config_compat import LD_INFERIOR, LD_SUPERIOR

def compute_ld_m(diam_name, fc, face):
    """Calcula la longitud de desarrollo (Ld) en metros."""
    fc_key = int(round(fc))
    table = LD_INFERIOR if face == 'BOT' else LD_SUPERIOR
    if diam_name in table and fc_key in table[diam_name]:
        return table[diam_name][fc_key] / 100.0
    return 0.40

def get_anchorage_length(diam_name, fc, face, type='straight'):
    """Retorna la longitud de anclaje requerida segun el tipo."""
    if type == 'straight':
        return compute_ld_m(diam_name, fc, face)
    # Podrian agregarse mas tipos (hook, etc)
    return 0.30
