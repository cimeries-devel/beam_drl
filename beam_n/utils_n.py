"""Funciones auxiliares para el GA de vigas de N tramos."""

import json
from .config_n import REBAR_CATALOG_N

def prompt_h_col(beam: dict) -> None:
    """Prompt interactivo para h_col en cada apoyo."""
    n_spans = int(beam['n_spans'])
    supports_x = list(beam['supports_x_m'])
    n_joints = n_spans + 1

    print("\n" + "=" * 60)
    print("INGRESO DE h_col (ancho de columna en cada apoyo)")
    print("=" * 60)
    print(f"Viga de {n_spans} tramos, {n_joints} apoyos.")
    print("Presione Enter para aceptar el default [0.30 m].\n")

    joints = []
    for i in range(n_joints):
        x = float(supports_x[i]) if i < len(supports_x) else 0.0
        jtype = 'exterior' if (i == 0 or i == n_spans) else 'interior'
        label = f"Apoyo {i} ({jtype:8s}, x={x:6.2f} m)"
        while True:
            raw = input(f"  {label}  h_col [m] (default 0.30): ").strip()
            if raw == '':
                h = 0.30
                break
            try:
                h = float(raw)
                if h <= 0:
                    print("    valor debe ser > 0")
                    continue
                break
            except ValueError:
                print("    ingresar un numero valido")
        joints.append({'x_m': x, 'h_col_m': h, 'type': jtype})

    beam['joints'] = joints

def name2idx(dname: str) -> int:
    """Busca el indice dO de un diametro por su nombre (ej. '5/8')."""
    for dO, info in REBAR_CATALOG_N.items():
        if info['name'] == dname:
            return dO
    return 0

def export_json(beam, result_ga, path):
    """Exporta los resultados a un archivo JSON."""
    top = result_ga['top3'][0]
    data = {
        'beam_id': beam['id'],
        'n_spans': beam['n_spans'],
        'top_result': {
            'total_weight_kg': top['total_weight_kg'],
            'feasible': top['feasible'],
            'zones_ok': top.get('zones_ok'),
        }
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Resultados exportados a {path}")
