"""Extractor de datos de viga de N tramos desde ETABS via COM.

Conecta a una instancia activa de ETABS, lee los frames seleccionados
(N tramos de una viga continua), los ordena por posicion, valida
continuidad y seccion, y retorna un dict con la envolvente de momentos
continua y la informacion de cada tramo.

Requisitos:
  - ETABS abierto con un modelo cargado y analizado
  - N frames seleccionados (los tramos de la viga)
  - Combo de envolvente configurado para output
  - comtypes instalado (`pip install comtypes`)

Uso:
  from etabs_extractor_n import extract_multispan_beam
  beam = extract_multispan_beam(fc_kg_cm2=210.0)
"""

import sys
import comtypes.client


# ---------------------------------------------------------------------------
# Conexion a ETABS
# ---------------------------------------------------------------------------
def _connect_etabs():
    """Retorna (SapModel, myETABSObject)."""
    try:
        helper = comtypes.client.CreateObject('ETABSv1.Helper')
        helper = helper.QueryInterface(comtypes.gen.ETABSv1.cHelper)
        myETABSObject = helper.GetObject("CSI.ETABS.API.ETABSObject")
    except (OSError, comtypes.COMError):
        raise RuntimeError(
            "No se encontro una instancia de ETABS activa. "
            "Abre ETABS con un modelo cargado e intenta de nuevo."
        )
    return myETABSObject.SapModel, myETABSObject


# ---------------------------------------------------------------------------
# Utilidades de geometria
# ---------------------------------------------------------------------------
def _get_joint_coords(SapModel, joint_name: str) -> tuple:
    """Retorna (x, y, z) de un joint."""
    res = SapModel.PointObj.GetCoordCartesian(Name=joint_name)
    return (float(res[0]), float(res[1]), float(res[2]))


def _get_frame_joints(SapModel, frame_name: str) -> tuple:
    """Retorna (joint_i, joint_j) de un frame."""
    res = SapModel.FrameObj.GetPoints(Name=frame_name)
    return (res[0], res[1])


def _detect_orientation(coords_by_frame: dict) -> str:
    """Autodetecta orientacion ('X' o 'Y') segun que eje varia mas.

    Parameters
    ----------
    coords_by_frame : dict
        {frame_name: ((xi, yi, zi), (xj, yj, zj))}

    Returns
    -------
    'X' o 'Y'
    """
    dx_total = 0.0
    dy_total = 0.0
    for (ci, cj) in coords_by_frame.values():
        dx_total += abs(cj[0] - ci[0])
        dy_total += abs(cj[1] - ci[1])
    return 'X' if dx_total >= dy_total else 'Y'


def _coord_along_axis(coord_3d: tuple, axis: str) -> float:
    """Extrae la coordenada segun el eje de orientacion."""
    return coord_3d[0] if axis == 'X' else coord_3d[1]


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------
def extract_multispan_beam(
    fc_kg_cm2: float = 210.0,
    item_type_elm: int = 0,
    combo: str = 'ENV',
) -> dict:
    """Extrae geometria y envolvente de momentos de N tramos seleccionados.

    Parameters
    ----------
    fc_kg_cm2 : float
        Resistencia del concreto en kg/cm2 (input manual).
    item_type_elm : int
        Valor de ItemTypeElm para FrameForce (0 o 1). Ajustar segun
        como ETABS reporta los sub-elementos.

    Returns
    -------
    dict :
        {
            'id': str,                   # nombres de frames concatenados
            'n_spans': int,
            'inputs': {
                'b_m': float,
                'h_m': float,
                'fc_kg_cm2': float,
            },
            'supports_x_m': list[float],  # n+1 posiciones de apoyo
            'spans': [
                {
                    'frame': str,
                    'L_m': float,
                    'x0': float,
                    'x1': float,
                    'b_m': float,
                    'h_m': float,
                },
                ...
            ],
            'outputs': {
                'x_m': list[float],
                'M_tonf_m_max': list[float],
                'M_tonf_m_min': list[float],
            }
        }
    """
    SapModel, _ = _connect_etabs()

    # Unidades: Ton-m-C
    SapModel.SetPresentUnits(12)

    # ------------------------------------------------------------------
    # 0. Configurar combo de envolvente para output
    # ------------------------------------------------------------------
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    ret = SapModel.Results.Setup.SetComboSelectedForOutput(combo)
    if ret != 0:
        raise ValueError(
            f"No se pudo seleccionar el combo '{combo}' para output. "
            "Verifica que exista en ETABS."
        )

    # ------------------------------------------------------------------
    # 1. Obtener frames seleccionados
    # ------------------------------------------------------------------
    sel = SapModel.SelectObj.GetSelected()
    # sel[1] = tipos, sel[2] = nombres
    obj_types = sel[1]
    obj_names = sel[2]

    if not obj_names:
        raise ValueError(
            "No hay elementos seleccionados en ETABS. "
            "Selecciona los tramos de la viga."
        )

    # Filtrar solo frames (tipo 2 en ETABS API)
    frame_names = [
        name for name, otype in zip(obj_names, obj_types)
        if otype == 2
    ]
    if not frame_names:
        raise ValueError(
            "Ningun frame seleccionado. Los objetos seleccionados no son frames."
        )

    # ------------------------------------------------------------------
    # 2. Leer joints y coordenadas de cada frame
    # ------------------------------------------------------------------
    coords_by_frame = {}   # {frame: ((xi,yi,zi), (xj,yj,zj))}
    joints_by_frame = {}   # {frame: (joint_i, joint_j)}

    for fname in frame_names:
        ji, jj = _get_frame_joints(SapModel, fname)
        ci = _get_joint_coords(SapModel, ji)
        cj = _get_joint_coords(SapModel, jj)
        joints_by_frame[fname] = (ji, jj)
        coords_by_frame[fname] = (ci, cj)

    # ------------------------------------------------------------------
    # 3. Autodetectar orientacion y ordenar frames izq → der
    # ------------------------------------------------------------------
    axis = _detect_orientation(coords_by_frame)

    def _frame_start(fname):
        ci, cj = coords_by_frame[fname]
        return min(_coord_along_axis(ci, axis),
                   _coord_along_axis(cj, axis))

    frame_names_sorted = sorted(frame_names, key=_frame_start)

    # Asegurar que cada frame tenga su punto_i < punto_j en el eje
    # (si esta invertido, swapear para que la logica sea consistente)
    ordered_frames = []
    for fname in frame_names_sorted:
        ci, cj = coords_by_frame[fname]
        pi = _coord_along_axis(ci, axis)
        pj = _coord_along_axis(cj, axis)
        ji, jj = joints_by_frame[fname]
        if pi <= pj:
            ordered_frames.append({
                'frame': fname,
                'joint_i': ji, 'joint_j': jj,
                'coord_i': pi, 'coord_j': pj,
            })
        else:
            # Invertir para que i sea el de menor coordenada
            ordered_frames.append({
                'frame': fname,
                'joint_i': jj, 'joint_j': ji,
                'coord_i': pj, 'coord_j': pi,
            })

    n_spans = len(ordered_frames)

    # ------------------------------------------------------------------
    # 4. Validar continuidad (joint_j[i] == joint_i[i+1])
    # ------------------------------------------------------------------
    for i in range(n_spans - 1):
        jj_curr = ordered_frames[i]['joint_j']
        ji_next = ordered_frames[i + 1]['joint_i']
        cj_curr = ordered_frames[i]['coord_j']
        ci_next = ordered_frames[i + 1]['coord_i']
        if jj_curr != ji_next and abs(cj_curr - ci_next) > 0.01:
            raise ValueError(
                f"Los tramos '{ordered_frames[i]['frame']}' y "
                f"'{ordered_frames[i+1]['frame']}' no son continuos. "
                f"Joint final={jj_curr} (x={cj_curr:.3f}), "
                f"joint inicial={ji_next} (x={ci_next:.3f}). "
                f"Verifica la seleccion."
            )

    # ------------------------------------------------------------------
    # 5. Leer seccion de cada frame y validar
    # ------------------------------------------------------------------
    sections = []
    for of in ordered_frames:
        fname = of['frame']
        sec_name = SapModel.FrameObj.GetSection(fname)[0]
        props = SapModel.PropFrame.GetRectangle(Name=sec_name)
        h_m = float(props[2])
        b_m = float(props[3])
        sections.append({'b_m': round(b_m, 4), 'h_m': round(h_m, 4)})

    # Usar la seccion del primer tramo como referencia
    b_ref = sections[0]['b_m']
    h_ref = sections[0]['h_m']
    all_same_section = all(
        abs(s['b_m'] - b_ref) < 0.001 and abs(s['h_m'] - h_ref) < 0.001
        for s in sections
    )
    if not all_same_section:
        diff_spans = [
            f"  Tramo '{ordered_frames[i]['frame']}': b={sections[i]['b_m']}m, h={sections[i]['h_m']}m"
            for i in range(n_spans)
        ]
        print(
            "ADVERTENCIA: No todos los tramos tienen la misma seccion:\n"
            + "\n".join(diff_spans)
            + f"\nSe usara la seccion del primer tramo (b={b_ref}m, h={h_ref}m) "
            "como referencia global."
        )

    # ------------------------------------------------------------------
    # 6. Calcular posiciones de apoyos y construir info de tramos
    # ------------------------------------------------------------------
    # Normalizar: el primer apoyo esta en x=0
    x_origin = ordered_frames[0]['coord_i']

    spans_info = []
    supports_x = []

    for i, of in enumerate(ordered_frames):
        x0 = of['coord_i'] - x_origin
        x1 = of['coord_j'] - x_origin
        L = x1 - x0
        spans_info.append({
            'frame': of['frame'],
            'L_m': round(L, 4),
            'x0': round(x0, 4),
            'x1': round(x1, 4),
            'b_m': sections[i]['b_m'],
            'h_m': sections[i]['h_m'],
        })
        if i == 0:
            supports_x.append(round(x0, 4))
        supports_x.append(round(x1, 4))

    # ------------------------------------------------------------------
    # 7. Extraer envolvente de momentos (todos los frames)
    # ------------------------------------------------------------------
    all_x_max = []
    all_m_max = []
    all_x_min = []
    all_m_min = []

    for of in ordered_frames:
        fname = of['frame']
        offset_global = of['coord_i'] - x_origin

        res = SapModel.Results.FrameForce(Name=fname, ItemTypeElm=item_type_elm)
        elms = res[3]
        stations = res[4]
        step_types = res[6]
        m3_vals = res[13]

        if not elms:
            raise ValueError(
                f"No se obtuvieron resultados para el frame '{fname}'. "
                "Verifica que haya un combo de envolvente configurado."
            )

        # Agrupar sub-elementos y ordenar por sufijo numerico
        unique_elms = []
        seen = set()
        for e in elms:
            if e not in seen:
                unique_elms.append(e)
                seen.add(e)

        def _elm_sort_key(name: str):
            parts = name.rsplit('-', 1)
            return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

        unique_elms.sort(key=_elm_sort_key)

        # Calcular offsets acumulados dentro del frame
        # (mismo enfoque del extractor original: offset del sub-elemento N+1
        #  = max ElmSta del sub-elemento N)
        elm_offsets = {}
        cumulative = 0.0
        for elm_name in unique_elms:
            elm_offsets[elm_name] = cumulative
            max_sta = max(
                stations[i] for i in range(len(elms))
                if elms[i] == elm_name
            )
            cumulative += max_sta

        # Construir vectores x, M para Max y Min
        for elm_name in unique_elms:
            for step in ('Max', 'Min'):
                indices = [
                    i for i in range(len(elms))
                    if elms[i] == elm_name and step_types[i] == step
                ]
                sub_offset = elm_offsets[elm_name]

                target_x = all_x_max if step == 'Max' else all_x_min
                target_m = all_m_max if step == 'Max' else all_m_min

                for idx in indices:
                    x_val = offset_global + sub_offset + float(stations[idx])
                    m_val = float(m3_vals[idx])
                    target_x.append(round(x_val, 6))
                    target_m.append(round(m_val, 6))

    # ------------------------------------------------------------------
    # 8. Eliminar duplicados consecutivos (juntas entre sub-elementos)
    # ------------------------------------------------------------------
    def _dedup(xs, ms):
        if not xs:
            return xs, ms
        x_out, m_out = [xs[0]], [ms[0]]
        for i in range(1, len(xs)):
            if abs(xs[i] - x_out[-1]) < 1e-6 and abs(ms[i] - m_out[-1]) < 1e-6:
                continue
            x_out.append(xs[i])
            m_out.append(ms[i])
        return x_out, m_out

    all_x_max, all_m_max = _dedup(all_x_max, all_m_max)
    all_x_min, all_m_min = _dedup(all_x_min, all_m_min)

    # Verificar que ambos ejes x tengan la misma longitud
    if len(all_x_max) != len(all_x_min):
        # Intentar interpolar al eje x mas denso
        import numpy as np
        x_union = sorted(set(all_x_max + all_x_min))
        m_max_interp = list(np.interp(x_union, all_x_max, all_m_max))
        m_min_interp = list(np.interp(x_union, all_x_min, all_m_min))
        all_x_max = x_union
        all_m_max = m_max_interp
        all_x_min = x_union
        all_m_min = m_min_interp

    # ------------------------------------------------------------------
    # 9. Armar dict de salida
    # ------------------------------------------------------------------
    beam_id = '+'.join(of['frame'] for of in ordered_frames)

    beam = {
        'id': beam_id,
        'n_spans': n_spans,
        'inputs': {
            'b_m': b_ref,
            'h_m': h_ref,
            'fc_kg_cm2': float(fc_kg_cm2),
        },
        'supports_x_m': supports_x,
        'spans': spans_info,
        'outputs': {
            'x_m': all_x_max,
            'M_tonf_m_max': all_m_max,
            'M_tonf_m_min': all_m_min,
        },
    }

    return beam


# ---------------------------------------------------------------------------
# CLI rapido para pruebas
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import json
    import argparse

    parser = argparse.ArgumentParser(description='Extraer viga N tramos de ETABS')
    parser.add_argument('--fc', type=float, default=210.0,
                        help='fc en kg/cm2 (default: 210)')
    parser.add_argument('--item_type_elm', type=int, default=0,
                        help='ItemTypeElm para FrameForce (0 o 1)')
    parser.add_argument('--combo', type=str, default='ENV',
                        help='Nombre del combo de envolvente (default: ENV)')
    parser.add_argument('--json_out', type=str, default=None,
                        help='Ruta para exportar JSON')
    args = parser.parse_args()

    beam = extract_multispan_beam(
        fc_kg_cm2=args.fc,
        item_type_elm=args.item_type_elm,
        combo=args.combo,
    )

    # Resumen en consola
    print(f"\nViga: {beam['id']}")
    print(f"Tramos: {beam['n_spans']}")
    print(f"Seccion: b={beam['inputs']['b_m']}m x h={beam['inputs']['h_m']}m")
    print(f"fc: {beam['inputs']['fc_kg_cm2']} kg/cm2")
    print(f"Apoyos en x: {beam['supports_x_m']}")
    print()
    for i, sp in enumerate(beam['spans']):
        print(f"  Tramo {i+1}: frame='{sp['frame']}', "
              f"L={sp['L_m']:.3f}m, x=[{sp['x0']:.3f}, {sp['x1']:.3f}]")
    print(f"\nPuntos en envolvente: {len(beam['outputs']['x_m'])}")
    print(f"Rango x: [{beam['outputs']['x_m'][0]:.3f}, {beam['outputs']['x_m'][-1]:.3f}]")

    m_max = beam['outputs']['M_tonf_m_max']
    m_min = beam['outputs']['M_tonf_m_min']
    print(f"M_max rango: [{min(m_max):.3f}, {max(m_max):.3f}] tonf-m")
    print(f"M_min rango: [{min(m_min):.3f}, {max(m_min):.3f}] tonf-m")

    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(beam, f, indent=2, ensure_ascii=False)
        print(f"\nJSON exportado a: {args.json_out}")
