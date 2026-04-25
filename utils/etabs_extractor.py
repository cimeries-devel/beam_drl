"""Extractor de datos de viga desde ETABS via COM.

Conecta a una instancia activa de ETABS, lee la viga seleccionada y retorna
un dict `beam` compatible con el GA (misma estructura que el dataset JSON).

Requisitos:
  - ETABS abierto con un modelo cargado
  - Exactamente 1 frame seleccionado
  - Combo de envolvente configurado para output en ETABS
  - comtypes instalado (`pip install comtypes`)

Uso:
  from etabs_extractor import extract_beam_from_etabs
  beam = extract_beam_from_etabs(fc_kg_cm2=210.0)
"""

import comtypes.client


def extract_beam_from_etabs(fc_kg_cm2: float = 210.0) -> dict:
    """Extrae geometría y envolvente de momentos de la viga seleccionada en ETABS.

    Parameters
    ----------
    fc_kg_cm2 : float
        Resistencia del concreto en kg/cm² (input manual).

    Returns
    -------
    dict compatible con el GA:
        {
            'id': str,
            'inputs': {'b_m': float, 'h_m': float, 'fc_kg_cm2': float},
            'outputs': {
                'x_m': list[float],
                'M_tonf_m_max': list[float],   # envolvente positiva (M+)
                'M_tonf_m_min': list[float],   # envolvente negativa (M-)
            }
        }
    """
    # ------------------------------------------------------------------
    # 1. Conectar a ETABS
    # ------------------------------------------------------------------
    try:
        helper = comtypes.client.CreateObject('ETABSv1.Helper')
        helper = helper.QueryInterface(comtypes.gen.ETABSv1.cHelper)
        myETABSObject = helper.GetObject("CSI.ETABS.API.ETABSObject")
    except (OSError, comtypes.COMError):
        raise RuntimeError(
            "No se encontró una instancia de ETABS activa. "
            "Abre ETABS con un modelo cargado e intenta de nuevo."
        )

    SapModel = myETABSObject.SapModel

    # Unidades: Ton-m-°C
    TON_M_C = 12
    SapModel.SetPresentUnits(TON_M_C)

    # ------------------------------------------------------------------
    # 2. Obtener frame seleccionado
    # ------------------------------------------------------------------
    sel = SapModel.SelectObj.GetSelected()
    frame_names = sel[2]  # tupla de nombres

    if not frame_names:
        raise ValueError(
            "No hay ningún elemento seleccionado en ETABS. "
            "Selecciona exactamente 1 viga e intenta de nuevo."
        )
    if len(frame_names) > 1:
        raise ValueError(
            f"Hay {len(frame_names)} elementos seleccionados ({frame_names}). "
            "Selecciona exactamente 1 viga."
        )

    frame_name = frame_names[0]

    # ------------------------------------------------------------------
    # 3. Geometría de la sección
    # ------------------------------------------------------------------
    seccion = SapModel.FrameObj.GetSection(frame_name)
    sec_name = seccion[0]
    props = SapModel.PropFrame.GetRectangle(Name=sec_name)
    # props[2] = altura (m), props[3] = base (m)
    h_m = float(props[2])
    b_m = float(props[3])

    # ------------------------------------------------------------------
    # 4. Fuerzas del frame (envolvente)
    # ------------------------------------------------------------------
    res = SapModel.Results.FrameForce(Name=frame_name, ItemTypeElm=0)
    # res[3] = Elm, res[4] = ElmSta, res[6] = StepType, res[13] = M3

    elms = res[3]       # tupla de sub-elementos ("86-1", "86-2", ...)
    stations = res[4]   # tupla de ElmSta
    step_types = res[6] # tupla de "Max"/"Min"
    m3_vals = res[13]   # tupla de momentos M3

    if not elms:
        raise ValueError(
            f"No se obtuvieron resultados para el frame '{frame_name}'. "
            "Verifica que haya un combo de envolvente configurado para output."
        )

    # ------------------------------------------------------------------
    # 5. Reconstruir eje x continuo
    # ------------------------------------------------------------------
    # Agrupar por (sub-elemento, StepType)
    # Los sub-elementos se ordenan por sufijo numérico: "86-1" < "86-2"
    unique_elms = []
    seen = set()
    for e in elms:
        if e not in seen:
            unique_elms.append(e)
            seen.add(e)

    # Ordenar sub-elementos por sufijo
    def _elm_sort_key(name: str):
        parts = name.rsplit('-', 1)
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

    unique_elms.sort(key=_elm_sort_key)

    # Calcular offset acumulado para cada sub-elemento
    # El offset del sub-elemento N+1 = max ElmSta del sub-elemento N
    elm_offsets = {}
    cumulative = 0.0

    for elm_name in unique_elms:
        elm_offsets[elm_name] = cumulative
        # max station de este sub-elemento (de cualquier StepType)
        max_sta = max(
            stations[i] for i in range(len(elms))
            if elms[i] == elm_name
        )
        cumulative = cumulative + max_sta

    # Construir vectores x, M_max, M_min
    x_max, m_max = [], []
    x_min, m_min = [], []

    for elm_name in unique_elms:
        for step in ('Max', 'Min'):
            indices = [
                i for i in range(len(elms))
                if elms[i] == elm_name and step_types[i] == step
            ]
            offset = elm_offsets[elm_name]
            target_x = x_max if step == 'Max' else x_min
            target_m = m_max if step == 'Max' else m_min

            for idx in indices:
                x_val = offset + float(stations[idx])
                m_val = float(m3_vals[idx])
                target_x.append(x_val)
                target_m.append(m_val)

    # ------------------------------------------------------------------
    # 6. Eliminar duplicados en juntas
    # ------------------------------------------------------------------
    # En la junta entre sub-elementos, el último punto de N y el primero
    # de N+1 tienen el mismo x y mismo M. Quitar duplicados consecutivos.
    def _dedup(xs, ms):
        if not xs:
            return xs, ms
        x_out, m_out = [xs[0]], [ms[0]]
        for i in range(1, len(xs)):
            if abs(xs[i] - x_out[-1]) < 1e-9 and abs(ms[i] - m_out[-1]) < 1e-9:
                continue
            x_out.append(xs[i])
            m_out.append(ms[i])
        return x_out, m_out

    x_max, m_max = _dedup(x_max, m_max)
    x_min, m_min = _dedup(x_min, m_min)

    # ------------------------------------------------------------------
    # 7. Normalizar x para que empiece en 0
    # ------------------------------------------------------------------
    x0 = min(x_max[0], x_min[0]) if x_max and x_min else 0.0
    x_max = [x - x0 for x in x_max]
    x_min = [x - x0 for x in x_min]

    # Verificar que ambos ejes x coincidan
    if len(x_max) != len(x_min):
        raise ValueError(
            f"Los vectores x de Max ({len(x_max)}) y Min ({len(x_min)}) "
            "tienen distinta longitud. Revisa la configuración de output en ETABS."
        )

    # ------------------------------------------------------------------
    # 8. Armar dict beam
    # ------------------------------------------------------------------
    beam = {
        'id': frame_name,
        'inputs': {
            'b_m': round(b_m, 4),
            'h_m': round(h_m, 4),
            'fc_kg_cm2': float(fc_kg_cm2),
        },
        'outputs': {
            'x_m': x_max,
            'M_tonf_m_max': m_max,
            'M_tonf_m_min': m_min,
        },
    }

    return beam
