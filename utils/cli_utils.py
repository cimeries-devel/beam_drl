import json
from config.config_ga import ZONE_IDS

def get_corrido_spec(corr_bars, diam_a, diam_b):
    n_a = sum(1 for _, _, d in corr_bars if d == diam_a)
    n_b = sum(1 for _, _, d in corr_bars if d == diam_b)
    if diam_a == diam_b or n_b == 0:
        return f'{len(corr_bars)}ø{diam_a}"'
    return f'{n_a}ø{diam_a}"+{n_b}ø{diam_b}"'

def print_summary_table(ga_result, beam):
    print('\n' + '=' * 80)
    print(f'  Viga #{beam["id"]} | b={round(beam["inputs"]["b_m"],2)}m '
          f'h={beam["inputs"]["h_m"]}m fc={beam["inputs"]["fc_kg_cm2"]:.0f} '
          f'L={beam["outputs"]["x_m"][-1]:.2f}m')
    print(f'  Generaciones: {ga_result["generations_run"]} | '
          f'Tiempo: {ga_result["elapsed_s"]:.1f}s | '
          f'Early stop: {ga_result["early_stop"]}')
    print('=' * 80)
    header = f'{"RANK":^4} | {"W_total":^9} | {"W_corrido":^9} | {"W_bast":^7} | ' \
             f'{"Zonas OK":^9} | {"Corrido":^20} | {"Factible":^9}'
    print(header)
    print('-' * len(header))

    for r in ga_result['top3']:
        spec = get_corrido_spec(r['corrido_bars'], r['diam_A'], r['diam_B'])
        row = (f'  {r["rank"]:^2}  | '
               f'{r["total_weight_kg"]:>7.2f}kg | '
               f'{r["corrido_weight_kg"]:>7.2f}kg | '
               f'{r["baston_weight_kg"]:>5.2f}kg | '
               f'  {r["zones_ok"]}/6     | '
               f'{spec:^20} | '
               f'{"Sí" if r["feasible"] else "No":^9}')
        print(row)

    print('=' * 80)

    top3 = ga_result.get('top3', [])
    if top3:
        best = top3[0]
        print('\nDetalle de zonas (Rank 1):')
        print(f'  {"Zona":^8} | {"Mu (t-m)":^10} | {"phiMn(t-m)":^10} | {"OK":^4}')
        for zid in ZONE_IDS:
            v = best['zone_results'].get(zid, {})
            ok = 'SI' if v.get('ok') else 'NO'
            print(f'  {zid:^8} | {v.get("Mu", 0):^10.3f} | {v.get("phi_Mn", 0):^10.3f} | {ok:^4}')
    print()

def print_diagnostic(best_result, beam):
    """Imprime diagnóstico detallado cuando el mejor diseño no es factible."""
    inp = beam['inputs']
    b = round(float(inp['b_m']), 2)
    h = float(inp['h_m'])
    fc = float(inp['fc_kg_cm2'])

    outs = beam['outputs']
    if 'M_tonf_m_max' in outs and 'M_tonf_m_min' in outs:
        mu_pos = max(outs['M_tonf_m_max'])
        mu_neg = abs(min(outs['M_tonf_m_min']))
    elif 'M_tonf_m' in outs:
        ms = outs['M_tonf_m']
        mu_pos = max(m for m in ms) if any(m > 0 for m in ms) else 0.0
        mu_neg = abs(min(m for m in ms)) if any(m < 0 for m in ms) else 0.0
    else:
        mu_pos, mu_neg = 0.0, 0.0

    print()
    print('!' * 80)
    print('  ALERTA: EL GA NO ENCONTRO UN DISENO FACTIBLE')
    print('!' * 80)

    print(f'\n  Seccion: b={b}m x h={h}m | fc={fc:.0f} kg/cm2')
    print(f'  Demanda maxima M+ (BOT): {mu_pos:.2f} tonf-m')
    print(f'  Demanda maxima M- (TOP): {mu_neg:.2f} tonf-m')

    zr = best_result.get('zone_results', {})
    zonas_fail = [(zid, v) for zid, v in zr.items() if not v.get('ok', True)]

    if zonas_fail:
        print(f'\n  Zonas que fallan ({len(zonas_fail)}/6):')
        for zid, v in zonas_fail:
            mu = v.get('Mu', 0)
            phi = v.get('phi_Mn', 0)
            if mu > 0.01:
                deficit = (mu - phi) / mu * 100
                print(f'    {zid:<12}  Mu={mu:>7.2f}  phiMn={phi:>7.2f}  '
                      f'deficit={deficit:.1f}%')
            else:
                print(f'    {zid:<12}  Mu={mu:>7.2f}  phiMn={phi:>7.2f}')

    w = best_result.get('total_weight_kg', 0)
    n_bars = len(best_result.get('corrido_bars', []))

    if w < 0.01 or n_bars == 0:
        print('\n  El GA no logro generar un corrido con barras.')
        print('  Causa probable: la seccion es demasiado pequena para la demanda.')
        print(f'  Recomendacion: aumentar h (peralte) o b (ancho) de la seccion.')
    else:
        print(f'\n  Mejor intento: W={w:.2f}kg | corrido={n_bars} barras | '
              f'{best_result.get("zones_ok", 0)}/6 zonas OK')
        print('  Causa probable: el GA no convergio a una solucion factible.')
        print('  Recomendacion: aumentar generaciones (--gen 400) o poblacion (--pop 300).')

    rho = best_result.get('rho_corrido', 0)
    from domain.section_calculator import compute_rho_min
    rho_min = compute_rho_min(fc)
    if rho < rho_min and n_bars > 0:
        print(f'\n  Violacion normativa: rho={rho:.6f} < rho_min={rho_min:.6f}')

    print()
    print('!' * 80)
    print()

def export_json(ga_result, beam, json_out):
    inp = beam['inputs']
    b = round(float(inp['b_m']), 2)

    top3_out = []
    for r in ga_result['top3']:
        bastones_out = {}
        for zid in ZONE_IDS:
            zv = r['zone_results'].get(zid, {})
            bastones_out[zid] = {
                'n_barras': zv.get('n_bast', 0),
                'diam':     r['diam_B'],
                'phi_mn':   zv.get('phi_Mn', 0),
                'Mu':       zv.get('Mu', 0),
                'ok':       zv.get('ok', False),
                'weight_kg': zv.get('weight_kg', 0),
            }

        top3_out.append({
            'rank': r['rank'],
            'diam_A': r['diam_A'],
            'diam_B': r['diam_B'],
            'corrido': {
                'n_barras':    len(r['corrido_bars']),
                'as_cm2':      r['as_corrido'],
                'phi_mn_pos':  r['phi_mn_corrido']['positive'],
                'phi_mn_neg':  r['phi_mn_corrido']['negative'],
                'weight_kg':   r['corrido_weight_kg'],
            },
            'bastones': bastones_out,
            'total_weight_kg': r['total_weight_kg'],
            'corrido_weight_kg': r['corrido_weight_kg'],
            'baston_weight_kg':  r['baston_weight_kg'],
            'feasible': r['feasible'],
            'zones_ok': r['zones_ok'],
        })

    out = {
        'beam_id': beam['id'],
        'b': b,
        'h': float(inp['h_m']),
        'fc': float(inp['fc_kg_cm2']),
        'L':  float(beam['outputs']['x_m'][-1]),
        'top3': top3_out,
        'generations_run': ga_result['generations_run'],
        'elapsed_s': ga_result['elapsed_s'],
    }

    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'  JSON guardado: {json_out}')