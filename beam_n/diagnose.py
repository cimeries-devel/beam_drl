import json
from .beam_ga_n import BeamGANSpans
from .config_n import ensure_joints
from .etabs_extractor_n import extract_multispan_beam

beam = extract_multispan_beam(fc_kg_cm2=210.0, combo='ENV')
ensure_joints(beam)

# h_col reales de la viga actual (25cm, 25cm, 170cm)
beam['joints'][0]['h_col_m'] = 0.25
beam['joints'][1]['h_col_m'] = 0.25
beam['joints'][2]['h_col_m'] = 1.70

print(f"Viga: {beam['id']}, {beam['n_spans']} tramos")
for i, j in enumerate(beam['joints']):
    print(f"  Apoyo {i}: h_col={j['h_col_m']}m, tipo={j['type']}")

ga = BeamGANSpans(beam, pop_size=150, n_gen=200, seed=42)
res = ga.run()
r = res['top3'][0]
print('Weight:   ', r['total_weight_kg'])
print('Fitness:  ', r['fitness'])
print('Feasible: ', r['feasible'])
print('Zones OK: ', r['zones_ok'], '/', len(r['zone_results']))
print('Violations:')
for k, v in r['violations'].items():
    print(f'  {k}: {v}')
print('\nZonas que FALLAN:')
for zid, zr in r['zone_results'].items():
    ze = r['zone_extents'].get(zid, {})
    if ze.get('exists') and not zr['ok']:
        print(f'  {zid}: phiMn={zr["phi_Mn"]:.3f} < Mu={zr["Mu"]:.3f}, n_bast={zr["n_bast"]}')
print('\nZonas OK con bastones:')
for zid, zr in r['zone_results'].items():
    ze = r['zone_extents'].get(zid, {})
    if ze.get('exists') and zr['ok'] and zr['n_bast'] > 0:
        print(f'  {zid}: phiMn={zr["phi_Mn"]:.3f} >= Mu={zr["Mu"]:.3f}, n_bast={zr["n_bast"]}')
