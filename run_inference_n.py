"""CLI principal para el GA de vigas de N tramos."""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Mantener compatibilidad de paths si es necesario
sys.path.insert(0, os.path.join(_HERE, '..', 'GA_viga_completa'))

from config_n import ensure_joints, CORRIDO_SIMETRICO
from utils_n import prompt_h_col, export_json
from visualization_n import generate_figure_n, plot_convergence
from beam_ga_n import BeamGANSpans

def main():
    parser = argparse.ArgumentParser(description="GA para vigas multispan.")
    parser.add_argument('--source', choices=['json', 'etabs'], default='json')
    parser.add_argument('--json_beam', type=str, help="Path al JSON de la viga")
    parser.add_argument('--fc', type=float, default=210.0)
    parser.add_argument('--pop', type=int, default=None)
    parser.add_argument('--gen', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', type=str, help="Folder para guardar figs")
    parser.add_argument('--json_out', type=str, help="Exportar resultado a JSON")
    parser.add_argument('--h_col_default', action='store_true')
    parser.add_argument('--item_type_elm', type=int, default=0)
    parser.add_argument('--combo', type=str, default=None)
    parser.add_argument('--no_sim', action='store_true', help="Desactivar corrido simetrico")

    args = parser.parse_args()
    corrido_sim = not args.no_sim and CORRIDO_SIMETRICO

    # 1. Cargar datos de la viga
    if args.source == 'etabs':
        from etabs_extractor_n import extract_multispan_beam
        beam = extract_multispan_beam(
            fc_kg_cm2=args.fc,
            item_type_elm=args.item_type_elm,
            combo=args.combo,
        )
    else:
        json_path = args.json_beam or os.path.join(_HERE, 'data/dataset.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            beam = json.load(f)
    
    print(f"Viga cargada: {beam['id']}, {beam['n_spans']} tramos")
    ensure_joints(beam)

    if not args.h_col_default:
        prompt_h_col(beam)
    else:
        print("Usando h_col default (0.30m)")

    # 2. Ejecutar Algoritmo Genetico
    ga = BeamGANSpans(
        beam,
        pop_size=args.pop,
        n_gen=args.gen,
        seed=args.seed,
        corrido_simetrico=corrido_sim,
    )
    
    print(f"\nIniciando GA: pop={ga.pop_size}, gen={ga.n_gen}")
    result_ga = ga.run()

    # 3. Procesar resultados
    print(f"\nGA finalizado en {result_ga['elapsed_s']:.2f}s")
    top = result_ga['top3'][0]
    print(f"Mejor resultado: {top['total_weight_kg']:.2f}kg, Feasible: {top['feasible']}")

    # 4. Visualizacion y Exportacion
    if args.save:
        for r in result_ga['top3']:
            generate_figure_n(beam, r, r['rank'], save_dir=args.save)
        plot_convergence(result_ga, beam['id'], save_dir=args.save)

    if args.json_out:
        export_json(beam, result_ga, args.json_out)

if __name__ == '__main__':
    main()
