import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from ga.beam_ga_complete import BeamGAComplete
import config.config_ga as config_ga

def run_experiment(name, params, beam, seed=42):
    print(f"\n>>> Ejecutando Experimento: {name}")
    ga = BeamGAComplete(beam, seed=seed, **params)
    t0 = time.time()
    result = ga.run()
    elapsed = time.time() - t0
    
    top1 = result['top3'][0] if result['top3'] else None
    
    metrics = {
        'Experiment': name,
        'Best Fitness': result['history_best'][-1],
        'Best Weight (kg)': top1['total_weight_kg'] if top1 else float('inf'),
        'Feasible': top1['feasible'] if top1 else False,
        'Time (s)': elapsed,
        'Gens': result['generations_run']
    }
    return metrics, result['history_best']

def main():
    # Cargar una viga de ejemplo del dataset
    dataset_path = "data/dataset.json"
    with open(dataset_path, "r") as f:
        data = json.load(f)
    beam = data[0] # Usar la primera viga para consistencia
    
    experiments = [
        ("Baseline", {}),
        ("Var1: Alta Mutación", {"p_mut_oni": 0.25, "p_mut_choice": 0.15}),
        ("Var2: Gran Población", {"pop_size": 300, "n_gen": 100})
    ]
    
    results_metrics = []
    histories = {}
    
    for name, params in experiments:
        metrics, history = run_experiment(name, params, beam)
        results_metrics.append(metrics)
        histories[name] = history
        
    # Crear Tabla Estándar
    df = pd.DataFrame(results_metrics)
    print("\n=== RESULTADOS COMPARABLES ===")
    print(df.to_string(index=False))
    df.to_csv("logs/experiment_results.csv", index=False)
    
    # Crear Gráfico Clave: Best-so-far (Convergencia)
    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        plt.plot(history, label=name)
    
    plt.title(f"Convergencia GA - Viga {beam['id']}")
    plt.xlabel("Generación")
    plt.ylabel("Best Fitness (Log scale)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig("test_figs/experiment_convergence.png")
    print("\nGráfico guardado en test_figs/experiment_convergence.png")

if __name__ == "__main__":
    if not os.path.exists("logs"): os.makedirs("logs")
    if not os.path.exists("test_figs"): os.makedirs("test_figs")
    main()
