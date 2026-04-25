import argparse
import os
import sys

from pipelines.training_pipeline import TrainingPipeline


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="GA_viga_completa: diseño óptimo de viga con GA"
    )
    parser.add_argument("--beam_id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--json_out", type=str, default=None)
    parser.add_argument("--pop", type=int, default=None)
    parser.add_argument("--gen", type=int, default=None)
    parser.add_argument("--source", type=str, choices=["dataset", "etabs"], default="dataset")
    parser.add_argument("--fc", type=float, default=210.0)
    return parser.parse_args()


def main():
    args = parse_arguments()

    config = {
        "args": args,
        "project_root": _HERE,
    }

    pipeline = TrainingPipeline(config=config)
    result = pipeline.run()

    print("\n=== RESULTADO PIPELINE INFERENCE ===")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Elapsed: {result.elapsed_seconds:.2f}s" if result.elapsed_seconds is not None else "Elapsed: N/A")
    print(f"Metrics: {result.metrics}")
    print(f"Artifacts: {result.artifacts}")


if __name__ == "__main__":
    main()