import json
import os
import numpy as np
from pipelines.base_pipeline import BasePipeline
import config.config_ga as config_ga
import config.config as config
from ga.beam_ga_complete import BeamGAComplete
import utils.visualizer as visualizer
import utils.cli_utils as cli_utils


class TrainingPipeline(BasePipeline):
    def load_data(self):
        args = self.config["args"]
        here = self.config["project_root"]
        dataset_path = os.path.join(here, "data", "dataset.json")

        if args.source == "etabs":
            import utils.etabs_extractor as etabs_extractor
            self.logger.info("Conectando a ETABS...")
            beam = etabs_extractor.extract_beam_from_etabs(args.fc)
            return beam

        self.logger.info("Cargando dataset desde JSON")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        valid_beams = [
            b for b in data
            if round(float(b["inputs"]["b_m"]), 2) in config.VARILLAS_POR_ANCHO
        ]

        if not valid_beams:
            raise ValueError("No hay vigas válidas en el dataset")

        if args.beam_id is not None:
            beam_map = {b["id"]: b for b in valid_beams}
            if args.beam_id not in beam_map:
                raise ValueError(
                    f"beam_id={args.beam_id} no encontrado. "
                    f"IDs disponibles: {list(beam_map.keys())[:10]}"
                )
            return beam_map[args.beam_id]

        rng = np.random.default_rng(args.seed)
        return valid_beams[int(rng.integers(0, len(valid_beams)))]

    def validate_data(self, beam):
        if not isinstance(beam, dict):
            return False
        return "inputs" in beam and "outputs" in beam and "id" in beam

    def preprocess(self, beam):
        # limpieza, transformación, conversión de formatos
        return beam

    def run_core(self, beam):
        args = self.config["args"]
        pop_size = args.pop or config_ga.POP_SIZE
        n_gen = args.gen or config_ga.N_GEN

        self.logger.info(f"Ejecutando GA (pop={pop_size}, gen={n_gen})")
        ga = BeamGAComplete(
            beam,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=args.seed
        )
        result = ga.run()
        result["beam"] = beam
        return result

    def evaluate(self, result):
        top3 = result.get("top3", [])
        best = top3[0] if top3 else None

        return {
            "generations_run": result.get("generations_run"),
            "elapsed_s": result.get("elapsed_s"),
            "best_feasible": best["feasible"] if best else None,
            "best_weight_kg": best["total_weight_kg"] if best else None,
            "best_fitness": best["fitness"] if best else None,
        }

    def save_artifacts(self, result):
        args = self.config["args"]
        beam = result["beam"]
        artifacts = {}

        if result.get("top3"):
            for r in result["top3"]:
                visualizer.generate_figure(beam, r, r["rank"], save_dir=args.save)
            artifacts["figures_saved"] = True

        visualizer.generate_convergence_figure(result, beam["id"], save_dir=args.save)
        artifacts["convergence_saved"] = True

        if args.json_out:
            cli_utils.export_json(result, beam, args.json_out)
            artifacts["json_saved"] = args.json_out

        return artifacts