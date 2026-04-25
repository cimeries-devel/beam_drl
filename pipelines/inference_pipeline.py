import json
from pipelines.base_pipeline import BasePipeline


class InferencePipeline(BasePipeline):
    def load_data(self):
        input_path = self.config["input_path"]
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def validate_data(self, data):
        return isinstance(data, dict) and "beam_id" in data

    def preprocess(self, data):
        return data

    def run_core(self, data):
        # Aquí iría la lógica de inferencia con un modelo o solución ya guardada
        return {
            "input": data,
            "prediction": None
        }

    def save_artifacts(self, result):
        output_path = self.config.get("output_path")
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        return {"output_path": output_path}