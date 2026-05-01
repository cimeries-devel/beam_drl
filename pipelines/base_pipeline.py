from dataclasses import dataclass
from typing import Any, Dict, Optional
import time
import logging


@dataclass
class PipelineResult:
    success: bool
    message: str
    metrics: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None
    elapsed_seconds: Optional[float] = None


class BasePipeline:
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def load_data(self):
        raise NotImplementedError

    def validate_data(self, data):
        return True

    def preprocess(self, data):
        return data

    def run_core(self, data):
        raise NotImplementedError

    def evaluate(self, result):
        return {}

    def save_artifacts(self, result):
        return {}

    def run(self) -> PipelineResult:
        start = time.time()

        try:
            self.logger.info("Iniciando pipeline")
            data = self.load_data()

            self.logger.info("Validando datos")
            if not self.validate_data(data):
                return PipelineResult(
                    success=False,
                    message="La validación de datos falló"
                )

            self.logger.info("Preprocesando datos")
            processed_data = self.preprocess(data)

            self.logger.info("Ejecutando núcleo del pipeline")
            result = self.run_core(processed_data)

            self.logger.info("Evaluando resultados")
            metrics = self.evaluate(result)

            self.logger.info("Guardando artefactos")
            artifacts = self.save_artifacts(result)

            elapsed = time.time() - start
            self.logger.info(f"Pipeline finalizado en {elapsed:.2f}s")

            return PipelineResult(
                success=True,
                message="Ejecutado correctamente",
                metrics=metrics,
                artifacts=artifacts,
                elapsed_seconds=elapsed
            )

        except Exception as e:
            elapsed = time.time() - start
            self.logger.exception("Error ejecutando pipeline")
            return PipelineResult(
                success=False,
                message=str(e),
                elapsed_seconds=elapsed)