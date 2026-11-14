import json
import math
import sys
import time
import unittest
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.config import CONFIG
from pipeline.model import PredictionInput, UrbanHeatInferenceEngine


class TestUrbanHeatModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = Path(CONFIG["paths"]["output_dir"])
        cls.rf_path = cls.output_dir / "rf_temperature_model.joblib"
        cls.gb_path = cls.output_dir / "gb_grid_risk_model.joblib"
        cls.eval_path = Path(CONFIG["paths"]["metrics_report"])
        cls.engine = UrbanHeatInferenceEngine.get_instance()
        # Warm-up inicial del motor singleton
        cls.engine.predecir(PredictionInput(city="Mexico", target_year=2026))

    def test_01_integridad_artefactos(self) -> None:
        self.assertTrue(self.rf_path.is_file(), f"Falta artefacto: {self.rf_path}")
        self.assertTrue(self.gb_path.is_file(), f"Falta artefacto: {self.gb_path}")
        self.assertTrue(self.eval_path.is_file(), f"Falta matriz de evaluación: {self.eval_path}")
        self.assertGreater(self.rf_path.stat().st_size, 1024)
        self.assertGreater(self.gb_path.stat().st_size, 1024)

    def test_02_calidad_metricas_test(self) -> None:
        matrix = json.loads(self.eval_path.read_text(encoding="utf-8"))
        reg_metrics = matrix["task_a_regression"]["main_random_forest"]["test_metrics"]
        clf_metrics = matrix["task_b_classification"]["main_gradient_boosting"]["test_metrics"]

        r2_val = float(reg_metrics["r2"])
        rmse_val = float(reg_metrics["rmse"])
        f1_macro_val = float(clf_metrics["f1_macro"])

        self.assertGreater(r2_val, 0.85, f"R² en test ({r2_val}) debe ser > 0.85")
        self.assertLess(rmse_val, 1.5, f"RMSE en test ({rmse_val} °C) debe ser < 1.5 °C")
        self.assertGreater(f1_macro_val, 0.90, f"F1-Score Macro ({f1_macro_val}) debe ser > 0.90")

    def test_03_inferencia_casos_limite_sin_nulos_ni_infinitos(self) -> None:
        casos_limite = [
            {"city": "Polo Norte", "lat": 89.9, "lon": 0.0, "target_year": 1850, "month": 1, "co2": 0.0},
            {"city": "Polo Sur", "lat": -89.9, "lon": 180.0, "target_year": 2050, "month": 7, "co2": 500.0},
            {"city": "Ecuador", "lat": 0.0, "lon": -78.5, "target_year": 1850, "month": 6, "co2": 10.0},
            {"city": "Ecuador Futuro", "lat": 0.0, "lon": 32.5, "target_year": 2050, "month": 12, "co2": 800.0},
            {"city": "Baghdad", "target_year": 2050, "month": 7, "co2": 300.0},
        ]

        for payload in casos_limite:
            res = self.engine.predecir(PredictionInput(**payload)).model_dump()
            for key, val in res.items():
                self.assertIsNotNone(val, f"Campo '{key}' es None para entrada {payload}")
                if isinstance(val, (int, float)):
                    self.assertFalse(math.isnan(val), f"Campo '{key}' es NaN para entrada {payload}")
                    self.assertFalse(math.isinf(val), f"Campo '{key}' es Inf para entrada {payload}")
                elif isinstance(val, dict):
                    for sub_k, sub_v in val.items():
                        self.assertIsNotNone(sub_v)
                        self.assertFalse(math.isnan(float(sub_v)))
                        self.assertFalse(math.isinf(float(sub_v)))

            prob_sum = sum(res["probabilidades_riesgo"].values())
            self.assertAlmostEqual(prob_sum, 100.0, delta=0.5)
            self.assertLessEqual(res["rango_min"], res["prediccion_temperatura"])
            self.assertGreaterEqual(res["rango_max"], res["prediccion_temperatura"])

    def test_04_latencia_estricta_menor_10ms(self) -> None:
        entrada = PredictionInput(city="Baghdad", target_year=2040, month=7, co2=200.0)
        t0 = time.perf_counter()
        _ = self.engine.predecir(entrada)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(
            elapsed_ms,
            10.0,
            f"La inferencia individual tardó {elapsed_ms:.3f} ms (límite estricto: 10.0 ms)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
