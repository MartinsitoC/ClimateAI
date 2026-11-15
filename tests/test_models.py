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

from fastapi.testclient import TestClient
from main import app
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


class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        # Warm-up del endpoint /api/predict para estabilizar el pipeline HTTP en memoria
        cls.client.post("/api/predict", json={"city": "Mexico", "target_year": 2030, "month": 7})

    def test_01_get_config_100_ciudades(self) -> None:
        resp = self.client.get("/api/config")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("ciudades", data)
        self.assertEqual(len(data["ciudades"]), 100, "El catálogo debe contener exactamente 100 ciudades")

    def test_02_post_predict_latencia_y_contrato(self) -> None:
        payload = {"city": "Baghdad", "target_year": 2040, "month": 7, "co2": 220.0}
        t0 = time.perf_counter()
        resp = self.client.post("/api/predict", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(resp.status_code, 200)
        self.assertLess(elapsed_ms, 15.0, f"POST /api/predict tardó {elapsed_ms:.2f} ms (SLA < 15 ms)")

        body = resp.json()
        campos_esperados = [
            "city",
            "country",
            "lat",
            "lon",
            "target_year",
            "prediccion_temperatura",
            "sigma_temp",
            "rango_min",
            "rango_max",
            "anomalia_proyectada",
            "cdd_anual",
            "porcentaje_aumento_cdd",
            "semaforo_hex",
            "alerta_operativa",
            "nivel_estres_red",
            "probabilidades_riesgo",
            "plan_accion",
            "confianza_r2",
        ]
        for campo in campos_esperados:
            self.assertIn(campo, body, f"Falta el campo '{campo}' en la respuesta de /api/predict")
            self.assertIsNotNone(body[campo], f"El campo '{campo}' es nulo en /api/predict")

    def test_03_get_geo_search_cancun(self) -> None:
        resp = self.client.get("/api/geo/search", params={"q": "Cancun"})
        self.assertEqual(resp.status_code, 200)
        resultados = resp.json()
        self.assertIsInstance(resultados, list)
        self.assertGreater(len(resultados), 0, "La búsqueda de 'Cancun' debe retornar al menos un resultado")
        top = resultados[0]
        self.assertIn("lat", top)
        self.assertIn("lon", top)
        lat = float(top["lat"])
        lon = float(top["lon"])
        self.assertTrue(-90.0 <= lat <= 90.0, f"Latitud fuera de rango: {lat}")
        self.assertTrue(-180.0 <= lon <= 180.0, f"Longitud fuera de rango: {lon}")
        self.assertAlmostEqual(lat, 21.16, delta=1.0)
        self.assertAlmostEqual(lon, -86.85, delta=1.0)

    def test_04_post_export_report_image_png(self) -> None:
        payload = {
            "city": "Cancun",
            "country": "Mexico",
            "lat": 21.16,
            "lon": -86.85,
            "target_year": 2045,
            "mode": "live",
            "predicted_temp": 28.04,
            "sigma_temp": 0.49,
            "rango_min": 27.07,
            "rango_max": 29.01,
            "anomaly": 1.04,
            "cdd": 10.04,
            "pct_cdd": 11.5,
            "risk_level": "RIESGO MODERADO",
            "grid_stress": "Alerta (70-85%)",
            "probabilities": {"BAJO": 0.0, "MODERADO": 100.0, "CRÍTICO": 0.0},
            "action_plan": ["Monitoreo preventivo de subestaciones y transformadores."],
        }
        resp = self.client.post("/api/export/report-image", json=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("content-type"), "image/png")
        self.assertTrue(resp.content.startswith(b"\x89PNG\r\n\x1a\n"), "El binario devuelto no tiene firma PNG válida")
        self.assertGreater(len(resp.content), 10240)


if __name__ == "__main__":
    unittest.main(verbosity=2)
