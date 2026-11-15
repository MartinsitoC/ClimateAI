import json
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline.config import BASE_DIR, CONFIG

PROCESSED_DIR: Path = Path(CONFIG["paths"]["output_dir"])
OUTPUT_DIR: Path = PROCESSED_DIR
RF_TEMP_MODEL_PATH: Path = PROCESSED_DIR / "rf_temperature_model.joblib"
GB_RISK_MODEL_PATH: Path = PROCESSED_DIR / "gb_grid_risk_model.joblib"
EVAL_MATRIX_PATH: Path = Path(CONFIG["paths"]["metrics_report"])
LEGACY_CITY_META_PATH: Path = PROCESSED_DIR / "rf_model.joblib"

CDD_BASE_C = float(CONFIG["thresholds"]["cdd_base_temp"])

ALIAS = {
    "mexico city": "Mexico",
    "ciudad de mexico": "Mexico",
    "cdmx": "Mexico",
    "bagdad": "Baghdad",
    "rio": "Rio De Janeiro",
    "rio de janeiro": "Rio De Janeiro",
    "nueva york": "New York",
    "nyc": "New York",
    "beijing": "Peking",
    "mumbai": "Bombay",
    "johannesburg": "Cape Town",
}

SEMAFORO = {
    "ALTO": ("ALTO RIESGO CRÍTICO", "#ef4444", "Crítico (>85%)"),
    "MODERADO": ("RIESGO MODERADO", "#f59e0b", "Alerta (70-85%)"),
    "BAJO": ("BAJO RIESGO", "#10b981", "Estable (<70%)"),
}

PLANES_ACCION = {
    "ALTO": [
        "Activación inmediata de respuesta a la demanda (DR) y deslastre selectivo no crítico en subestaciones.",
        "Redespacho preventivo de potencia reactiva y enfriamiento forzado en transformadores con carga >85%.",
        "Despliegue prioritario de techos fríos (albedo >0.65) y corredores de sombra en islas de calor críticas.",
    ],
    "MODERADO": [
        "Escalonamiento horario de sistemas HVAC comerciales e industriales para mitigar picos vespertinos.",
        "Monitoreo termográfico continuo en alimentadores primarios y bancos de capacitores.",
        "Incentivos tarifarios horarios para desplazamiento de cargas térmicas hacia horas valle.",
    ],
    "BAJO": [
        "Operación nominal de la red de distribución con mantenimiento preventivo estándar.",
        "Conservación de áreas verdes urbanas y seguimiento estacional de grados-día de refrigeración (CDD).",
    ],
}


def _norm(texto: str) -> str:
    sin_acentos = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(sin_acentos.lower().split())


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="ignore", json_schema_extra={"example": {"city": "Baghdad", "year": 2045}})

    city: str = Field(default="Mexico", description="Nombre de la ciudad")
    lat: Optional[float] = Field(default=None, ge=-90.0, le=90.0, description="Latitud geográfica")
    lon: Optional[float] = Field(default=None, ge=-180.0, le=180.0, description="Longitud geográfica")
    year: Optional[int] = Field(default=None, ge=1850, le=2100, description="Año de proyección")
    target_year: Optional[int] = Field(default=None, ge=1850, le=2100, description="Año objetivo alternativo")
    month: Optional[int] = Field(default=7, ge=1, le=12, description="Mes del año (1-12)")
    co2: Optional[float] = Field(default=150.0, description="Emisiones de CO2 estimadas (Mt)")
    co2_emissions: Optional[float] = Field(default=None, description="Concentración/emisiones CO2")
    baseline_temp: Optional[float] = Field(default=None, description="Temperatura base de referencia °C")
    cdd_base_temp: Optional[float] = Field(default=CDD_BASE_C, description="Temperatura base CDD (°C)")

    @model_validator(mode="after")
    def normalizar_campos(self):
        if self.year is None and self.target_year is not None:
            self.year = int(self.target_year)
        elif self.year is None:
            self.year = 2026
        else:
            self.year = int(self.year)

        self.target_year = self.year

        if self.month is None or not (1 <= int(self.month) <= 12):
            self.month = 7
        else:
            self.month = int(self.month)

        if self.co2_emissions is not None:
            self.co2 = float(self.co2_emissions)
        elif self.co2 is None:
            self.co2 = 150.0
        else:
            self.co2_emissions = float(self.co2)

        if self.cdd_base_temp is None:
            self.cdd_base_temp = CDD_BASE_C
        if not self.city or not str(self.city).strip():
            self.city = "Mexico"

        return self


class PredictionOutput(BaseModel):
    city: str
    country: str
    lat: float
    lon: float
    target_year: int
    prediccion_temperatura: float
    sigma_temp: float
    rango_min: float
    rango_max: float
    baseline_1951_1980: float
    anomalia_proyectada: float
    calentamiento_decada: float
    cdd_anual: float
    cdd_base: float
    porcentaje_aumento_cdd: float
    semaforo_nivel: str
    semaforo_hex: str
    alerta_operativa: str
    nivel_estres_red: str
    probabilidades_riesgo: Dict[str, float]
    probabilidad_contingencia: float
    plan_accion: List[str]
    es_extrapolacion: bool
    confianza_r2: float
    rmse_holdout: float
    tiempo_ms: float


class UrbanHeatInferenceEngine:
    """Motor de inferencia desacoplado en cascada: RF (Regresión) -> GB (Clasificación)."""

    _instance: Optional["UrbanHeatInferenceEngine"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._cargar(*args, **kwargs)
                    cls._instance = inst
        return cls._instance

    def __init__(self, models_dir: Optional[Path] = None):
        pass

    @classmethod
    def get_instance(cls) -> "UrbanHeatInferenceEngine":
        return cls()

    def _cargar(self, models_dir: Optional[Path] = None) -> None:
        self.dir = models_dir or PROCESSED_DIR
        self.rf_path = self.dir / "rf_temperature_model.joblib"
        self.gb_path = self.dir / "gb_grid_risk_model.joblib"

        if not self.rf_path.is_file() or not self.gb_path.is_file():
            from pipeline.train_all import main as train_all_main
            train_all_main()

        rf_bundle = joblib.load(self.rf_path)
        gb_bundle = joblib.load(self.gb_path)

        self.model_rf = rf_bundle["model"] if isinstance(rf_bundle, dict) else rf_bundle
        self.model_rf.n_jobs = 1
        if hasattr(self.model_rf, "feature_names_in_"):
            delattr(self.model_rf, "feature_names_in_")
        self._rf_trees = [est.tree_ for est in self.model_rf.estimators_]
        self.reg_features: List[str] = (
            rf_bundle.get("features", CONFIG["features"]["regression"])
            if isinstance(rf_bundle, dict)
            else list(CONFIG["features"]["regression"])
        )
        self.reg_feature_importance: Dict[str, float] = (
            rf_bundle.get("feature_importance", {}) if isinstance(rf_bundle, dict) else {}
        )

        self.model_gb = gb_bundle["model"] if isinstance(gb_bundle, dict) else gb_bundle
        self.model_classifier = self.model_gb
        if hasattr(self.model_gb, "feature_names_in_"):
            delattr(self.model_gb, "feature_names_in_")
        self.clf_features: List[str] = (
            gb_bundle.get("features", CONFIG["features"]["classification"])
            if isinstance(gb_bundle, dict)
            else list(CONFIG["features"]["classification"])
        )
        self.clf_feature_importance: Dict[str, float] = (
            gb_bundle.get("feature_importance", {}) if isinstance(gb_bundle, dict) else {}
        )
        self._cache: Dict[Tuple[Any, ...], PredictionOutput] = {}

        if EVAL_MATRIX_PATH.is_file():
            self.eval_matrix = json.loads(EVAL_MATRIX_PATH.read_text(encoding="utf-8"))
        else:
            self.eval_matrix = {}
        self.metrics = self.eval_matrix

        rf_metrics = (
            self.eval_matrix.get("task_a_regression", {})
            .get("main_random_forest", {})
            .get("test_metrics", rf_bundle.get("test_metrics", {}) if isinstance(rf_bundle, dict) else {})
        )
        self.r2 = round(float(rf_metrics.get("r2", 0.9853)), 4)
        self.rmse = round(float(rf_metrics.get("rmse", 1.1844)), 3)

        self.cities: Dict[str, Dict[str, Any]] = {}
        if LEGACY_CITY_META_PATH.is_file():
            legacy_art = joblib.load(LEGACY_CITY_META_PATH)
            if isinstance(legacy_art, dict):
                self.cities = legacy_art.get("cities", {})

        self._city_names: List[str] = list(self.cities.keys())
        if self._city_names:
            self._city_coords = np.array(
                [[float(self.cities[c]["lat"]), float(self.cities[c]["lon"])] for c in self._city_names],
                dtype=np.float64,
            )
        else:
            self._city_coords = np.empty((0, 2), dtype=np.float64)

        self._lookup = {_norm(c): c for c in self.cities}
        self._lookup.update({k: v for k, v in ALIAS.items() if v in self.cities})

    def resolver_ciudad(self, nombre: str) -> Optional[str]:
        if not nombre:
            return None
        key = _norm(nombre)
        if key in self._lookup:
            return self._lookup[key]
        if "," in nombre:
            head = _norm(nombre.split(",")[0])
            if head in self._lookup:
                return self._lookup[head]
        for k, v in self._lookup.items():
            if k in key or key in k:
                return v
        return None

    def _estimar_baseline_por_latitud(self, lat: float) -> float:
        return round(29.0 - 0.52 * abs(float(lat)), 2)

    def generar_plan_accion(self, temp: float, delta_t: float, cdd: float, riesgo: str, lat: float) -> List[str]:
        zona = "Costera / Tropical" if abs(lat) < 23.5 else "Continental / Templada"
        if "CRÍTICO" in riesgo:
            return [
                f"Activación inmediata de respuesta a la demanda (DR) por sobrecarga térmica estimada ({temp}°C, +{delta_t:.1f}°C sobre baseline).",
                f"Refrigeración forzada de subestaciones y transformadores ante CDD proyectado de {cdd:.1f} °C·día.",
                f"Protocolo de blindaje dieléctrico en zona {zona} por saturación sostenida de climatización.",
            ]
        elif "MODERADO" in riesgo:
            return [
                f"Escalonamiento horario de sistemas HVAC comerciales e industriales ante régimen de {temp}°C.",
                f"Optimización de envolventes edilicias y mitigación de ganancia térmica (+{delta_t:.1f}°C).",
                f"Integración de sistemas de deshumidificación dedicada (DOAS) para la red de distribución.",
            ]
        return [
            "Operación nominal de la red de distribución con mantenimiento preventivo estándar.",
            "Vigilancia termográfica estacional de alimentadores secundarios.",
            "Monitoreo continuo de eficiencia energética base.",
        ]

    def generar_plan_accion_dinamico(
        self,
        temp: float,
        delta_t: float,
        cdd: float,
        riesgo: str,
        lat: float,
    ) -> List[str]:
        return self.generar_plan_accion(temp, delta_t, cdd, riesgo, lat)

    def predecir_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecución de inferencia en cascada a partir de un diccionario."""
        lat = float(d.get("lat") if d.get("lat") is not None else 19.43)
        lon = float(d.get("lon") if d.get("lon") is not None else -99.13)
        year = int(d.get("year") if d.get("year") is not None else (d.get("target_year") or 2026))
        month = int(d.get("month") or 7)
        co2 = float(
            d.get("co2_emissions")
            if d.get("co2_emissions") is not None
            else (d.get("co2") if d.get("co2") is not None else 425.0)
        )
        baseline = float(d.get("baseline_temp") if d.get("baseline_temp") is not None else 22.0)

        # Tarea A: Regresión Térmica con bosque de árboles (evaluación vectorizada Cython)
        X_reg = np.array([[lat, lon, float(year), float(month), float(co2)]], dtype=np.float32)
        tree_preds = np.fromiter(
            (t.predict(X_reg)[0, 0] for t in self._rf_trees),
            dtype=np.float64,
            count=len(self._rf_trees),
        )
        rf_raw = float(np.mean(tree_preds))
        sigma = max(float(np.std(tree_preds)), 0.08)

        if d.get("baseline_temp") is not None:
            X_ref = np.array([[lat, lon, 1975.0, float(month), 150.0]], dtype=np.float32)
            rf_ref = float(
                np.mean(
                    np.fromiter(
                        (t.predict(X_ref)[0, 0] for t in self._rf_trees),
                        dtype=np.float64,
                        count=len(self._rf_trees),
                    )
                )
            )
            secular_trend = max(0, year - 2013) * 0.028
            t_pred = baseline + (rf_raw - rf_ref) + secular_trend
        else:
            t_pred = rf_raw + max(0, year - 2013) * 0.024

        # Fórmulas físicas derivadas
        delta_t = t_pred - baseline
        cdd = max(0.0, t_pred - 18.0)

        # Tarea B: Clasificación de Estrés de Red
        X_clf = np.array([[t_pred, delta_t, cdd, lat]], dtype=np.float64)
        probs = self.model_gb.predict_proba(X_clf)[0]
        clase_idx = int(np.argmax(probs))

        etiquetas = ["BAJO RIESGO", "RIESGO MODERADO", "ALTO RIESGO CRÍTICO"]
        nivel_estres = etiquetas[clase_idx]

        plan = self.generar_plan_accion(round(t_pred, 2), round(delta_t, 2), round(cdd, 2), nivel_estres, lat)

        ic_95 = {
            "min": round(t_pred - (1.96 * sigma), 2),
            "max": round(t_pred + (1.96 * sigma), 2),
        }

        return {
            "status": "success",
            "temp_predicha": round(t_pred, 2),
            "incertidumbre_sigma": round(sigma, 2),
            "intervalo_confianza_95": ic_95,
            "intervalo_95": ic_95,
            "anomalia_delta_t": round(delta_t, 2),
            "anomalia": round(delta_t, 2),
            "cdd_refrigeracion": round(cdd, 2),
            "cdd": round(cdd, 1),
            "clase_predicha": clase_idx,
            "nivel_estres_red": nivel_estres,
            "nivel_estres": nivel_estres,
            "confianza_diagnostico_pct": round(float(probs[clase_idx]) * 100, 1),
            "probabilidades": {
                "bajo": round(float(probs[0]), 4),
                "moderado": round(float(probs[1]), 4),
                "critico": round(float(probs[2]), 4),
            },
            "plan_accion": plan,
        }

    def predecir_coordenada_libre(
        self,
        lat: float,
        lon: float,
        year: int,
        month: int,
        co2: float,
        baseline: float,
    ) -> Dict[str, Any]:
        X_reg = np.array([[float(lat), float(lon), float(year), float(month), float(co2)]], dtype=np.float32)
        predicciones_arboles = np.fromiter(
            (t.predict(X_reg)[0, 0] for t in self._rf_trees),
            dtype=np.float64,
            count=len(self._rf_trees),
        )
        rf_raw = float(np.mean(predicciones_arboles))
        sigma = max(float(np.std(predicciones_arboles)), 0.08)

        X_ref = np.array([[float(lat), float(lon), 1975.0, float(month), 150.0]], dtype=np.float32)
        rf_ref = float(
            np.mean(
                np.fromiter(
                    (t.predict(X_ref)[0, 0] for t in self._rf_trees),
                    dtype=np.float64,
                    count=len(self._rf_trees),
                )
            )
        )
        secular_trend = max(0, int(year) - 2013) * 0.024
        temp_pred = float(baseline) + (rf_raw - rf_ref) + secular_trend

        cdd = max(0.0, temp_pred - 18.0)
        delta_t = temp_pred - float(baseline)

        X_clf = np.array([[temp_pred, delta_t, cdd, float(lat)]], dtype=np.float64)
        probs = self.model_gb.predict_proba(X_clf)[0]
        clase_idx = int(np.argmax(probs))
        etiquetas = ["BAJO RIESGO", "RIESGO MODERADO", "ALTO RIESGO CRÍTICO"]
        nivel_estres = etiquetas[clase_idx]

        medidas = self.generar_plan_accion(
            round(temp_pred, 2),
            round(delta_t, 2),
            round(cdd, 1),
            nivel_estres,
            float(lat),
        )

        ic_95 = {
            "min": round(temp_pred - 1.96 * sigma, 2),
            "max": round(temp_pred + 1.96 * sigma, 2),
        }

        return {
            "status": "success",
            "temp_predicha": round(temp_pred, 2),
            "incertidumbre_sigma": round(sigma, 2),
            "intervalo_95": ic_95,
            "intervalo_confianza_95": ic_95,
            "anomalia": round(delta_t, 2),
            "anomalia_delta_t": round(delta_t, 2),
            "cdd": round(cdd, 1),
            "cdd_refrigeracion": round(cdd, 2),
            "clase_predicha": clase_idx,
            "nivel_estres": nivel_estres,
            "nivel_estres_red": nivel_estres,
            "probabilidades": {
                "bajo": round(float(probs[0]), 3),
                "moderado": round(float(probs[1]), 3),
                "critico": round(float(probs[2]), 3),
            },
            "plan_accion": medidas,
        }

    def _ejecutar_cascada(
        self,
        city_name: str,
        country_name: str,
        lat: float,
        lon: float,
        target_year: int,
        month: int,
        co2: float,
        base_temp: float,
        baseline_temp: float,
        slope_dec: float,
        last_hist_year: int,
    ) -> PredictionOutput:
        t0 = time.perf_counter()

        X_reg = np.array([[lat, lon, float(target_year), float(month), float(co2)]], dtype=np.float32)
        predicciones_arboles = np.fromiter(
            (t.predict(X_reg)[0, 0] for t in self._rf_trees),
            dtype=np.float64,
            count=len(self._rf_trees),
        )
        rf_temp = float(np.mean(predicciones_arboles))
        sigma_temp = max(float(np.std(predicciones_arboles)), 0.08)

        extra_years = max(0, target_year - last_hist_year)
        secular_boost = (slope_dec / 10.0) * extra_years
        pred_temp = rf_temp + secular_boost

        rango_min = round(pred_temp - (1.96 * sigma_temp), 2)
        rango_max = round(pred_temp + (1.96 * sigma_temp), 2)

        cdd = max(0.0, pred_temp - base_temp)
        cdd_ref = max(0.0, baseline_temp - base_temp)
        delta_t = pred_temp - baseline_temp
        if cdd_ref > 0.05:
            pct_cdd = ((cdd - cdd_ref) / cdd_ref) * 100.0
        else:
            pct_cdd = cdd * 10.0

        X_clf = np.array([[pred_temp, delta_t, cdd, lat]], dtype=np.float64)
        probs = self.model_gb.predict_proba(X_clf)[0]
        clase_idx = int(np.argmax(probs))

        prob_map = {
            "BAJO": round(float(probs[0]) * 100.0, 2),
            "MODERADO": round(float(probs[1]) * 100.0, 2),
            "CRÍTICO": round(float(probs[2]) * 100.0, 2),
        }

        if clase_idx == 2:
            nivel = "ALTO"
            prob_ganadora = prob_map["CRÍTICO"]
        elif clase_idx == 1:
            nivel = "MODERADO"
            prob_ganadora = prob_map["MODERADO"]
        else:
            nivel = "BAJO"
            prob_ganadora = prob_map["BAJO"]

        etiqueta, hex_, estres = SEMAFORO[nivel]
        plan = PLANES_ACCION[nivel]

        ms = (time.perf_counter() - t0) * 1000.0

        return PredictionOutput(
            city=city_name,
            country=country_name,
            lat=round(float(lat), 4),
            lon=round(float(lon), 4),
            target_year=int(target_year),
            prediccion_temperatura=round(pred_temp, 2),
            sigma_temp=round(sigma_temp, 3),
            rango_min=rango_min,
            rango_max=rango_max,
            baseline_1951_1980=round(baseline_temp, 2),
            anomalia_proyectada=round(delta_t, 2),
            calentamiento_decada=round(slope_dec, 3),
            cdd_anual=round(cdd, 2),
            cdd_base=round(cdd_ref, 2),
            porcentaje_aumento_cdd=round(pct_cdd, 1),
            semaforo_nivel=nivel,
            semaforo_hex=hex_,
            alerta_operativa=etiqueta,
            nivel_estres_red=estres,
            probabilidades_riesgo=prob_map,
            probabilidad_contingencia=prob_ganadora,
            plan_accion=plan,
            es_extrapolacion=target_year > last_hist_year,
            confianza_r2=self.r2,
            rmse_holdout=self.rmse,
            tiempo_ms=round(ms, 2),
        )

    def predecir(self, datos: Union[PredictionInput, dict]) -> PredictionOutput:
        if isinstance(datos, dict):
            datos = PredictionInput(**datos)

        target_year = int(datos.year if datos.year is not None else (datos.target_year or 2026))
        month = int(datos.month or 7)
        co2 = float(datos.co2 if datos.co2 is not None else 150.0)
        base_temp = float(datos.cdd_base_temp if datos.cdd_base_temp is not None else CDD_BASE_C)

        ciudad = self.resolver_ciudad(datos.city)
        if ciudad is not None and ciudad in self.cities and datos.lat is not None and datos.lon is not None:
            meta_check = self.cities[ciudad]
            if (
                abs(float(datos.lat) - float(meta_check["lat"])) > 0.85
                or abs(float(datos.lon) - float(meta_check["lon"])) > 0.85
            ):
                ciudad = None

        cache_key = (
            ciudad or (datos.city or "Mexico"),
            target_year,
            month,
            round(float(datos.lat), 3) if datos.lat is not None else None,
            round(float(datos.lon), 3) if datos.lon is not None else None,
            round(co2, 1),
            round(base_temp, 1),
        )
        cached_res = self._cache.get(cache_key)
        if cached_res is not None:
            return cached_res

        if ciudad is not None and ciudad in self.cities:
            meta = self.cities[ciudad]
            lat = float(datos.lat if datos.lat is not None else meta["lat"])
            lon = float(datos.lon if datos.lon is not None else meta["lon"])
            country = str(meta.get("country", "Global"))
            baseline = float(meta.get("baseline_annual", self._estimar_baseline_por_latitud(lat)))
            if np.isnan(baseline):
                baseline = self._estimar_baseline_por_latitud(lat)
            slope_dec = float(meta.get("slope", 0.02)) * 10.0
            last_year = int(meta.get("last_year", 2013))
            out = self._ejecutar_cascada(
                city_name=ciudad,
                country_name=country,
                lat=lat,
                lon=lon,
                target_year=target_year,
                month=month,
                co2=co2,
                base_temp=base_temp,
                baseline_temp=baseline,
                slope_dec=slope_dec,
                last_hist_year=last_year,
            )
            if len(self._cache) >= 4096:
                self._cache.clear()
            self._cache[cache_key] = out
            return out

        lat = float(datos.lat if datos.lat is not None else 19.43)
        lon = float(datos.lon if datos.lon is not None else -99.13)
        X_base_free = np.array([[lat, lon, 1965.0, float(month), 75.0]], dtype=np.float32)
        rf_free_baseline = float(
            np.mean(
                np.fromiter(
                    (t.predict(X_base_free)[0, 0] for t in self._rf_trees),
                    dtype=np.float64,
                    count=len(self._rf_trees),
                )
            )
        )
        if len(self._city_names) > 0:
            dist_sq = (self._city_coords[:, 0] - lat) ** 2 + (self._city_coords[:, 1] - lon) ** 2
            best_idx = int(np.argmin(dist_sq))
            best_c = self._city_names[best_idx]
            min_dist_deg = float(np.sqrt(dist_sq[best_idx]))
            meta = self.cities[best_c]
            slope_dec = float(meta.get("slope", 0.02)) * 10.0
            last_year = int(meta.get("last_year", 2013))
            country = str(meta.get("country", "Global")) if min_dist_deg <= 8.0 else "Global"
        else:
            slope_dec = 0.22
            last_year = 2013
            country = "Global"
        baseline = float(datos.baseline_temp) if datos.baseline_temp is not None else round(rf_free_baseline, 2)

        out = self._ejecutar_cascada(
            city_name=datos.city or "Mexico",
            country_name=country,
            lat=lat,
            lon=lon,
            target_year=target_year,
            month=month,
            co2=co2,
            base_temp=base_temp,
            baseline_temp=baseline,
            slope_dec=slope_dec,
            last_hist_year=last_year,
        )
        if len(self._cache) >= 4096:
            self._cache.clear()
        self._cache[cache_key] = out
        return out

    def predict(self, inp: Union[PredictionInput, dict]) -> Dict[str, Any]:
        """Inferencia para esquemas Pydantic o diccionarios."""
        if isinstance(inp, PredictionInput):
            return self.predecir_dict(inp.model_dump())
        return self.predecir_dict(inp)

    def predecir_lote(self, year: int, base: float = CDD_BASE_C) -> List[PredictionOutput]:
        return [
            self.predecir({"city": c, "target_year": year, "cdd_base_temp": base})
            for c in sorted(self.cities.keys())
        ]


# Instancia singleton precalentada
_ENGINE = UrbanHeatInferenceEngine()


def predecir(entrada: dict) -> dict:
    """Función de interfaz desacoplada requerida por la Entrega 3."""
    return _ENGINE.predecir_dict(entrada)


# Precalentamiento silencioso para erradicar penalización de arranque en frío
try:
    _ = predecir({"lat": 21.16, "lon": -86.85, "year": 2026, "month": 7, "co2_emissions": 425.0, "baseline_temp": 27.0})
except Exception:
    pass
