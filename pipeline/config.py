# pipeline/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG = {
    "paths": {
        "temperatures": BASE_DIR / "Temperaturas" / "Global Land Temperatures" / "GlobalLandTemperaturesByMajorCity.csv",
        "co2": BASE_DIR / "data" / "raw" / "annual-co2-emissions-per-country.csv",
        "output_dir": BASE_DIR / "data" / "processed",
        "metrics_report": BASE_DIR / "data" / "processed" / "evaluation_matrix.json"
    },
    "split": {
        "test_size": 0.20,
        "random_state": 42,
        "shuffle": True,
        "stratify_classification": True
    },
    "features": {
        "regression": ["lat", "lon", "year", "month", "co2_emissions"],
        "classification": ["AverageTemperature", "Delta_T", "CDD", "lat"]
    },
    "thresholds": {
        "cdd_base_temp": 18.0,
        "risk_moderate_temp": 26.0,
        "risk_critical_temp": 32.0,
        "risk_critical_cdd": 14.0
    }
}
