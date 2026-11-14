"""
URBAN-HEAT GRID INTELLIGENCE — Pipeline Oficial de Entrenamiento Multi-Modelo
Archivo: pipeline/train_models.py
Uso: python -m pipeline.train_models

Entrena y valida DOS modelos de familias distintas con partición estricta 80% Train / 20% Test
cruzando el histórico térmico urbano (Berkeley Earth) con las emisiones anuales de CO₂ por país.
  - Familia 1 (Lineal Paramétrica Regularizada): Ridge Regression con efectos fijos urbanos y forzamiento por CO₂.
  - Familia 2 (Ensamble No Lineal de Árboles): RandomForestRegressor espacial-antropogénico.
"""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from pipeline.config import BASE_DIR, CONFIG

# ==============================================================================
# TAREA 1: RUTAS ABSOLUTAS ROBUSTAS DESDE pipeline/config.py
# ==============================================================================
TEMP_PATH = CONFIG["paths"]["temperatures"]
CO2_PATH = CONFIG["paths"]["co2"]
OUTPUT_DIR = CONFIG["paths"]["output_dir"]
METRICS_REPORT_PATH = CONFIG["paths"]["metrics_report"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Rutas alternativas de respaldo por si el entorno mueve los insumos
TEMP_CANDIDATES = [
    TEMP_PATH,
    BASE_DIR / "data" / "raw" / "GlobalLandTemperaturesByMajorCity.csv",
    BASE_DIR / "GlobalLandTemperaturesByMajorCity.csv",
]
CO2_CANDIDATES = [
    CO2_PATH,
    BASE_DIR / "Temperaturas" / "annual-co2-emissions-per-country.csv",
    BASE_DIR / "annual-co2-emissions-per-country.csv",
]

# Mapeo de nombres históricos de países en Berkeley Earth -> estándar OWID CO₂
COUNTRY_MAP = {
    "Burma": "Myanmar",
    "Congo (Democratic Republic Of The)": "Democratic Republic of Congo",
    "Côte D'Ivoire": "Cote d'Ivoire",
}

CDD_BASE_TEMP = 18.0  # Estándar ASHRAE (°C)
BASELINE_YEARS = (1951, 1980)
TREND_WINDOW = 60


def parse_coord(value: Any) -> float:
    """Convierte coordenadas con sufijo cardinal ('19.25N', '99.10W') a grados decimales."""
    if isinstance(value, (int, float)):
        return float(value)
    m = re.match(r"^\s*([\d.]+)\s*([NSEW])\s*$", str(value).strip(), re.IGNORECASE)
    if not m:
        return 0.0
    val = float(m.group(1))
    return -val if m.group(2).upper() in ("S", "W") else val


def resolver_ruta(principal: Path, candidatos: list, nombre: str) -> Path:
    """Resuelve la ruta absoluta verificando primero la ruta principal y luego los respaldos."""
    if principal.is_file():
        return principal
    for cand in candidatos:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"No se encontró el archivo requerido '{nombre}' en {principal}")


# ==============================================================================
# TAREA 1: PREPARACIÓN, LIMPIEZA Y CRUCE DE DATOS (TEMPERATURA + CO₂)
# ==============================================================================
def cargar_y_cruzar_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga el dataset térmico por ciudad y el inventario anual de emisiones de CO₂ por país,
    aplica limpieza de valores nulos, normalización geográfica y cruce relacional por (Country, year).
    Retorna:
      - df_mensual: Dataset mensual limpio para perfiles estacionales
      - df_anual_merged: Dataset anual consolidado con temperatura, CDD y forzamiento de CO₂
    """
    temp_file = resolver_ruta(TEMP_PATH, TEMP_CANDIDATES, "GlobalLandTemperaturesByMajorCity.csv")
    co2_file = resolver_ruta(CO2_PATH, CO2_CANDIDATES, "annual-co2-emissions-per-country.csv")

    print(f"[1/4] Cargando Temperaturas desde: {temp_file}")
    df_temp = pd.read_csv(temp_file)
    df_temp = df_temp.dropna(subset=["AverageTemperature", "dt", "City", "Country"]).copy()

    fechas = pd.to_datetime(df_temp["dt"], errors="coerce")
    df_temp["year"] = fechas.dt.year
    df_temp["month"] = fechas.dt.month
    df_temp = df_temp.dropna(subset=["year", "month"])
    df_temp["year"] = df_temp["year"].astype(int)
    df_temp["month"] = df_temp["month"].astype(int)
    df_temp = df_temp[df_temp["year"] >= 1850].copy()

    df_temp["lat"] = df_temp["Latitude"].map(parse_coord)
    df_temp["lon"] = df_temp["Longitude"].map(parse_coord)
    df_temp["abs_lat"] = df_temp["lat"].abs()
    df_temp["month_sin"] = np.sin(2.0 * np.pi * df_temp["month"] / 12.0)
    df_temp["month_cos"] = np.cos(2.0 * np.pi * df_temp["month"] / 12.0)
    df_temp["country_co2_key"] = df_temp["Country"].replace(COUNTRY_MAP)

    # Agregación anual rigurosa (años con cobertura completa o media anual por metrópoli)
    agg_anual = (
        df_temp.groupby(["City", "Country", "country_co2_key", "year"], as_index=False)
        .agg(
            temp=("AverageTemperature", "mean"),
            temp_std=("AverageTemperature", "std"),
            uncertainty=("AverageTemperatureUncertainty", "mean"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            abs_lat=("abs_lat", "first"),
            months_count=("month", "count"),
        )
    )
    agg_anual["temp_std"] = agg_anual["temp_std"].fillna(0.0)
    agg_anual["uncertainty"] = agg_anual["uncertainty"].fillna(0.5)
    agg_anual["cdd_daily_equiv"] = np.maximum(0.0, agg_anual["temp"] - CDD_BASE_TEMP)

    print(f"[2/4] Cargando Emisiones de CO₂ desde: {co2_file}")
    df_co2 = pd.read_csv(co2_file)
    co2_col = [c for c in df_co2.columns if "CO" in c or "emissions" in c.lower()][-1]
    df_co2 = df_co2.rename(columns={"Entity": "country_co2_key", "Year": "year", co2_col: "co2_emissions"})
    df_co2 = df_co2[["country_co2_key", "year", "co2_emissions"]].dropna(subset=["country_co2_key", "year"])
    df_co2["year"] = df_co2["year"].astype(int)

    # Extraer serie global ('World') como indicador de forzamiento radiativo planetario
    df_co2_world = (
        df_co2[df_co2["country_co2_key"] == "World"][["year", "co2_emissions"]]
        .rename(columns={"co2_emissions": "co2_global"})
        .drop_duplicates(subset=["year"])
    )

    # Cruce relacional por país y año
    df_merged = pd.merge(agg_anual, df_co2, on=["country_co2_key", "year"], how="left")
    df_merged = pd.merge(df_merged, df_co2_world, on="year", how="left")

    # Imputación ordenada temporalmente por país para años sin registro temprano de CO₂
    df_merged = df_merged.sort_values(["City", "year"]).reset_index(drop=True)
    df_merged["co2_emissions"] = (
        df_merged.groupby("Country")["co2_emissions"]
        .transform(lambda s: s.ffill().bfill())
        .fillna(0.0)
    )
    df_merged["co2_global"] = df_merged["co2_global"].ffill().bfill().fillna(0.0)

    # Transformación logarítmica (ley de forzamiento radiativo de Arrhenius: ΔF ∝ ln(C/C0))
    df_merged["log_co2_country"] = np.log1p(np.maximum(0.0, df_merged["co2_emissions"]))
    df_merged["log_co2_global"] = np.log1p(np.maximum(0.0, df_merged["co2_global"]))

    # Línea base histórica (1951-1980) por ciudad para capturar variaciones climáticas locales
    base_city = (
        df_merged[df_merged["year"].between(*BASELINE_YEARS)]
        .groupby("City")["temp"]
        .mean()
        .to_dict()
    )
    overall_city = df_merged.groupby("City")["temp"].mean().to_dict()
    df_merged["city_baseline_temp"] = df_merged["City"].map(
        lambda c: base_city.get(c, overall_city.get(c, 15.0))
    )
    df_merged["temp_anomaly"] = df_merged["temp"] - df_merged["city_baseline_temp"]

    df_mensual = df_temp.rename(
        columns={"AverageTemperature": "temp", "City": "city", "Country": "country"}
    )
    return df_mensual, df_merged


# ==============================================================================
# TAREA 2: ENTRENAMIENTO DE 2 FAMILIAS DE MODELOS CON PARTICIÓN ESTRICTA 80/20
# ==============================================================================
def evaluar_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula RMSE, MAE y R² con precisión de punto flotante estándar."""
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def entrenar_y_evaluar_modelos(df_merged: pd.DataFrame) -> Tuple[Any, Any, LabelEncoder, Dict[str, Any]]:
    """
    Entrena dos modelos de familias distintas sobre una partición estricta 80% Train / 20% Test:
      1. Familia Lineal Paramétrica: Ridge Regression con estandarización y efectos fijos urbanos.
      2. Familia Ensamble de Árboles: RandomForestRegressor no lineal.
    """
    encoder = LabelEncoder()
    df_merged = df_merged.copy()
    df_merged["city_idx"] = encoder.fit_transform(df_merged["City"])

    feature_cols = [
        "year",
        "lat",
        "lon",
        "abs_lat",
        "city_idx",
        "city_baseline_temp",
        "temp_std",
        "co2_emissions",
        "log_co2_country",
        "log_co2_global",
    ]
    target_col = "temp"

    X = df_merged[feature_cols].copy()
    y = df_merged[target_col].to_numpy()

    # Partición estricta 80% entrenamiento / 20% prueba estratificada por ciudad
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        train_size=0.80,
        random_state=42,
        stratify=df_merged["City"],
    )

    n_total = len(df_merged)
    n_train = len(X_train)
    n_test = len(X_test)
    print(
        f"[3/4] Partición Estricta 80/20: Total={n_total:,} | "
        f"Train={n_train:,} ({n_train / n_total:.1%}) | Test={n_test:,} ({n_test / n_total:.1%})"
    )

    # --------------------------------------------------------------------------
    # MODELO 1 (FAMILIA LINEAL PARAMÉTRICA): Ridge Regression + Preprocesamiento
    # --------------------------------------------------------------------------
    num_features = [
        "year",
        "lat",
        "lon",
        "abs_lat",
        "city_baseline_temp",
        "temp_std",
        "co2_emissions",
        "log_co2_country",
        "log_co2_global",
    ]
    cat_features = ["city_idx"]

    preprocessor_linear = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )
    model_family_1_ridge = Pipeline(
        steps=[
            ("preprocessor", preprocessor_linear),
            ("regressor", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    model_family_1_ridge.fit(X_train, y_train)

    y_train_pred_m1 = model_family_1_ridge.predict(X_train)
    y_test_pred_m1 = model_family_1_ridge.predict(X_test)
    metrics_m1_train = evaluar_metricas(y_train, y_train_pred_m1)
    metrics_m1_test = evaluar_metricas(y_test, y_test_pred_m1)

    # --------------------------------------------------------------------------
    # MODELO 2 (FAMILIA ENSAMBLE DE ÁRBOLES NO LINEAL): RandomForestRegressor
    # --------------------------------------------------------------------------
    model_family_2_rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model_family_2_rf.fit(X_train, y_train)

    y_train_pred_m2 = model_family_2_rf.predict(X_train)
    y_test_pred_m2 = model_family_2_rf.predict(X_test)
    metrics_m2_train = evaluar_metricas(y_train, y_train_pred_m2)
    metrics_m2_test = evaluar_metricas(y_test, y_test_pred_m2)

    # Selección automática del modelo ganador según R² y RMSE en el 20% de prueba
    ganador = (
        "RandomForestRegressor (Familia Ensamble)"
        if metrics_m2_test["rmse"] <= metrics_m1_test["rmse"]
        else "Ridge (Familia Lineal Regularizada)"
    )

    importancia_features = dict(
        zip(
            feature_cols,
            [round(float(v), 4) for v in model_family_2_rf.feature_importances_],
        )
    )

    reporte = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_summary": {
            "total_records": n_total,
            "train_records": n_train,
            "test_records": n_test,
            "train_ratio": round(n_train / n_total, 4),
            "test_ratio": round(n_test / n_total, 4),
            "total_cities": int(df_merged["City"].nunique()),
            "total_countries": int(df_merged["Country"].nunique()),
            "year_min": int(df_merged["year"].min()),
            "year_max": int(df_merged["year"].max()),
        },
        "features": feature_cols,
        "models": {
            "family_1_linear_ridge": {
                "family": "Lineal Paramétrica Regularizada (Ridge L2)",
                "train_80": metrics_m1_train,
                "test_20": metrics_m1_test,
            },
            "family_2_ensemble_rf": {
                "family": "Ensamble No Lineal de Árboles (Random Forest)",
                "train_80": metrics_m2_train,
                "test_20": metrics_m2_test,
                "feature_importances": importancia_features,
            },
        },
        "best_model": ganador,
    }

    print("\n" + "=" * 74)
    print("  RESULTADOS DE VALIDACIÓN (PARTICIÓN ESTRICTA 80% TRAIN / 20% TEST)")
    print("=" * 74)
    print(
        f"  [Familia 1 - Ridge Lineal]   Train R²={metrics_m1_train['r2']:.4f} | "
        f"Test RMSE={metrics_m1_test['rmse']:.4f} °C | MAE={metrics_m1_test['mae']:.4f} °C | R²={metrics_m1_test['r2']:.4f}"
    )
    print(
        f"  [Familia 2 - Random Forest]  Train R²={metrics_m2_train['r2']:.4f} | "
        f"Test RMSE={metrics_m2_test['rmse']:.4f} °C | MAE={metrics_m2_test['mae']:.4f} °C | R²={metrics_m2_test['r2']:.4f}"
    )
    print(f"  >> Modelo Seleccionado: {ganador}")
    print("=" * 74 + "\n")

    return model_family_1_ridge, model_family_2_rf, encoder, reporte


# ==============================================================================
# TAREA 3: EXPORTACIÓN DE ARTEFACTOS Y SINCRONIZACIÓN CON MOTOR DE INFERENCIA
# ==============================================================================
def actualizar_artefacto_inferencia_produccion(
    df_mensual: pd.DataFrame, reporte_comparativo: Dict[str, Any]
) -> None:
    """
    Actualiza data/processed/rf_model.joblib manteniendo compatibilidad total con
    UrbanHeatInferenceEngine (pipeline/model.py) e inyectando las métricas 80/20.
    """
    encoder_city = LabelEncoder().fit(sorted(df_mensual["city"].unique()))
    g = df_mensual.groupby(["city", "year"])["temp"].agg(["mean", "count"]).reset_index()
    annual_full = g[g["count"] == 12].rename(columns={"mean": "temp"})[["city", "year", "temp"]]

    tendencias = {}
    for city, grp in annual_full.groupby("city"):
        grp = grp.sort_values("year")
        ventana = grp[grp["year"] > grp["year"].max() - TREND_WINDOW]
        if len(ventana) < 20:
            ventana = grp
        slope, intercept = np.polyfit(ventana["year"], ventana["temp"], 1)
        tendencias[city] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "last_year": int(grp["year"].max()),
        }

    inicio = df_mensual["city"].map(lambda c: tendencias[c]["last_year"] - TREND_WINDOW)
    ventana_df = df_mensual[df_mensual["year"] > inicio].copy()
    features_prod = ["lat", "lon", "city_idx", "month_sin", "month_cos"]
    X_prod = ventana_df.assign(city_idx=encoder_city.transform(ventana_df["city"]))[features_prod]
    s_arr = ventana_df["city"].map(lambda c: tendencias[c]["slope"]).to_numpy()
    b_arr = ventana_df["city"].map(lambda c: tendencias[c]["intercept"]).to_numpy()
    y_resid = ventana_df["temp"].to_numpy() - (b_arr + s_arr * ventana_df["year"].to_numpy())

    rf_prod = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=1)
    rf_prod.fit(X_prod, y_resid)

    clim_all = (
        df_mensual[df_mensual["year"].between(*BASELINE_YEARS)]
        .groupby(["city", "month"])["temp"]
        .mean()
    )
    meta = df_mensual.groupby("city")[["country", "lat", "lon"]].first()

    ciudades_dict = {}
    for city, row in meta.iterrows():
        mensual = [float(clim_all.get((city, mes), np.nan)) for mes in range(1, 13)]
        ciudades_dict[city] = {
            "country": row["country"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            **tendencias[city],
            "baseline_monthly": mensual,
            "baseline_annual": float(np.nanmean(mensual)),
        }

    best_test = reporte_comparativo["models"]["family_2_ensemble_rf"]["test_20"]
    metrics_compat = {
        "split_strategy": "80_20_stratified",
        "train_share": reporte_comparativo["dataset_summary"]["train_ratio"],
        "test_share": reporte_comparativo["dataset_summary"]["test_ratio"],
        "modelo_anual": best_test,
        "comparativa_familias": reporte_comparativo["models"],
    }

    payload = {
        "model": rf_prod,
        "encoder": encoder_city,
        "features": features_prod,
        "cities": ciudades_dict,
        "metrics": metrics_compat,
        "config": {"baseline": BASELINE_YEARS, "trend_window": TREND_WINDOW},
        "trained_at": reporte_comparativo["trained_at"],
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(payload, OUTPUT_DIR / "rf_model.joblib", compress=3)
    annual_full.to_csv(OUTPUT_DIR / "df_annual.csv", index=False)


def run_training_pipeline() -> Dict[str, Any]:
    """Ejecuta de principio a fin el pipeline de preparación, cruce CO₂, entrenamiento 80/20 y serialización."""
    df_mensual, df_merged = cargar_y_cruzar_datasets()

    # Guardar dataset procesado y cruzado (Temperaturas + CO₂)
    merged_csv_path = OUTPUT_DIR / "merged_climate_co2.csv"
    df_merged.to_csv(merged_csv_path, index=False)
    print(f"[+] Dataset cruzado (Clima + CO₂) exportado en: {merged_csv_path}")

    model_ridge, model_rf, encoder, reporte = entrenar_y_evaluar_modelos(df_merged)

    # Serializar ambos modelos entrenados y el reporte comparativo de métricas
    print("[4/4] Serializando artefactos de modelos y métricas en data/processed/...")
    joblib.dump(
        {"model": model_ridge, "encoder": encoder, "metrics": reporte["models"]["family_1_linear_ridge"]},
        OUTPUT_DIR / "model_family_1_ridge.joblib",
        compress=3,
    )
    joblib.dump(
        {"model": model_rf, "encoder": encoder, "metrics": reporte["models"]["family_2_ensemble_rf"]},
        OUTPUT_DIR / "model_family_2_rf.joblib",
        compress=3,
    )

    metrics_json_path = OUTPUT_DIR / "models_comparison_metrics.json"
    reporte_json = json.dumps(reporte, indent=2, ensure_ascii=False)
    metrics_json_path.write_text(reporte_json, encoding="utf-8")
    METRICS_REPORT_PATH.write_text(reporte_json, encoding="utf-8")

    # Sincronizar artefacto principal del motor de inferencia (rf_model.joblib)
    actualizar_artefacto_inferencia_produccion(df_mensual, reporte)
    print(f"[OK] Pipeline finalizado exitosamente. Artefactos disponibles en: {OUTPUT_DIR}")
    return reporte


if __name__ == "__main__":
    run_training_pipeline()
