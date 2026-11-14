"""
URBAN-HEAT GRID INTELLIGENCE — Entrenamiento del modelo predictivo
Archivo: pipeline/train_model.py   |   Uso: python -m pipeline.train_model

Modelo híbrido:
  T(ciudad, año, mes) = tendencia_lineal_ciudad(año) + RandomForest(lat, lon, ciudad, mes)
  - La tendencia (OLS sobre los últimos TREND_WINDOW años completos) aporta la extrapolación temporal.
  - El RF modela el residuo mensual (ciclo estacional + estructura espacial) que alimenta el CDD mensual.
  Un RF puro con 'year' como feature no extrapola: toda predicción posterior al último año de
  entrenamiento colapsa a ese valor (verificado en el artefacto anterior: 2013 == 2040 == 2050).
"""
import re
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

BASE_DIR: Path = Path(__file__).resolve().parent.parent
CSV_NAME = "GlobalLandTemperaturesByMajorCity.csv"
RAW_CANDIDATES = [
    BASE_DIR / "Temperaturas" / "Global Land Temperatures" / CSV_NAME,
    BASE_DIR / "data" / "raw" / CSV_NAME,
    BASE_DIR / CSV_NAME,
]
OUT_DIR: Path = BASE_DIR / "data" / "processed"
MODEL_PATH: Path = OUT_DIR / "rf_model.joblib"

SPLIT_YEAR = 1980            # Train <= 1980 | Test > 1980 (partición temporal, sin fuga de futuro)
BASELINE = (1951, 1980)      # Línea base climatológica oficial
TREND_WINDOW = 60            # Años completos más recientes usados para la recta de cada ciudad
FEATURES = ["lat", "lon", "city_idx", "month_sin", "month_cos"]
RF_PARAMS = dict(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)


def parse_coord(value) -> float:
    """'19.25N' -> 19.25 | '99.10W' -> -99.10"""
    m = re.match(r"^\s*([\d.]+)\s*([NSEW])\s*$", str(value))
    if not m:
        raise ValueError(f"Coordenada inválida: {value!r}")
    val = float(m.group(1))
    return -val if m.group(2) in ("S", "W") else val


def cargar_dataset() -> pd.DataFrame:
    path = next((p for p in RAW_CANDIDATES if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No se encontró {CSV_NAME} en: {[str(p) for p in RAW_CANDIDATES]}")
    print(f"[+] Dataset: {path}")
    df = pd.read_csv(path).dropna(subset=["AverageTemperature", "dt", "City"])
    fechas = pd.to_datetime(df["dt"])
    df["year"], df["month"] = fechas.dt.year, fechas.dt.month
    df = df[df["year"] >= 1850].copy()
    df["lat"] = df["Latitude"].map(parse_coord)
    df["lon"] = df["Longitude"].map(parse_coord)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df.rename(columns={"AverageTemperature": "temp", "City": "city", "Country": "country"})


def medias_anuales_completas(df: pd.DataFrame) -> pd.DataFrame:
    """Solo años con los 12 meses (evita sesgo estacional en años parciales como 2013)."""
    g = df.groupby(["city", "year"])["temp"].agg(["mean", "count"]).reset_index()
    return g[g["count"] == 12].rename(columns={"mean": "temp"})[["city", "year", "temp"]]


def ajustar_tendencias(annual: pd.DataFrame) -> dict:
    tendencias = {}
    for city, g in annual.groupby("city"):
        g = g.sort_values("year")
        ventana = g[g["year"] > g["year"].max() - TREND_WINDOW]
        if len(ventana) < 20:
            ventana = g
        slope, intercept = np.polyfit(ventana["year"], ventana["temp"], 1)
        tendencias[city] = {"slope": float(slope), "intercept": float(intercept),
                            "last_year": int(g["year"].max())}
    return tendencias


def aplicar_tendencia(df: pd.DataFrame, tendencias: dict) -> np.ndarray:
    s = df["city"].map(lambda c: tendencias[c]["slope"]).to_numpy()
    b = df["city"].map(lambda c: tendencias[c]["intercept"]).to_numpy()
    return b + s * df["year"].to_numpy()


def entrenar(df: pd.DataFrame, encoder: LabelEncoder):
    """Ajusta tendencias + RF sobre el residuo. Devuelve (rf, tendencias)."""
    tendencias = ajustar_tendencias(medias_anuales_completas(df))
    # El residuo se aprende solo dentro de la ventana de la recta: fuera de ella la recta
    # extrapolada hacia atrás sesga el nivel y contamina el perfil estacional.
    inicio = df["city"].map(lambda c: tendencias[c]["last_year"] - TREND_WINDOW)
    ventana = df[df["year"] > inicio]
    X = ventana.assign(city_idx=encoder.transform(ventana["city"]))[FEATURES]
    y_resid = ventana["temp"].to_numpy() - aplicar_tendencia(ventana, tendencias)
    rf = RandomForestRegressor(**RF_PARAMS).fit(X, y_resid)
    return rf, tendencias


def predecir(df: pd.DataFrame, rf, tendencias, encoder) -> np.ndarray:
    X = df.assign(city_idx=encoder.transform(df["city"]))[FEATURES]
    return aplicar_tendencia(df, tendencias) + rf.predict(X)


def metricas(y, yhat) -> dict:
    return {"rmse": float(np.sqrt(mean_squared_error(y, yhat))),
            "mae": float(mean_absolute_error(y, yhat)),
            "r2": float(r2_score(y, yhat))}


def run_pipeline() -> dict:
    df = cargar_dataset()
    encoder = LabelEncoder().fit(sorted(df["city"].unique()))

    train, test = df[df["year"] <= SPLIT_YEAR], df[df["year"] > SPLIT_YEAR]
    print(f"[+] Partición temporal: train={len(train)} ({len(train)/len(df):.1%}) | test={len(test)} ({len(test)/len(df):.1%})")

    # Línea base: climatología ciudad-mes 1951-1980 (predice 'sin calentamiento')
    base = train[train["year"].between(*BASELINE)]
    clim = base.groupby(["city", "month"])["temp"].mean()
    clim_fallback = train.groupby(["city", "month"])["temp"].mean()
    idx = pd.MultiIndex.from_frame(test[["city", "month"]])
    y_base = clim.reindex(idx).fillna(clim_fallback.reindex(idx)).to_numpy()

    # Evaluación en holdout (> 1980) con modelo ajustado solo en <= 1980
    rf_eval, tend_eval = entrenar(train, encoder)
    y_pred = predecir(test, rf_eval, tend_eval, encoder)

    # Métrica anual (años completos): mide capacidad de capturar la anomalía, no solo la estacionalidad
    t = test.assign(pred=y_pred, base=y_base)
    full = t.groupby(["city", "year"]).filter(lambda g: len(g) == 12)
    anual = full.groupby(["city", "year"])[["temp", "pred", "base"]].mean()

    m = {
        "split_year": SPLIT_YEAR,
        "train_share": round(len(train) / len(df), 4),
        "modelo_mensual": metricas(test["temp"], y_pred),
        "baseline_mensual": metricas(test["temp"], y_base),
        "modelo_anual": metricas(anual["temp"], anual["pred"]),
        "baseline_anual": metricas(anual["temp"], anual["base"]),
    }
    m["mejora_rmse_anual_pct"] = round(
        (1 - m["modelo_anual"]["rmse"] / m["baseline_anual"]["rmse"]) * 100, 2)

    print("\n" + "=" * 62)
    for k in ("modelo_mensual", "baseline_mensual", "modelo_anual", "baseline_anual"):
        v = m[k]
        print(f"  {k:18s} RMSE={v['rmse']:.3f} °C  MAE={v['mae']:.3f} °C  R²={v['r2']:.4f}")
    print(f"  Mejora RMSE anual vs. baseline 1951-1980: {m['mejora_rmse_anual_pct']:+.2f}%")
    print("=" * 62 + "\n")

    # Modelo de producción: reajuste con todo el histórico (métricas = holdout anterior)
    rf_prod, tend_prod = entrenar(df, encoder)
    clim_all = df[df["year"].between(*BASELINE)].groupby(["city", "month"])["temp"].mean()
    meta = df.groupby("city")[["country", "lat", "lon"]].first()

    ciudades = {}
    for city, row in meta.iterrows():
        mensual = [float(clim_all.get((city, mes), np.nan)) for mes in range(1, 13)]
        ciudades[city] = {
            "country": row["country"], "lat": float(row["lat"]), "lon": float(row["lon"]),
            **tend_prod[city],
            "baseline_monthly": mensual,
            "baseline_annual": float(np.nanmean(mensual)),
        }

    payload = {
        "model": rf_prod,
        "encoder": encoder,
        "features": FEATURES,
        "cities": ciudades,
        "metrics": m,
        "config": {"baseline": BASELINE, "trend_window": TREND_WINDOW, "rf_params": RF_PARAMS},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL_PATH, compress=3)
    medias_anuales_completas(df).to_csv(OUT_DIR / "df_annual.csv", index=False)
    print(f"[OK] Artefacto serializado: {MODEL_PATH}")
    return m


if __name__ == "__main__":
    run_pipeline()
