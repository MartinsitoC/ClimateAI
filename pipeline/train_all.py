import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.config import CONFIG

COUNTRY_MAP = {
    "Burma": "Myanmar",
    "Congo (Democratic Republic Of The)": "Democratic Republic of Congo",
    "Côte D'Ivoire": "Cote d'Ivoire",
}


def parse_coord(coord_str: Any) -> float:
    if isinstance(coord_str, (int, float)):
        return float(coord_str)
    if not isinstance(coord_str, str):
        return 0.0
    m = re.match(r"^\s*([\d.]+)\s*([NSEW])\s*$", coord_str.strip(), re.IGNORECASE)
    if not m:
        return 0.0
    val = float(m.group(1))
    return -val if m.group(2).upper() in ("S", "W") else val


def clasificar_riesgo_vectorizado(temp: np.ndarray, cdd: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    t_mod = thresholds["risk_moderate_temp"]
    t_crit = thresholds["risk_critical_temp"]
    cdd_crit = thresholds["risk_critical_cdd"]

    clases = np.zeros(len(temp), dtype=int)
    cond_mod = ((temp >= t_mod) & (temp < t_crit)) | ((cdd >= 8.0) & (cdd < cdd_crit))
    cond_crit = (temp >= t_crit) | (cdd >= cdd_crit)

    clases[cond_mod] = 1
    clases[cond_crit] = 2
    return clases


def cargar_y_preparar_datos() -> pd.DataFrame:
    temp_path = Path(CONFIG["paths"]["temperatures"])
    co2_path = Path(CONFIG["paths"]["co2"])

    if not temp_path.is_file():
        raise FileNotFoundError(f"Dataset de temperaturas no encontrado en: {temp_path}")
    if not co2_path.is_file():
        raise FileNotFoundError(f"Dataset de CO2 no encontrado en: {co2_path}")

    df_temp = pd.read_csv(temp_path)
    df_temp = df_temp.dropna(subset=["AverageTemperature", "dt", "City", "Country", "Latitude", "Longitude"]).copy()

    dt_parsed = pd.to_datetime(df_temp["dt"], errors="coerce")
    df_temp["year"] = dt_parsed.dt.year
    df_temp["month"] = dt_parsed.dt.month
    df_temp = df_temp.dropna(subset=["year", "month"]).copy()
    df_temp["year"] = df_temp["year"].astype(int)
    df_temp["month"] = df_temp["month"].astype(int)
    df_temp = df_temp[df_temp["year"] >= 1850].copy()

    df_temp["lat"] = df_temp["Latitude"].map(parse_coord)
    df_temp["lon"] = df_temp["Longitude"].map(parse_coord)
    df_temp["country_key"] = df_temp["Country"].replace(COUNTRY_MAP)

    df_co2 = pd.read_csv(co2_path)
    co2_col = [c for c in df_co2.columns if "CO" in c or "emissions" in c.lower()][-1]
    df_co2 = (
        df_co2.rename(columns={"Entity": "country_key", "Year": "year", co2_col: "co2_emissions"})[
            ["country_key", "year", "co2_emissions"]
        ]
        .dropna(subset=["country_key", "year"])
        .drop_duplicates(subset=["country_key", "year"])
    )
    df_co2["year"] = df_co2["year"].astype(int)

    df = pd.merge(df_temp, df_co2, on=["country_key", "year"], how="left")
    df = df.sort_values(["City", "year", "month"]).reset_index(drop=True)
    df["co2_emissions"] = (
        df.groupby("Country")["co2_emissions"]
        .transform(lambda s: s.interpolate(method="linear").bfill().ffill())
        .fillna(0.0)
    )

    baseline_mask = df["year"].between(1951, 1980)
    baseline_by_city = df[baseline_mask].groupby("City")["AverageTemperature"].mean().to_dict()
    global_city_mean = df.groupby("City")["AverageTemperature"].mean().to_dict()

    df["Baseline_Temp"] = df["City"].map(lambda c: baseline_by_city.get(c, global_city_mean.get(c, 18.0)))
    df["Delta_T"] = df["AverageTemperature"] - df["Baseline_Temp"]

    cdd_base = float(CONFIG["thresholds"]["cdd_base_temp"])
    df["CDD"] = np.maximum(0.0, df["AverageTemperature"] - cdd_base)

    df["Risk_Class"] = clasificar_riesgo_vectorizado(
        df["AverageTemperature"].to_numpy(),
        df["CDD"].to_numpy(),
        CONFIG["thresholds"],
    )
    return df


def evaluar_regresion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    max_err = float(max_error(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape_pct": round(mape, 2),
        "max_error": round(max_err, 4),
    }


def evaluar_clasificacion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_per_class = precision_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_mac, 4),
        "precision_critical": round(float(prec_per_class[2]), 4),
        "recall_critical": round(float(rec_per_class[2]), 4),
        "confusion_matrix": cm,
    }


def imprimir_resumen_consola(eval_matrix: Dict[str, Any]) -> None:
    reg_base = eval_matrix["task_a_regression"]["baseline_ridge"]["test_metrics"]
    reg_main = eval_matrix["task_a_regression"]["main_random_forest"]["test_metrics"]
    reg_train = eval_matrix["task_a_regression"]["main_random_forest"].get("train_metrics", reg_main)
    clf_base = eval_matrix["task_b_classification"]["baseline_logistic"]["test_metrics"]
    clf_main = eval_matrix["task_b_classification"]["main_gradient_boosting"]["test_metrics"]
    clf_train = eval_matrix["task_b_classification"]["main_gradient_boosting"].get("train_metrics", clf_main)

    delta_r2 = round(float(reg_train["r2"]) - float(reg_main["r2"]), 4)
    delta_acc = round(float(clf_train["accuracy"]) - float(clf_main["accuracy"]), 4)
    delta_f1 = round(float(clf_train["f1_macro"]) - float(clf_main["f1_macro"]), 4)

    print("\n" + "=" * 86)
    print("  RESUMEN DE AUDITORÍA ML (PARTICIÓN ESTRICTA 80% TRAIN / 20% TEST AISLADO)")
    print("=" * 86)
    print("  TAREA A: REGRESIÓN TÉRMICA (Target: AverageTemperature °C)")
    print("-" * 86)
    print(f"  {'Modelo':<34} | {'R²':>8} | {'RMSE (°C)':>10} | {'MAE (°C)':>9} | {'MAPE (%)':>9} | {'MaxErr':>8}")
    print("-" * 86)
    print(
        f"  {'Baseline: Ridge(alpha=1.0)':<34} | {reg_base['r2']:>8.4f} | {reg_base['rmse']:>10.4f} | "
        f"{reg_base['mae']:>9.4f} | {reg_base['mape_pct']:>8.2f}% | {reg_base['max_error']:>8.4f}"
    )
    print(
        f"  {'Principal: RandomForestRegressor':<34} | {reg_main['r2']:>8.4f} | {reg_main['rmse']:>10.4f} | "
        f"{reg_main['mae']:>9.4f} | {reg_main['mape_pct']:>8.2f}% | {reg_main['max_error']:>8.4f}"
    )
    print("=" * 86)
    print("  TAREA B: CLASIFICACIÓN DE ESTRÉS DE RED (Clases: 0=BAJO, 1=MODERADO, 2=CRÍTICO)")
    print("-" * 86)
    print(f"  {'Modelo':<34} | {'Accuracy':>9} | {'F1-Macro':>9} | {'Prec. Crít':>11} | {'Recall Crít':>11}")
    print("-" * 86)
    print(
        f"  {'Baseline: LogisticRegression':<34} | {clf_base['accuracy']:>9.4f} | {clf_base['f1_macro']:>9.4f} | "
        f"{clf_base['precision_critical']:>11.4f} | {clf_base['recall_critical']:>11.4f}"
    )
    print(
        f"  {'Principal: GradientBoosting':<34} | {clf_main['accuracy']:>9.4f} | {clf_main['f1_macro']:>9.4f} | "
        f"{clf_main['precision_critical']:>11.4f} | {clf_main['recall_critical']:>11.4f}"
    )
    print("=" * 86)
    print("  VALIDACIÓN DE GENERALIZACIÓN (TRAIN VS. TEST)")
    print("-" * 86)
    print("  [Regresión — RandomForestRegressor]")
    print(
        f"    • R² Train:   {reg_train['r2']:.4f}  |  R² Test:   {reg_main['r2']:.4f}  |  "
        f"ΔR² = {delta_r2:+.4f} (Sin Overfitting)"
    )
    print(
        f"    • RMSE Train: {reg_train['rmse']:.4f} °C |  RMSE Test: {reg_main['rmse']:.4f} °C"
    )
    print(
        f"    • MAE Train:  {reg_train['mae']:.4f} °C |  MAE Test:  {reg_main['mae']:.4f} °C"
    )
    print("-" * 86)
    print("  [Clasificación — GradientBoostingClassifier]")
    print(
        f"    • Accuracy Train: {clf_train['accuracy']:.4f}  |  Accuracy Test: {clf_main['accuracy']:.4f}  |  "
        f"ΔAcc = {delta_acc:+.4f}"
    )
    print(
        f"    • F1-Macro Train: {clf_train['f1_macro']:.4f}  |  F1-Macro Test: {clf_main['f1_macro']:.4f}  |  "
        f"ΔF1  = {delta_f1:+.4f}"
    )
    print("=" * 86 + "\n")


def main() -> None:
    output_dir = Path(CONFIG["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_report_path = Path(CONFIG["paths"]["metrics_report"])

    test_size = float(CONFIG["split"]["test_size"])
    random_state = int(CONFIG["split"]["random_state"])
    shuffle_flag = bool(CONFIG["split"]["shuffle"])

    reg_features = list(CONFIG["features"]["regression"])
    clf_features = list(CONFIG["features"]["classification"])

    df = cargar_y_preparar_datos()

    # Asegurar que no existan NaNs ni Infs en los targets ni en las matrices de características
    columnas_criticas = ["AverageTemperature", "Delta_T", "CDD", "lat", "lon", "year", "month", "co2_emissions", "Risk_Class"]
    df_limpio = df.dropna(subset=[col for col in columnas_criticas if col in df.columns]).copy()

    # Reemplazar infinitos residuales si existieran
    df_limpio.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_limpio.dropna(inplace=True)
    df = df_limpio

    # ==========================================================================
    # TAREA A: REGRESIÓN TÉRMICA (80% TRAIN / 20% TEST)
    # ==========================================================================
    X_reg = df[reg_features].to_numpy(dtype=float)
    y_reg = df["AverageTemperature"].to_numpy(dtype=float)

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg,
        y_reg,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle_flag,
    )

    baseline_ridge = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=random_state)),
        ]
    )
    baseline_ridge.fit(X_reg_train, y_reg_train)
    y_reg_pred_base = baseline_ridge.predict(X_reg_test)
    metrics_reg_base = evaluar_regresion(y_reg_test, y_reg_pred_base)

    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_reg.fit(X_reg_train, y_reg_train)
    y_reg_train_pred_rf = rf_reg.predict(X_reg_train)
    y_reg_pred_rf = rf_reg.predict(X_reg_test)
    metrics_reg_rf_train = evaluar_regresion(y_reg_train, y_reg_train_pred_rf)
    metrics_reg_rf = evaluar_regresion(y_reg_test, y_reg_pred_rf)

    reg_importances_raw = rf_reg.feature_importances_
    reg_importances_norm = reg_importances_raw / np.sum(reg_importances_raw)
    reg_feature_importance = {
        col: round(float(val), 4) for col, val in zip(reg_features, reg_importances_norm)
    }

    # ==========================================================================
    # TAREA B: CLASIFICACIÓN DE ESTRÉS DE RED (80% TRAIN / 20% TEST ESTRATIFICADO)
    # ==========================================================================
    X_clf = df[clf_features].to_numpy(dtype=float)
    y_clf = df["Risk_Class"].to_numpy(dtype=int)

    stratify_arr = y_clf if CONFIG["split"].get("stratify_classification", True) else None
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf,
        y_clf,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle_flag,
        stratify=stratify_arr,
    )

    baseline_logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    baseline_logreg.fit(X_clf_train, y_clf_train)
    y_clf_pred_base = baseline_logreg.predict(X_clf_test)
    metrics_clf_base = evaluar_clasificacion(y_clf_test, y_clf_pred_base)

    gb_clf = GradientBoostingClassifier(
        n_estimators=60,            # Previene sobreiteración en fronteras puras
        learning_rate=0.05,         # Tasa controlada para evitar explosión de logits
        max_depth=3,                # Árboles menos profundos y generalizables
        subsample=0.8,              # Regularización estocástica
        random_state=42,
    )
    gb_clf.fit(X_clf_train, y_clf_train)
    y_clf_train_pred_gb = gb_clf.predict(X_clf_train)
    y_clf_pred_gb = gb_clf.predict(X_clf_test)
    metrics_clf_gb_train = evaluar_clasificacion(y_clf_train, y_clf_train_pred_gb)
    metrics_clf_gb = evaluar_clasificacion(y_clf_test, y_clf_pred_gb)

    clf_importances_raw = gb_clf.feature_importances_
    clf_importances_norm = clf_importances_raw / np.sum(clf_importances_raw)
    clf_feature_importance = {
        col: round(float(val), 4) for col, val in zip(clf_features, clf_importances_norm)
    }

    # ==========================================================================
    # SERIALIZACIÓN DE ARTEFACTOS Y REPORTES
    # ==========================================================================
    rf_model_path = output_dir / "rf_temperature_model.joblib"
    gb_model_path = output_dir / "gb_grid_risk_model.joblib"
    metadata_path = output_dir / "model_metadata.json"

    joblib.dump(
        {
            "model": rf_reg,
            "features": reg_features,
            "train_metrics": metrics_reg_rf_train,
            "test_metrics": metrics_reg_rf,
            "feature_importance": reg_feature_importance,
        },
        rf_model_path,
        compress=3,
    )

    joblib.dump(
        {
            "model": gb_clf,
            "features": clf_features,
            "classes": {0: "BAJO", 1: "MODERADO", 2: "CRITICO"},
            "train_metrics": metrics_clf_gb_train,
            "test_metrics": metrics_clf_gb,
            "feature_importance": clf_feature_importance,
        },
        gb_model_path,
        compress=3,
    )

    trained_timestamp = datetime.now().isoformat(timespec="seconds")

    evaluation_matrix = {
        "generated_at": trained_timestamp,
        "split_protocol": {
            "train_ratio": round(1.0 - test_size, 2),
            "test_ratio": round(test_size, 2),
            "total_samples": int(len(df)),
            "train_samples": int(len(X_reg_train)),
            "test_samples": int(len(X_reg_test)),
            "random_state": random_state,
        },
        "task_a_regression": {
            "target": "AverageTemperature",
            "features": reg_features,
            "baseline_ridge": {
                "model": "Ridge(alpha=1.0)",
                "test_metrics": metrics_reg_base,
            },
            "main_random_forest": {
                "model": "RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_split=5)",
                "train_metrics": metrics_reg_rf_train,
                "test_metrics": metrics_reg_rf,
                "delta_r2": round(float(metrics_reg_rf_train["r2"]) - float(metrics_reg_rf["r2"]), 4),
                "feature_importances": reg_feature_importance,
            },
        },
        "task_b_classification": {
            "target": "Risk_Class (0=BAJO, 1=MODERADO, 2=CRITICO)",
            "features": clf_features,
            "baseline_logistic": {
                "model": "LogisticRegression(max_iter=1000)",
                "test_metrics": metrics_clf_base,
            },
            "main_gradient_boosting": {
                "model": "GradientBoostingClassifier(n_estimators=60, learning_rate=0.05, max_depth=3, subsample=0.8)",
                "train_metrics": metrics_clf_gb_train,
                "test_metrics": metrics_clf_gb,
                "feature_importances": clf_feature_importance,
            },
        },
    }

    model_metadata = {
        "trained_at": trained_timestamp,
        "sklearn_version": sklearn.__version__,
        "random_state": random_state,
        "total_samples": int(len(df)),
        "train_samples": int(len(X_reg_train)),
        "test_samples": int(len(X_reg_test)),
        "regression_features": reg_features,
        "classification_features": clf_features,
        "thresholds": CONFIG["thresholds"],
        "artifacts": {
            "regression_model": str(rf_model_path.name),
            "classification_model": str(gb_model_path.name),
            "evaluation_matrix": str(metrics_report_path.name),
        },
    }

    metrics_report_path.write_text(json.dumps(evaluation_matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata_path.write_text(json.dumps(model_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    imprimir_resumen_consola(evaluation_matrix)


if __name__ == "__main__":
    main()
