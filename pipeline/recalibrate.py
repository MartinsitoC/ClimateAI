import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use('Agg')  # Previene bloqueos de hilos gráficos en terminal
import matplotlib.pyplot as plt

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.config import CONFIG
from pipeline.train_all import (
    cargar_y_preparar_datos,
    evaluar_clasificacion,
    evaluar_regresion,
)


def imprimir_matriz_confusion_consola(cm: np.ndarray) -> None:
    etiquetas = ["Bajo (0)", "Moderado (1)", "Crítico (2)"]
    print("\n" + "-" * 88)
    print("  MATRIZ DE CONFUSIÓN EN TEST (Real \\ Predicho: Bajo, Moderado, Crítico)")
    print("-" * 88)
    print(f"  {'Real \\ Predicho':<20} | {'Bajo (0)':>14} | {'Moderado (1)':>14} | {'Crítico (2)':>14} | {'Total Fila':>12}")
    print("-" * 88)
    for idx, nombre_real in enumerate(etiquetas):
        fila = cm[idx]
        total_f = int(np.sum(fila))
        print(
            f"  {nombre_real:<20} | {int(fila[0]):>14,d} | {int(fila[1]):>14,d} | "
            f"{int(fila[2]):>14,d} | {total_f:>12,d}"
        )
    print("-" * 88)


def exportar_graficos_diagnostico(
    output_dir: Path,
    y_reg_true: np.ndarray,
    y_reg_pred: np.ndarray,
    cm: np.ndarray,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    residuos = y_reg_true - y_reg_pred
    media_res = float(np.mean(residuos))
    residuos_path = output_dir / "curva_residuos.png"
    cm_path = output_dir / "matriz_confusion.png"

    # 1. Curva y dispersión de residuos térmicos
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    axes[0].hist(residuos, bins=50, color="#ea580c", edgecolor="#1e293b", alpha=0.85)
    axes[0].axvline(media_res, color="#0f172a", linestyle="--", linewidth=1.5, label=f"Media = {media_res:+.4f} °C")
    axes[0].set_title(f"Distribución de Residuos Térmicos (Media ≈ {media_res:+.3f} °C)", fontweight="bold")
    axes[0].set_xlabel("Error Residual (y_test - y_pred) [°C]")
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.25)

    sample_idx = np.linspace(0, len(y_reg_pred) - 1, min(3000, len(y_reg_pred)), dtype=int)
    axes[1].scatter(
        y_reg_pred[sample_idx],
        residuos[sample_idx],
        s=8,
        alpha=0.35,
        color="#f59e0b",
        edgecolors="none",
    )
    axes[1].axhline(0.0, color="#ef4444", linestyle="--", linewidth=1.5)
    axes[1].set_title("Dispersión de Residuos vs. Predicción", fontweight="bold")
    axes[1].set_xlabel("Temperatura Predicha (°C)")
    axes[1].set_ylabel("Residuo (y_test - y_pred) [°C]")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(residuos_path, dpi=300)
    plt.close(fig)

    # 2. Matriz de confusión de alta legibilidad (300 DPI, paleta Blues)
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm.astype(float) / row_sums
    etiquetas = ["Bajo (0)", "Moderado (1)", "Crítico (2)"]

    fig2, ax = plt.subplots(figsize=(7.5, 6.2), dpi=300)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proporción por Clase Real", rotation=270, labelpad=15, fontsize=10, fontweight="bold")

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(etiquetas, fontsize=11, fontweight="bold")
    ax.set_yticklabels(etiquetas, fontsize=11, fontweight="bold")
    ax.set_title(
        "Matriz de Confusión — Clasificación de Estrés de Red\n(Conteo Absoluto y % por Fila)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Clase Real (Observada)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Clase Predicha (Gradient Boosting)", fontsize=11, fontweight="bold")

    for i in range(3):
        for j in range(3):
            conteo = int(cm[i, j])
            pct = cm_norm[i, j] * 100.0
            valor_txt = f"{conteo:,d}\n({pct:.1f}%)"
            color_txt = "white" if cm_norm[i, j] > 0.5 else "#0f172a"
            ax.text(
                j,
                i,
                valor_txt,
                ha="center",
                va="center",
                color=color_txt,
                fontsize=13,
                fontweight="bold",
            )

    plt.tight_layout()
    fig2.savefig(cm_path, dpi=300)
    plt.close(fig2)
    plt.close('all')

    return {
        "curva_residuos": str(residuos_path),
        "matriz_confusion": str(cm_path),
    }


def ejecutar_recalibracion(
    split_ratio: float = 0.20,
    kfold: Optional[int] = None,
    n_trees: int = 100,
    export_plots: bool = False,
) -> Dict[str, Any]:
    if not (0.05 <= split_ratio <= 0.50):
        raise ValueError("El argumento --split debe estar entre 0.05 y 0.50.")

    output_dir = Path(CONFIG["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_report_path = Path(CONFIG["paths"]["metrics_report"])
    metadata_path = output_dir / "model_metadata.json"

    random_state = int(CONFIG["split"]["random_state"])
    reg_features = list(CONFIG["features"]["regression"])
    clf_features = list(CONFIG["features"]["classification"])

    df = cargar_y_preparar_datos()

    columnas_requeridas = ["AverageTemperature", "Delta_T", "CDD", "lat", "lon", "year", "month", "co2_emissions", "Risk_Class"]
    df_clean = df.dropna(subset=[c for c in columnas_requeridas if c in df.columns]).copy()
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.dropna(inplace=True)
    df = df_clean

    X_reg = df[reg_features].to_numpy(dtype=float)
    y_reg = df["AverageTemperature"].to_numpy(dtype=float)

    X_clf = df[clf_features].to_numpy(dtype=float)
    y_clf = df["Risk_Class"].to_numpy(dtype=int)

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=split_ratio, random_state=random_state, shuffle=True
    )

    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=split_ratio, random_state=random_state, shuffle=True, stratify=y_clf
    )

    # Entrenamiento Regresión
    baseline_ridge = Pipeline(
        steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=random_state))]
    )
    baseline_ridge.fit(X_reg_train, y_reg_train)
    metrics_reg_base = evaluar_regresion(y_reg_test, baseline_ridge.predict(X_reg_test))

    rf_reg = RandomForestRegressor(
        n_estimators=int(n_trees),
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

    reg_imp_norm = rf_reg.feature_importances_ / np.sum(rf_reg.feature_importances_)
    reg_feature_importance = {col: round(float(v), 4) for col, v in zip(reg_features, reg_imp_norm)}

    # Entrenamiento Clasificación
    baseline_logreg = Pipeline(
        steps=[("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000, random_state=random_state))]
    )
    baseline_logreg.fit(X_clf_train, y_clf_train)
    metrics_clf_base = evaluar_clasificacion(y_clf_test, baseline_logreg.predict(X_clf_test))

    gb_clf = GradientBoostingClassifier(
        n_estimators=60,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    gb_clf.fit(X_clf_train, y_clf_train)
    y_clf_train_pred_gb = gb_clf.predict(X_clf_train)
    y_clf_pred_gb = gb_clf.predict(X_clf_test)
    metrics_clf_gb_train = evaluar_clasificacion(y_clf_train, y_clf_train_pred_gb)
    metrics_clf_gb = evaluar_clasificacion(y_clf_test, y_clf_pred_gb)

    clf_imp_norm = gb_clf.feature_importances_ / np.sum(gb_clf.feature_importances_)
    clf_feature_importance = {col: round(float(v), 4) for col, v in zip(clf_features, clf_imp_norm)}

    # Validación Cruzada K-Fold opcional
    kfold_summary = None
    if kfold is not None and int(kfold) >= 2:
        k_val = int(kfold)
        # Submuestra representativa estratificada si el dataset supera 40,000 filas para mantener agilidad en CLI
        if len(df) > 40000:
            rng = np.random.default_rng(random_state)
            idx_cv = rng.choice(len(df), size=40000, replace=False)
            X_reg_cv, y_reg_cv = X_reg[idx_cv], y_reg[idx_cv]
            X_clf_cv, y_clf_cv = X_clf[idx_cv], y_clf[idx_cv]
        else:
            X_reg_cv, y_reg_cv = X_reg, y_reg
            X_clf_cv, y_clf_cv = X_clf, y_clf

        kf_reg = KFold(n_splits=k_val, shuffle=True, random_state=random_state)
        r2_scores = cross_val_score(rf_reg, X_reg_cv, y_reg_cv, cv=kf_reg, scoring="r2", n_jobs=-1)

        skf_clf = StratifiedKFold(n_splits=k_val, shuffle=True, random_state=random_state)
        acc_scores = cross_val_score(gb_clf, X_clf_cv, y_clf_cv, cv=skf_clf, scoring="accuracy", n_jobs=-1)

        kfold_summary = {
            "k_folds": k_val,
            "regression_r2_mean": round(float(np.mean(r2_scores)), 4),
            "regression_r2_std": round(float(np.std(r2_scores)), 4),
            "classification_acc_mean": round(float(np.mean(acc_scores)), 4),
            "classification_acc_std": round(float(np.std(acc_scores)), 4),
        }

    cm_array = np.array(metrics_clf_gb["confusion_matrix"], dtype=int)
    plots_exported = {}
    if export_plots:
        plots_exported = exportar_graficos_diagnostico(output_dir, y_reg_test, y_reg_pred_rf, cm_array)

    rf_model_path = output_dir / "rf_temperature_model.joblib"
    gb_model_path = output_dir / "gb_grid_risk_model.joblib"

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
            "train_ratio": round(1.0 - split_ratio, 2),
            "test_ratio": round(split_ratio, 2),
            "total_samples": int(len(df)),
            "train_samples": int(len(X_reg_train)),
            "test_samples": int(len(X_reg_test)),
            "random_state": random_state,
            "rf_trees": int(n_trees),
        },
        "kfold_validation": kfold_summary,
        "exported_plots": plots_exported,
        "task_a_regression": {
            "target": "AverageTemperature",
            "features": reg_features,
            "baseline_ridge": {"model": "Ridge(alpha=1.0)", "test_metrics": metrics_reg_base},
            "main_random_forest": {
                "model": f"RandomForestRegressor(n_estimators={n_trees}, max_depth=14, min_samples_split=5)",
                "train_metrics": metrics_reg_rf_train,
                "test_metrics": metrics_reg_rf,
                "delta_r2": round(float(metrics_reg_rf_train["r2"]) - float(metrics_reg_rf["r2"]), 4),
                "feature_importances": reg_feature_importance,
            },
        },
        "task_b_classification": {
            "target": "Risk_Class (0=BAJO, 1=MODERADO, 2=CRITICO)",
            "features": clf_features,
            "baseline_logistic": {"model": "LogisticRegression(max_iter=1000)", "test_metrics": metrics_clf_base},
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
        "train_ratio": round(1.0 - split_ratio, 2),
        "test_ratio": round(split_ratio, 2),
        "rf_trees": int(n_trees),
        "kfold": kfold_summary,
        "exported_plots": plots_exported,
    }

    metrics_report_path.write_text(json.dumps(evaluation_matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata_path.write_text(json.dumps(model_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 88)
    print("  RESUMEN EJECUTIVO DE RECALIBRACIÓN CLI — URBAN-HEAT GRID INTELLIGENCE")
    print("=" * 88)
    print(
        f"  • Partición Aplicada:   Train {1.0 - split_ratio:.0%} ({len(X_reg_train):,}) / "
        f"Test {split_ratio:.0%} ({len(X_reg_test):,}) | Árboles RF: {n_trees}"
    )
    print(
        f"  • Regresión RF (Train): R² = {metrics_reg_rf_train['r2']:.4f} | "
        f"RMSE = {metrics_reg_rf_train['rmse']:.4f} °C | MAE = {metrics_reg_rf_train['mae']:.4f} °C"
    )
    print(
        f"  • Regresión RF (Test):  R² = {metrics_reg_rf['r2']:.4f} | "
        f"RMSE = {metrics_reg_rf['rmse']:.4f} °C | MAE = {metrics_reg_rf['mae']:.4f} °C | "
        f"ΔR² = {metrics_reg_rf_train['r2'] - metrics_reg_rf['r2']:+.4f}"
    )
    print(
        f"  • Clasificación GB:     Acc Train = {metrics_clf_gb_train['accuracy']:.4f} vs Test = {metrics_clf_gb['accuracy']:.4f} | "
        f"F1-Macro Train = {metrics_clf_gb_train['f1_macro']:.4f} vs Test = {metrics_clf_gb['f1_macro']:.4f}"
    )
    if kfold_summary:
        print(
            f"  • K-Fold ({kfold_summary['k_folds']}-Fold CV):  "
            f"R² = {kfold_summary['regression_r2_mean']:.4f} ± {kfold_summary['regression_r2_std']:.4f} | "
            f"Acc = {kfold_summary['classification_acc_mean']:.4f} ± {kfold_summary['classification_acc_std']:.4f}"
        )
    imprimir_matriz_confusion_consola(cm_array)
    if plots_exported:
        print(f"  • Gráficos Exportados:  {plots_exported['curva_residuos']} | {plots_exported['matriz_confusion']}")
    print("=" * 88 + "\n")

    return evaluation_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI de recalibración instantánea y validación cruzada para URBAN-HEAT GRID INTELLIGENCE."
    )
    parser.add_argument("--split", type=float, default=0.20, help="Porcentaje de prueba (ej. 0.20, 0.25, 0.30)")
    parser.add_argument("--kfold", type=int, default=None, help="Número de particiones para validación cruzada K-Fold")
    parser.add_argument("--trees", type=int, default=100, help="Número de árboles para RandomForestRegressor")
    parser.add_argument(
        "--export-plots",
        action="store_true",
        help="Exportar curva_residuos.png y matriz_confusion.png en data/processed/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ejecutar_recalibracion(
        split_ratio=args.split,
        kfold=args.kfold,
        n_trees=args.trees,
        export_plots=args.export_plots,
    )
