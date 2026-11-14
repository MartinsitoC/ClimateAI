"""
URBAN-HEAT GRID INTELLIGENCE — Script de Verificación en Cascada, Benchmark de Latencia y XAI
Archivo: pipeline/benchmark_inference.py
Uso: python pipeline/benchmark_inference.py
"""
import sys
import time
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.model import PredictionInput, UrbanHeatInferenceEngine


def ejecutar_validacion_casos_criticos(engine: UrbanHeatInferenceEngine) -> None:
    casos = [
        {"city": "Baghdad", "target_year": 2045, "month": 7, "co2": 220.0},
        {"city": "London", "target_year": 2030, "month": 7, "co2": 340.0},
        {"city": "Singapore", "target_year": 2050, "month": 5, "co2": 85.0},
    ]

    print("\n" + "=" * 92)
    print("  1. VALIDACIÓN DE CASOS CRÍTICOS EN CASCADA (REGRESIÓN RF -> DERIVADAS -> CLASIFICACIÓN GB)")
    print("=" * 92)

    for caso in casos:
        res = engine.predecir(PredictionInput(**caso))
        probs = res.probabilidades_riesgo
        print(
            f"  [{res.city.upper()} - {res.target_year}] ({res.country}) | "
            f"Lat/Lon: ({res.lat:.2f}, {res.lon:.2f})"
        )
        print(
            f"    • Temp Predicha: {res.prediccion_temperatura:.2f} °C "
            f"(±{res.sigma_temp:.2f} °C | IC 95%: [{res.rango_min:.2f}, {res.rango_max:.2f}] °C)"
        )
        print(
            f"    • Derivadas:     ΔT = {res.anomalia_proyectada:+.2f} °C | "
            f"CDD = {res.cdd_anual:.2f} °C·día ({res.porcentaje_aumento_cdd:+.1f}%)"
        )
        print(
            f"    • Probabilidades (predict_proba): "
            f"BAJO={probs['BAJO']:.2f}% | MODERADO={probs['MODERADO']:.2f}% | CRÍTICO={probs['CRÍTICO']:.2f}%"
        )
        print(
            f"    • Diagnóstico:   [{res.alerta_operativa}] — Estrés Red: {res.nivel_estres_red} "
            f"(Certeza: {res.probabilidad_contingencia:.2f}%)"
        )
        print(f"    • Plan Acción:   1) {res.plan_accion[0]}")
        print("-" * 92)


def ejecutar_benchmark_latencia(engine: UrbanHeatInferenceEngine, n_iter: int = 1000) -> None:
    rng = np.random.default_rng(42)
    ciudades_catalogo = list(engine.cities.keys()) if engine.cities else ["Baghdad", "London", "Singapore", "Mexico"]

    # Pre-generar 1,000 payloads sintéticos representativos
    consultas = []
    for _ in range(n_iter):
        c = str(rng.choice(ciudades_catalogo))
        yr = int(rng.integers(1980, 2051))
        mo = int(rng.integers(1, 13))
        co2_val = float(rng.uniform(20.0, 500.0))
        consultas.append(PredictionInput(city=c, target_year=yr, month=mo, co2=co2_val))

    # Calentamiento de caché (warm-up)
    for i in range(10):
        engine.predecir(consultas[i])

    latencias_ms = np.empty(n_iter, dtype=float)
    t_start_total = time.perf_counter()
    for i in range(n_iter):
        t0 = time.perf_counter()
        engine.predecir(consultas[i])
        latencias_ms[i] = (time.perf_counter() - t0) * 1000.0
    total_s = time.perf_counter() - t_start_total

    media_ms = float(np.mean(latencias_ms))
    p50_ms = float(np.percentile(latencias_ms, 50))
    p95_ms = float(np.percentile(latencias_ms, 95))
    p99_ms = float(np.percentile(latencias_ms, 99))
    throughput = n_iter / max(total_s, 1e-6)
    cumple_sla = media_ms < 15.0

    print("\n" + "=" * 92)
    print(f"  2. BENCHMARK DE LATENCIA Y ESTRÉS ({n_iter:,} CONSULTAS CONSECUTIVAS EN CASCADA)")
    print("=" * 92)
    print(f"  • Tiempo Total (1,000 inferencias): {total_s:.3f} s ({throughput:,.1f} req/s)")
    print(f"  • Latencia Promedio:                {media_ms:.3f} ms")
    print(f"  • Mediana (P50):                    {p50_ms:.3f} ms")
    print(f"  • Percentil 95 (P95):               {p95_ms:.3f} ms")
    print(f"  • Percentil 99 (P99):               {p99_ms:.3f} ms")
    print(f"  • Verificación SLA (< 15.0 ms):     {'[OK - APROBADO]' if cumple_sla else '[FALLO - EXCEDE SLA]'}")
    print("=" * 92)


def imprimir_resumen_xai(engine: UrbanHeatInferenceEngine) -> None:
    importances = engine.reg_feature_importance
    if not importances and hasattr(engine.model_rf, "feature_importances_"):
        raw = engine.model_rf.feature_importances_
        norm = raw / np.sum(raw)
        importances = {col: float(val) for col, val in zip(engine.reg_features, norm)}

    top_3 = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:3]

    print("\n" + "=" * 92)
    print("  3. RESUMEN DE EXPLICABILIDAD (XAI) — TOP 3 VARIABLES EN REGRESIÓN TÉRMICA")
    print("=" * 92)
    for rank, (feat, weight) in enumerate(top_3, start=1):
        pct = weight * 100.0
        bar = "█" * int(round(pct / 2.5))
        print(f"  #{rank}  {feat:<18} | Peso: {weight:.4f} ({pct:6.2f}%) | {bar}")
    print("=" * 92 + "\n")


def main() -> None:
    engine = UrbanHeatInferenceEngine.get_instance()
    ejecutar_validacion_casos_criticos(engine)
    ejecutar_benchmark_latencia(engine, n_iter=1000)
    imprimir_resumen_xai(engine)


if __name__ == "__main__":
    main()
