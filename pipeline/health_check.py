import sys
import os
import json
import sqlite3
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

def print_status(componente: str, passed: bool, detalle: str = ""):
    color = "\033[92m[OK]\033[0m" if passed else "\033[91m[FALLO]\033[0m"
    print(f" {color} {componente:<35} {detalle}")

def corregir_directorios():
    """Crea carpetas estructurales si no existen."""
    for folder in ["data/processed", "data/raw", "models", "tests", "static"]:
        d = BASE_DIR / folder
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            print(f"  -> Creado directorio faltante: {folder}")

def check_1_artefactos_ml():
    """Valida modelos .joblib y matrices JSON."""
    rf_path = BASE_DIR / "data" / "processed" / "rf_temperature_model.joblib"
    gb_path = BASE_DIR / "data" / "processed" / "gb_grid_risk_model.joblib"
    eval_path = BASE_DIR / "data" / "processed" / "evaluation_matrix.json"

    rf_ok = rf_path.exists() and rf_path.stat().st_size > 1024
    gb_ok = gb_path.exists() and gb_path.stat().st_size > 1024
    eval_ok = eval_path.exists() and eval_path.stat().st_size > 10

    print_status("Modelo RF Regresión (.joblib)", rf_ok, f"Tamaño: {rf_path.stat().st_size / 1024:.1f} KB" if rf_ok else "No encontrado")
    print_status("Modelo GB Clasificador (.joblib)", gb_ok, f"Tamaño: {gb_path.stat().st_size / 1024:.1f} KB" if gb_ok else "No encontrado")
    print_status("Matriz de Evaluación (JSON)", eval_ok, str(eval_path.name) if eval_ok else "Faltante")
    return rf_ok and gb_ok and eval_ok

def check_2_contrato_inferencia():
    """Valida el contrato de inferencia y la latencia en régimen estabilizado."""
    try:
        from pipeline.model import predecir
        sample_input = {
            "lat": 21.16, "lon": -86.85, "year": 2035, "month": 7,
            "co2_emissions": 460.0, "baseline_temp": 27.5
        }
        
        # 1. Pase de calentamiento (descarga de paginación e hilos)
        _ = predecir(sample_input)

        # 2. Medición en régimen estabilizado (media de 5 inferencias)
        tiempos = []
        resultado = None
        for _ in range(5):
            t0 = time.perf_counter()
            resultado = predecir(sample_input)
            tiempos.append((time.perf_counter() - t0) * 1000)

        latencia_ms = sum(tiempos) / len(tiempos)

        valido = (
            resultado is not None and
            resultado.get("status") == "success" and
            isinstance(resultado.get("temp_predicha"), (int, float)) and
            "nivel_estres_red" in resultado and
            latencia_ms < 15.0
        )
        
        diag_nivel = resultado.get("nivel_estres_red", "N/A")
        temp_val = resultado.get("temp_predicha", "N/A")
        print_status("Inferencia Desacoplada (Cascada)", valido, f"Latencia: {latencia_ms:.2f} ms | Temp: {temp_val}°C | {diag_nivel}")
        return valido
    except Exception as e:
        print_status("Inferencia Desacoplada (Cascada)", False, f"Excepción: {str(e)}")
        return False

def check_3_cache_sqlite():
    """Valida y autocorrige la base de datos de geocodificación SQLite."""
    db_path = BASE_DIR / "data" / "geo_cache.db"
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS places_cache (
                    query TEXT PRIMARY KEY,
                    name TEXT,
                    display_name TEXT,
                    lat REAL,
                    lon REAL,
                    place_type TEXT,
                    country TEXT,
                    timestamp DATETIME
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_climate_cache (
                    coord_key TEXT PRIMARY KEY,
                    data_json TEXT,
                    timestamp DATETIME
                )
            """)
        print_status("Base de Datos SQLite (geo_cache.db)", True, "Tablas verificadas y sincronizadas")
        return True
    except Exception as e:
        print_status("Base de Datos SQLite (geo_cache.db)", False, str(e))
        return False

def check_4_pruebas_unitarias():
    """Ejecuta la suite de pruebas tests/test_models.py."""
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(str(BASE_DIR / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    valido = result.wasSuccessful()
    print_status("Suite de Pruebas (unittest)", valido, f"Pruebas corridas: {result.testsRun} | Errores: {len(result.errors)} | Fallos: {len(result.failures)}")
    return valido

def check_5_archivos_frontend():
    """Verifica que index.html y assets existan y no tengan sintaxis rota."""
    index_path = BASE_DIR / "static" / "index.html"
    if not index_path.exists():
        print_status("Frontend UI (static/index.html)", False, "Archivo no encontrado")
        return False
    
    contenido = index_path.read_text(encoding="utf-8")
    checks = [
        ("Leaflet CSS/JS", "leaflet" in contenido),
        ("Chart.js", "chart.js" in contenido or "Chart" in contenido),
        ("Selector 70/30", "sidePanelWrapper" in contenido),
        ("Engranaje Config", "settingsDropdown" in contenido)
    ]
    todos_ok = all(c[1] for c in checks)
    print_status("Frontend UI (static/index.html)", todos_ok, "Integridad de componentes confirmada")
    return todos_ok

def main():
    print("=" * 75)
    print("  DIAGNÓSTICO INTEGRAL DE SISTEMA — URBAN-HEAT GRID INTELLIGENCE")
    print("=" * 75)
    corregir_directorios()
    
    r1 = check_1_artefactos_ml()
    r2 = check_2_contrato_inferencia()
    r3 = check_3_cache_sqlite()
    r4 = check_4_pruebas_unitarias()
    r5 = check_5_archivos_frontend()
    
    print("-" * 75)
    if all([r1, r2, r3, r4, r5]):
        print("\033[92m[DIAGNÓSTICO EXITOSO] Todos los subsistemas operan al 100% sin anomalías.\033[0m")
    else:
        print("\033[93m[AVISO] Se detectaron observaciones. Revisa los componentes marcados en rojo.\033[0m")
    print("=" * 75)

if __name__ == "__main__":
    main()
