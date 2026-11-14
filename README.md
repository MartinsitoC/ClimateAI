# URBAN-HEAT GRID INTELLIGENCE

Plataforma geoespacial y motor de inferencia de Machine Learning en cascada para la proyección de anomalías térmicas urbanas ($\Delta T$), Grados-Día de Refrigeración ($\text{CDD}_{18^\circ\text{C}}$ bajo estándar ASHRAE) y clasificación probabilística de estrés en redes eléctricas de distribución hacia el horizonte 2050.

---

## 1. Arquitectura del Sistema

El sistema implementa una arquitectura desacoplada en dos modos de operación (**Modo Demo Académico** sobre 100 metrópolis globales de Berkeley Earth + OWID $\text{CO}_2$, y **Modo Live Global** con geocodificación OpenStreetMap Nominatim + reanálisis histórico ERA5 1940–2026 vía Open-Meteo):

1. **Tarea A — Regresión Térmica Espacial-Antropogénica (`RandomForestRegressor`):**
   - **Entrada:** `[lat, lon, year, month, co2_emissions]`.
   - **Salida:** Temperatura media proyectada ($\hat{T}$) e incertidumbre epistémica inter-árbol ($\pm\sigma$, intervalo de confianza al 95%: $\hat{T} \pm 1.96\sigma$).
   - **Baseline de Comparación:** `Ridge(alpha=1.0)` con escalamiento estándar (`StandardScaler`).
2. **Derivadas Termodinámicas intermedias:**
   - Anomalía térmica respecto a la climatología base: $\Delta T = \hat{T} - T_{\text{baseline}}$.
   - Grados-Día de Refrigeración: $\text{CDD} = \max(0, \hat{T} - 18.0^\circ\text{C})$.
   - Temperatura de Bulbo Húmedo ($T_{wb}$) mediante la formulación empírica de Stull (2011) para carga latente en vivo.
3. **Tarea B — Clasificación Probabilística de Estrés de Red (`GradientBoostingClassifier`):**
   - **Entrada en cascada:** `[AverageTemperature, Delta_T, CDD, lat]`.
   - **Salida:** Probabilidades por clase (`predict_proba`) en 3 estados operativos: `0 = BAJO RIESGO`, `1 = RIESGO MODERADO`, `2 = ALTO RIESGO CRÍTICO`.
   - **Baseline de Comparación:** `LogisticRegression(max_iter=1000)`.
4. **Motor de Inferencia en Tiempo Real (`pipeline/model.py`):**
   - Evaluación vectorizada sobre estructuras Cython (`tree_`) con precalentamiento en memoria (*warm-up*) y latencia estabilizada $< 1\text{ ms}$ por consulta.

---

## 2. Estructura de Carpetas

```text
Proyecto IA/
├── main.py                           # Servidor FastAPI, endpoints REST y exportación de reportes
├── requirements.txt                  # Dependencias de producción y auditoría científica
├── README.md                         # Documentación técnica y guía de ejecución
├── .gitignore                        # Reglas de control de versiones y preservación de artefactos
├── pipeline/
│   ├── config.py                     # Configuración centralizada de rutas, features y umbrales ASHRAE
│   ├── train_all.py                  # Entrenamiento oficial 80/20 (Tarea A + Tarea B) y reporte Train vs Test
│   ├── recalibrate.py                # CLI de recalibración, K-Fold CV y exportación de gráficas a 300 DPI
│   ├── model.py                      # Motor singleton de inferencia en cascada y contrato `predecir()`
│   ├── benchmark_inference.py        # Benchmark de latencia (1,000 inferencias), casos críticos y XAI
│   └── health_check.py               # Diagnóstico integral automatizado de 6 etapas
├── services/
│   ├── geocoding.py                  # Geocodificación global Nominatim con caché SQLite WAL y filtro urbano
│   ├── weather_api.py                # Telemetría climática en vivo (Open-Meteo) y bulbo húmedo de Stull
│   └── historical_climate.py / historical_weather.py  # Series históricas ERA5 (1940–2026)
├── data/
│   ├── raw/                          # Insumos crudos (emisiones anuales de CO2 OWID)
│   └── processed/                    # Artefactos auditables (.joblib, JSON y diagnósticos PNG)
│       ├── rf_temperature_model.joblib
│       ├── gb_grid_risk_model.joblib
│       ├── evaluation_matrix.json
│       ├── model_metadata.json
│       ├── matriz_confusion.png
│       └── curva_residuos.png
├── tests/
│   └── test_models.py                # Suite de pruebas unitarias (integridad, métricas, bordes y SLA < 10 ms)
└── static/
    └── index.html                    # Interfaz geoespacial reactiva (Leaflet + Chart.js + Tailwind CSS)
```

---

## 3. Instalación

```bash
pip install -r requirements.txt
```

---

## 4. Entrenamiento, Recalibración y Auditoría de Modelos

### Entrenamiento Completo (Partición Estricta 80% Train / 20% Test)
Entrena los modelos base y principales de regresión y clasificación, imprime la tabla comparativa `VALIDACIÓN DE GENERALIZACIÓN (TRAIN VS. TEST)` y actualiza los artefactos en `data/processed/`:

```bash
python pipeline/train_all.py
```

### Recalibración Paramétrica y Exportación de Evidencias Gráficas (300 DPI)
Imprime la matriz de confusión en consola y genera `data/processed/matriz_confusion.png` y `data/processed/curva_residuos.png`:

```bash
python pipeline/recalibrate.py --export-plots
```

Opciones adicionales de recalibración por CLI:
```bash
python pipeline/recalibrate.py --split 0.20 --kfold 5 --trees 100 --export-plots
```

---

## 5. Pruebas Unitarias, Diagnóstico Integral y Benchmark

### Diagnóstico Integral de Salud del Sistema (6 Etapas)
Verifica directorios, artefactos `.joblib`, contrato de inferencia `predecir()`, caché SQLite, suite `unittest` e integridad del frontend:

```bash
python pipeline/health_check.py
```

### Suite de Pruebas Unitarias y Benchmark de Latencia / XAI
```bash
python -m unittest tests/test_models.py -v
python pipeline/benchmark_inference.py
```

---

## 6. Arranque del Servidor Web

Inicia el backend FastAPI y la plataforma visual interactiva en `http://127.0.0.1:8080`:

```bash
uvicorn main:app --port 8080
```
