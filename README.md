# URBAN-HEAT GRID INTELLIGENCE

Plataforma geoespacial y motor de inferencia de Machine Learning en cascada para la proyección de anomalías térmicas urbanas ($\Delta T$), Grados-Día de Refrigeración ($\text{CDD}_{18^\circ\text{C}}$ bajo estándar ASHRAE) y clasificación probabilística de estrés en redes eléctricas de distribución hacia el horizonte 2050.

---

## 1. Guía Rápida de Despliegue y Ejecución (Entrega 4 — Liberación del Sistema Funcional)

### Versiones Exactas del Stack Tecnológico
| Capa | Tecnología / Librería | Versión Requerida |
| :--- | :--- | :--- |
| **Lenguaje Base** | Python | `3.13` |
| **Backend API** | FastAPI / Uvicorn | `FastAPI >= 0.110`, `Uvicorn >= 0.28` |
| **Machine Learning** | scikit-learn / Joblib | `scikit-learn 1.9.1`, `joblib >= 1.3` |
| **Procesamiento Científico** | Pandas / NumPy | `Pandas >= 2.2`, `NumPy >= 1.26` |
| **Persistencia Geoespacial** | SQLite3 | `SQLite3 (Modo WAL concurrente)` |
| **Visualización Frontend** | Leaflet.js / Chart.js / Tailwind CSS | `Leaflet.js 1.9.4`, `Chart.js 4.x` |

### Instalación y Configuración de Entorno
1. Instalar las dependencias exactas del proyecto:
   ```bash
   pip install -r requirements.txt
   ```
2. *(Opcional)* Copiar el archivo de variables de entorno por defecto:
   ```bash
   cp .env.example .env
   ```
   > **Nota sobre APIs externas:** Los servicios integrados (*Open-Meteo Forecast API*, *Open-Meteo ERA5 Historical Archive* y *OpenStreetMap Nominatim*) son públicos y abiertos, por lo que **NO requieren API keys**.

### Comandos de Verificación y Arranque Inmediato
1. **Auditoría rápida de integridad E2E (6 subsistemas + certificado QA PNG):**
   ```bash
   python pipeline/health_check.py
   ```
2. **Arranque directo del servidor:**
   ```bash
   python main.py
   ```
   *(También compatible mediante CLI estándar: `uvicorn main:app --port 8080`)*
3. **URL de Acceso Local:**
   Abrir en el navegador: `http://127.0.0.1:8080`

### Acceso, Credenciales y Modos de Operación
- **Acceso abierto sin login:** El sistema está configurado con **acceso directo sin pantalla de autenticación ni credenciales**, permitiendo la evaluación funcional inmediata al abrir `http://127.0.0.1:8080`.
- **Modos de operación integrados (conmutables desde el icono de engranaje `⚙️` o con `Ctrl+K`):**
  1. **Modo Demo Académico (`demo`):** Opera en memoria RAM (`< 1 ms`) sobre el catálogo curado de **100 metrópolis globales (1850–2050)** combinando *Berkeley Earth Surface Temperature* y *OWID $\text{CO}_2$*.
  2. **Modo Live Global (`live`):** Permite buscar cualquier ciudad, municipio o pueblo del mundo (*Nominatim*) o hacer clic en **cualquier coordenada del planeta** (*Modo Inspección Puntual* `🎯`), combinando telemetría en vivo, series históricas *ERA5 (1940–2025)* e inferencia ML en cascada (*2026–2050*).

---

## 2. Arquitectura del Motor de Machine Learning en Cascada

1. **Tarea A — Regresión Térmica Espacial-Antropogénica (`RandomForestRegressor`):**
   - **Entrada:** `[lat, lon, year, month, co2_emissions]`.
   - **Salida:** Temperatura media proyectada ($\hat{T}$) e incertidumbre epistémica inter-árbol ($\pm\sigma$, intervalo de confianza al 95%: $\hat{T} \pm 1.96\sigma$).
   - **Desempeño validado (80/20 Hold-out):** $R^2_{\text{Test}} = 0.9853$, $\text{RMSE}_{\text{Test}} = 1.1844^\circ\text{C}$, $\Delta R^2 = +0.0040$ (sin *overfitting*) frente al baseline `Ridge` ($R^2 = 0.3005$).
2. **Derivadas Termodinámicas Intermedias:**
   - Anomalía térmica respecto a la climatología base: $\Delta T = \hat{T} - T_{\text{baseline}}$.
   - Grados-Día de Refrigeración: $\text{CDD} = \max(0, \hat{T} - 18.0^\circ\text{C})$.
   - Temperatura de Bulbo Húmedo ($T_{wb}$) mediante la formulación empírica de Stull (2011) para carga latente en vivo.
3. **Tarea B — Clasificación Probabilística de Estrés de Red (`GradientBoostingClassifier`):**
   - **Entrada en cascada:** `[AverageTemperature, Delta_T, CDD, lat]`.
   - **Salida:** Probabilidades por clase (`predict_proba`) en 3 estados operativos: `0 = BAJO RIESGO`, `1 = RIESGO MODERADO`, `2 = ALTO RIESGO CRÍTICO` ($F1\text{-Macro}_{\text{Test}} = 1.0000$).
4. **Motor de Inferencia en Tiempo Real (`pipeline/model.py`):**
   - Evaluación vectorizada sobre estructuras Cython (`tree_`) con precalentamiento en memoria (*warm-up*) y latencia estabilizada $P_{50} \approx 2.1\text{ ms}$ ($P_{95} < 3.5\text{ ms}$).

---

## 3. Estructura de Carpetas

```text
Proyecto IA/
├── main.py                           # Servidor FastAPI, endpoints REST, exportación PNG/JSON y arranque directo
├── .env.example                      # Plantilla de variables de entorno (sin necesidad de API keys externas)
├── requirements.txt                  # Dependencias de producción y auditoría científica
├── README.md                         # Documentación técnica y guía rápida de despliegue
├── .gitignore                        # Reglas de control de versiones y preservación de artefactos
├── pipeline/
│   ├── config.py                     # Configuración centralizada de rutas, features y umbrales ASHRAE
│   ├── train_all.py                  # Entrenamiento oficial 80/20 (Tarea A + Tarea B) y reporte Train vs Test
│   ├── recalibrate.py                # CLI de recalibración, K-Fold CV y exportación de gráficas a 300 DPI
│   ├── model.py                      # Motor singleton de inferencia en cascada y contrato `predecir()`
│   ├── visual_auditor.py             # Generador de evidencias académicas y fichas ejecutivas PNG (300 DPI)
│   ├── benchmark_inference.py        # Benchmark de latencia (1,000 inferencias), casos críticos y XAI
│   └── health_check.py               # Diagnóstico integral E2E de 6 etapas y emisor de certificado QA PNG
├── services/
│   ├── geocoding.py                  # Geocodificación global Nominatim con caché SQLite WAL y semáforo asíncrono
│   ├── weather_api.py                # Telemetría climática en vivo (Open-Meteo) y bulbo húmedo de Stull
│   └── historical_climate.py / historical_weather.py  # Series históricas ERA5 (1940–2026) no bloqueantes
├── data/
│   ├── raw/                          # Insumos crudos (emisiones anuales de CO2 OWID)
│   └── processed/                    # Artefactos auditables (.joblib, JSON y diagnósticos PNG)
│       ├── rf_temperature_model.joblib
│       ├── gb_grid_risk_model.joblib
│       ├── evaluation_matrix.json
│       ├── model_metadata.json
│       ├── matriz_confusion.png
│       └── curva_residuos.png
├── evidencias_finales/               # Evidencias gráficas de alta resolución (300 DPI, estilo Dark-Slate)
│   ├── 01_auditoria_train_test_confusion.png  # Comparativa Train vs Test, Matriz 3x3 y XAI
│   ├── 02_ficha_Cancun_2045.png               # Ficha ejecutiva de diagnóstico puntual por CLI
│   ├── 03_certificado_qa_e2e.png              # Certificado de salud E2E (6 subsistemas + P50/P95/P99)
│   └── export_{ciudad}_{anio}.png             # Fichas exportadas desde la interfaz web (/api/export/report-image)
├── tests/
│   └── test_models.py                # Suite unittest (artefactos ML, casos límite, SLA < 10 ms y endpoints FastAPI)
└── static/
    └── index.html                    # Interfaz geoespacial reactiva (Leaflet + Chart.js + Tailwind CSS + Ctrl+K)
```

---

## 4. Comandos de Entrenamiento, Recalibración y Auditoría Científica

### Entrenamiento Completo (Partición Estricta 80% Train / 20% Test)
```bash
python pipeline/train_all.py
```

### Recalibración Paramétrica y Exportación de Gráficas Base (300 DPI)
```bash
python pipeline/recalibrate.py --export-plots
```

### Generación de Evidencias Visuales Ejecutivas (`evidencias_finales/`)
```bash
python pipeline/visual_auditor.py --audit-ml
python pipeline/visual_auditor.py --lugar "Cancun" --lat 21.16 --lon -86.85 --year 2045
```

### Suite de Pruebas Unitarias + E2E y Benchmark de Latencia
```bash
python -m unittest tests/test_models.py -v
python pipeline/benchmark_inference.py
```
