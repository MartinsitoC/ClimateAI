import os
import sys
import json
import time
import asyncio
import urllib.request
import urllib.parse
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# Anclaje absoluto de rutas al directorio de main.py (independiente del CWD de la terminal)
BASE_DIR: Path = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pipeline.model import UrbanHeatInferenceEngine, PredictionInput, PredictionOutput
from services.geocoding import geocoding_service
from services.weather_api import weather_service
from services.historical_weather import obtener_historial_real_global

# Rutas de archivos normalizadas con pathlib.Path
CSV_CANDIDATES = [
    BASE_DIR / "Temperaturas" / "Global Land Temperatures" / "GlobalLandTemperaturesByMajorCity.csv",
    BASE_DIR / "data" / "raw" / "GlobalLandTemperaturesByMajorCity.csv",
    BASE_DIR / "GlobalLandTemperaturesByMajorCity.csv",
]
CSV_FILE: Path = next((p for p in CSV_CANDIDATES if p.is_file()), CSV_CANDIDATES[0])
GLOBAL_CSV_FILE: Path = BASE_DIR / "Temperaturas" / "Global Land Temperatures" / "GlobalTemperatures.csv"
CACHE_FILE: Path = BASE_DIR / "open_meteo_cache.json"
STATIC_DIR: Path = BASE_DIR / "static"
IMG_DIR: Path = BASE_DIR / "Img"

CURRENT_YEAR = datetime.now().year  # Año actual dinámico (ej. 2026)
HORIZONTE_PROYECCION = 2050         # Cobertura de proyección extendida hasta 2050
BASE_COMFORT_CDD = 18.0             # Estándar ASHRAE (18°C)

# Catálogo jerarquizado de metrópolis globales (Tier 1: Nodos Principales en Zoom <= 4)
TIER_1_CITIES = {
    "Mexico", "New York", "London", "Tokyo", "Cairo", "Baghdad",
    "Rio De Janeiro", "Paris", "Moscow", "Beijing", "Sydney",
    "Johannesburg", "Mumbai"
}
# Alias históricos en el dataset Berkeley Earth (ej. Peking -> Beijing, Bombay -> Mumbai, Cape Town -> Johannesburg)
TIER_1_ALIASES = TIER_1_CITIES | {"Peking", "Bombay", "Cape Town"}

# Rampa espectral GFS de alta saturación compartida entre Backend (Agente) y Frontend (Canvas)
GFS_SPECTRUM_STOPS = [
    (-25.0, (168, 85, 247)),  # Magenta glacial (#a855f7)
    (-10.0, (99, 102, 241)),  # Índigo polar (#6366f1)
    (0.0,   (14, 165, 233)),  # Azul eléctrico (#0ea5e9)
    (8.0,   (20, 184, 166)),  # Verde azulado (#14b8a6)
    (16.0,  (34, 197, 94)),   # Verde esmeralda (#22c55e)
    (22.0,  (234, 179, 8)),   # Amarillo oro (#eab308)
    (28.0,  (249, 115, 22)),  # Naranja fuego (#f97316)
    (34.0,  (239, 68, 68)),   # Rojo carmesí (#ef4444)
    (42.0,  (136, 19, 55))    # Borgoña térmico (#881337)
]


def obtener_tier_ciudad(city: str) -> int:
    """Retorna 1 si la metrópoli pertenece al catálogo global Tier 1, o 2 para el resto."""
    return 1 if city in TIER_1_ALIASES else 2


def asignar_color_hex_termico(temp: float) -> str:
    """Interpola suavemente sobre GFS_SPECTRUM_STOPS y retorna el color hexadecimal (#rrggbb)."""
    if temp is None or np.isnan(temp):
        return "#64748b"
    t = float(temp)
    if t <= GFS_SPECTRUM_STOPS[0][0]:
        r, g, b = GFS_SPECTRUM_STOPS[0][1]
        return f"#{r:02x}{g:02x}{b:02x}"
    if t >= GFS_SPECTRUM_STOPS[-1][0]:
        r, g, b = GFS_SPECTRUM_STOPS[-1][1]
        return f"#{r:02x}{g:02x}{b:02x}"
    for i in range(len(GFS_SPECTRUM_STOPS) - 1):
        t0, c0 = GFS_SPECTRUM_STOPS[i]
        t1, c1 = GFS_SPECTRUM_STOPS[i + 1]
        if t0 <= t <= t1:
            lin = (t - t0) / (t1 - t0)
            f = lin * lin * (3.0 - 2.0 * lin)
            r = int(round(c0[0] + (c1[0] - c0[0]) * f))
            g = int(round(c0[1] + (c1[1] - c0[1]) * f))
            b = int(round(c0[2] + (c1[2] - c0[2]) * f))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#0ea5e9"

# ==============================================================================
# ESTRUCTURA GLOBAL EN MEMORIA RAM (PRE-COMPUTADA EN LIFESPAN)
# ==============================================================================
GLOBAL_CACHE: Dict[str, Any] = {
    "series_temporales": {},          # Series históricas y proyectadas (1850-2050) de anomalías, CDD y timeline anual
    "ciudades": {},                   # { "Ciudad": { "coords": [lat, lon], "cdd_historico": [...], "metricas": {...}, ... } }
    "ciudades_lower": {},             # Mapa O(1) case-insensitive -> clave canónica
    "matriz_interpolacion_base": {},  # Matriz base pre-computada para el mapa rasterizado de calor
    "config": {},                     # Configuración general y catálogo de ciudades
    "index_html": "",                 # HTML principal en RAM
    "bytes": {                        # JSON pre-serializados en bytes UTF-8 para latencia < 1 ms
        "climate_data": b"{}",
        "config": b"{}",
        "years": {},
        "cities": {}
    }
}
METADATA_CIUDADES: Dict[str, Dict[str, Any]] = {}
ANOS_HISTORICOS: list = []
PROYECCION_ANOS: list = list(range(CURRENT_YEAR + 1, HORIZONTE_PROYECCION + 1))
GLOBAL_HISTORIAL: Dict[int, float] = {}
GLOBAL_MODEL: Dict[str, Any] = {}


def parse_coord(coord_str: str) -> float:
    """Convierte formato '19.25N' o '99.10W' a float positivo/negativo."""
    if not isinstance(coord_str, str):
        return 0.0
    coord_str = coord_str.strip()
    direction = coord_str[-1].upper()
    val = float(coord_str[:-1])
    return -val if direction in ['S', 'W'] else val


def calcular_cdd(temperatura: float) -> float:
    """Calcula Cooling Degree Days (ASHRAE base 18°C)."""
    return round(max(0.0, float(temperatura) - BASE_COMFORT_CDD), 2)


def calcular_peso_normalizado(temp: float) -> float:
    """Normaliza la temperatura al rango [0.1, 1.0] para la capa térmica."""
    if temp < 10.0:
        return round(max(0.1, min(0.35, 0.1 + (temp + 5.0) / 15.0 * 0.25)), 2)
    elif temp <= 26.0:
        return round(0.36 + (temp - 10.0) / 16.0 * 0.32, 2)
    else:
        return round(min(1.0, 0.70 + (temp - 26.0) / 14.0 * 0.30), 2)


def clasificar_riesgo_energetico(calentamiento_decada: float, pct_aumento_cdd: float) -> dict:
    """
    Segmentación de vulnerabilidad mediante semáforo universal:
    - ALTO RIESGO (Rojo): > +0.25°C/década o > +25% en proyección de CDD.
    - RIESGO MODERADO (Amarillo): +10% a +25% en CDD.
    - BAJO RIESGO / ESTABLE (Verde): < +10% en CDD.
    """
    if calentamiento_decada > 0.25 or pct_aumento_cdd > 25.0:
        return {
            "nivel": "ALTO",
            "etiqueta": "ALTO RIESGO CRÍTICO",
            "color": "red",
            "hex": "#ef4444",
            "estres_red": "Crítico (>85% saturación en picos térmicos)",
            "descripcion": "Riesgo inminente de sobrecarga en transformadores y contingencias por demanda simultánea de climatización."
        }
    elif pct_aumento_cdd >= 10.0 or calentamiento_decada > 0.15:
        return {
            "nivel": "MODERADO",
            "etiqueta": "RIESGO MODERADO",
            "color": "yellow",
            "hex": "#f59e0b",
            "estres_red": "Moderado-Alto (60%-85% capacidad nominal)",
            "descripcion": "Incremento sostenido que requiere refuerzo preventivo en subestaciones y auditorías de eficiencia."
        }
    else:
        return {
            "nivel": "BAJO",
            "etiqueta": "ESTABLE / BAJO RIESGO",
            "color": "green",
            "hex": "#10b981",
            "estres_red": "Estable (<60% capacidad)",
            "descripcion": "Variación térmica dentro de los márgenes de resiliencia del sistema de distribución."
        }


def clasificar_perfil_geoclimatico(city: str, country: str, lat: float, lon: float, temp_media: float) -> str:
    """
    Clasifica la urbe en uno de tres perfiles geoclimáticos:
    - DESERTICO: Alta radiación solar, aridez extrema o latitudes secas.
    - COSTERO_TROPICAL: Zonas costeras/tropicales de alta humedad o islas.
    - TEMPLADO_BOREAL: Europa Central/Norte, Norteamérica septentrional, zonas templadas frías.
    """
    city_l = city.lower()
    country_l = country.lower()
    lat_abs = abs(lat)

    desert_keywords = [
        "cairo", "baghdad", "riyadh", "dubai", "alexandria", "damascus", "amman", "jeddah",
        "kuwait", "doha", "casablanca", "tunis", "algiers", "hermosillo", "mexicali", "monterrey",
        "phoenix", "las vegas", "karachi", "lahore", "delhi", "jaipur", "ahmadabad"
    ]
    desert_countries = ["egypt", "iraq", "saudi arabia", "iran", "morocco", "algeria", "tunisia", "united arab emirates", "kuwait", "qatar"]

    if any(k in city_l for k in desert_keywords) or any(c in country_l for c in desert_countries):
        return "DESERTICO"

    if lat_abs >= 40.0:
        return "TEMPLADO_BOREAL"

    cold_transition_countries = ["germany", "united kingdom", "france", "russia", "poland", "canada", "ukraine", "japan", "sweden", "norway"]
    if any(c in country_l for c in cold_transition_countries):
        return "TEMPLADO_BOREAL"

    if lat_abs <= 28.0 or temp_media >= 22.0:
        return "COSTERO_TROPICAL"

    return "TEMPLADO_BOREAL"


def generar_medidas_prescriptivas(riesgo_info: dict, delta_temp: float, pct_cdd: float, perfil_clima: str = "COSTERO_TROPICAL", city_name: str = "") -> list:
    """
    Genera el catálogo de 4 medidas prescriptivas diferenciadas y contextuales
    según el perfil geoclimático, el semáforo de riesgo y los valores dinámicos calculados.
    """
    if perfil_clima == "DESERTICO":
        return [
            {
                "categoria": "1. Infraestructura Eléctrica",
                "titulo": f"Mitigación de Sobrecarga Térmica en Transformadores ({city_name})",
                "descripcion": f"Instalación de blindajes contra radiación solar directa y refrigeración forzada (aceite/aire ONAF) en subestaciones críticas para soportar el incremento proyectado de +{pct_cdd}% en demanda simultánea de aire acondicionado continuo.",
                "icono": "⚡"
            },
            {
                "categoria": "2. Normativa y Construcción Urbana",
                "titulo": "Pavimentos y Techos Reflectivos contra Isla de Calor",
                "descripcion": f"Modificación al reglamento urbano para exigir albedo superficial mínimo de 0.70 en techumbres (Cool Roofs) y pavimentos permeables/reflectivos para amortiguar el aumento térmico estimado de +{delta_temp}°C.",
                "icono": "🏜️"
            },
            {
                "categoria": "3. Eficiencia Energética",
                "titulo": "Sustitución HVAC Inverter para Climas Extremos",
                "descripcion": "Incentivos fiscales para migrar a equipos con compresores de velocidad variable certificados para operar hasta 52°C sin caída de eficiencia, gestionando la curva de demanda en horas pico de insolación.",
                "icono": "❄️"
            },
            {
                "categoria": "4. Alerta Temprana & Resiliencia",
                "titulo": "Respuesta Dinámica a la Demanda & Microredes Solares",
                "descripcion": "Acoplamiento de generación solar fotovoltaica con almacenamiento BESS en nodos industriales/residenciales para sincronizar el pico de generación solar con el pico de demanda de enfriamiento.",
                "icono": "☀️"
            }
        ]
    elif perfil_clima == "COSTERO_TROPICAL":
        return [
            {
                "categoria": "1. Infraestructura Eléctrica",
                "titulo": "Protección Anticorrosiva & Resiliencia de Subestaciones Costeras",
                "descripcion": f"Encapsulamiento blindado (tecnología GIS) y recubrimientos anti-salinos en líneas y transformadores para evitar fallas dieléctricas ante la combinación de alta humedad costera y sobrecarga de +{pct_cdd}% en climatización.",
                "icono": "⚡"
            },
            {
                "categoria": "2. Normativa y Construcción Urbana",
                "titulo": "Envolventes con Control de Humedad & Ventilación Pasiva",
                "descripcion": f"Requisitos obligatorios de barreras de vapor continuas, aislamiento térmico ASHRAE y diseño de ventilación cruzada para mitigar la carga latente provocada por el alza térmica de +{delta_temp}°C.",
                "icono": "🌊"
            },
            {
                "categoria": "3. Eficiencia Energética",
                "titulo": "Climatización con Deshumidificación Dedicada (DOAS)",
                "descripcion": "Adopción prioritaria de sistemas de aire exterior dedicados con recuperación entálpica para desacoplar el control de humedad del enfriamiento sensible, reduciendo hasta un 30% el consumo eléctrico.",
                "icono": "💧"
            },
            {
                "categoria": "4. Alerta Temprana & Resiliencia",
                "titulo": "Microredes Híbridas para Eventos Meteorológicos Extremos",
                "descripcion": "Protocolo de respaldo de energía descentralizada y desconexión controlada en zonas vulnerables durante tormentas tropicales u olas de calor simultáneas.",
                "icono": "🚨"
            }
        ]
    else:  # TEMPLADO_BOREAL
        return [
            {
                "categoria": "1. Infraestructura Eléctrica",
                "titulo": "Reconversión y Refuerzo de Redes para Picos Estivales",
                "descripcion": f"Modernización de transformadores envejecidos —diseñados históricamente sólo para calefacción invernal— para soportar la nueva demanda masiva estival de climatización proyectada en +{pct_cdd}%.",
                "icono": "⚡"
            },
            {
                "categoria": "2. Normativa y Construcción Urbana",
                "titulo": "Transición de Calefacción a Redes de Distrito de Frío (District Cooling)",
                "descripcion": f"Despliegue de canalizaciones de refrigeración centralizada por agua de lago/río y actualización de códigos de edificación ante el calentamiento estival sostenido de +{delta_temp}°C.",
                "icono": "🏢"
            },
            {
                "categoria": "3. Eficiencia Energética",
                "titulo": "Bombas de Calor Reversibles de Alta Eficiencia",
                "descripcion": "Reemplazo integral de calderas de gas por bombas de calor aerotérmicas y geotérmicas reversibles (SCOP/SEER > 4.5) con control horario automatizado.",
                "icono": "🔄"
            },
            {
                "categoria": "4. Alerta Temprana & Resiliencia",
                "titulo": "Protocolo de Emergencia para Ciudades en Transición Climática",
                "descripcion": "Monitoreo preventivo y habilitación de refugios urbanos de enfriamiento pasivo ante episodios inéditos de olas de calor en poblaciones vulnerables no aclimatadas.",
                "icono": "🛡️"
            }
        ]


def cargar_cache_meteo() -> dict:
    if CACHE_FILE.is_file():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def calcular_baseline_anomalia(historial: dict) -> float:
    """Línea base climatológica 1951-1980 (fallback: media completa) para anomalías térmicas."""
    vals = [v for y, v in historial.items() if 1951 <= y <= 1980]
    if not vals:
        vals = list(historial.values())
    return round(float(np.mean(vals)), 2) if vals else 0.0


def construir_payload_ciudad(city_label: str, meta: dict, historial_dict: dict) -> dict:
    """Construye y estructura el payload completo de una ciudad o del consolidado Global para GLOBAL_CACHE en RAM."""
    base = calcular_baseline_anomalia(historial_dict)
    serie_historica = [
        {
            "year": y,
            "temperatura": historial_dict[y],
            "anomalia": round(historial_dict[y] - base, 2),
            "cdd": calcular_cdd(historial_dict[y])
        }
        for y in sorted(historial_dict.keys())
        if y <= CURRENT_YEAR
    ]

    slope = meta["slope"]
    intercept = meta["intercept"]
    serie_proyeccion = []
    for y in PROYECCION_ANOS:
        pred_val = round(intercept + slope * y, 2)
        diff_years = y - CURRENT_YEAR
        serie_proyeccion.append({
            "year": y,
            "prediccion": pred_val,
            "anomalia_proy": round(pred_val - base, 2),
            "cdd_proy": calcular_cdd(pred_val),
            "lim_sup": round(pred_val + (0.04 * diff_years), 2),
            "lim_inf": round(pred_val - (0.04 * diff_years), 2)
        })

    perfil_clima = clasificar_perfil_geoclimatico(
        city_label,
        meta["country"],
        meta["lat"],
        meta["lon"],
        meta["media_hist"]
    )
    medidas = generar_medidas_prescriptivas(
        meta["riesgo"],
        meta["delta_temp_2040"],
        meta["pct_aumento_cdd"],
        perfil_clima,
        city_label if city_label else "Global"
    )

    metricas = {
        "calentamiento_decada": meta["calentamiento_decada"],
        "media_historica": meta["media_hist"],
        "baseline_anomalia": base,
        "cdd_hist_base": meta["cdd_hist_base"],
        "temp_actual": meta["temp_actual"],
        "temp_proy_2040": meta["temp_proy_2040"],
        "delta_temp_2040": meta["delta_temp_2040"],
        "cdd_proy_2040": meta["cdd_proy_2040"],
        "pct_aumento_cdd": meta["pct_aumento_cdd"],
        "riesgo": meta["riesgo"]
    }

    return {
        "city": city_label,
        "country": meta["country"],
        "coords": [meta["lat"], meta["lon"]],
        "lat": meta["lat"],
        "lon": meta["lon"],
        "perfil_clima": perfil_clima,
        "current_year": CURRENT_YEAR,
        "horizonte": HORIZONTE_PROYECCION,
        "cdd_historico": [item["cdd"] for item in serie_historica],
        "metricas": metricas,
        "calentamiento_decada": meta["calentamiento_decada"],
        "media_historica": meta["media_hist"],
        "cdd_hist_base": meta["cdd_hist_base"],
        "temp_actual": meta["temp_actual"],
        "temp_proy_2040": meta["temp_proy_2040"],
        "delta_temp_2040": meta["delta_temp_2040"],
        "cdd_proy_2040": meta["cdd_proy_2040"],
        "pct_aumento_cdd": meta["pct_aumento_cdd"],
        "riesgo": meta["riesgo"],
        "medidas_prescriptivas": medidas,
        "historico": serie_historica,
        "proyeccion": serie_proyeccion
    }


def _precomputar_matriz_interpolacion_base(grid_h: int = 90, grid_w: int = 180) -> dict:
    """Pre-computa en RAM la grilla geoclimática base para el mapa rasterizado de calor (IDW)."""
    lats = np.linspace(85.0, -85.0, grid_h, dtype=np.float32)
    lons = np.linspace(-180.0, 180.0, grid_w, dtype=np.float32)
    lat_rad = np.deg2rad(lats)
    temp_base_lat = np.round(-16.0 + 46.0 * np.power(np.cos(lat_rad), 1.35), 2)
    return {
        "grid_shape": [grid_h, grid_w],
        "lat_bounds": [85.0, -85.0],
        "lon_bounds": [-180.0, 180.0],
        "lats": np.round(lats, 2).tolist(),
        "lons": np.round(lons, 2).tolist(),
        "temp_base_lat": temp_base_lat.tolist(),
        "idw_params": {"power": 2.0, "radius_sq": 1050.0, "smoothing": 8.0, "blend": 0.90}
    }


def inicializar_dataset_en_memoria():
    """
    Carga única en memoria RAM al arranque del servidor (lifespan):
    1. Lee el CSV de ciudades una sola vez con extracción vectorizada de año.
    2. Pre-computa en GLOBAL_CACHE las series 1850-2050, el índice por ciudad y la matriz base de interpolación.
    3. Pre-serializa en bytes UTF-8 todas las respuestas para garantizar latencia < 1 ms (cero I/O en endpoints).
    """
    global METADATA_CIUDADES, ANOS_HISTORICOS, GLOBAL_HISTORIAL, GLOBAL_MODEL, PROYECCION_ANOS, GLOBAL_CACHE

    t0 = time.perf_counter()
    print("[LIFESPAN] Cargando y pre-computando datasets climáticos en GLOBAL_CACHE (RAM)...")

    if not CSV_FILE.is_file():
        raise FileNotFoundError(f"No se encontró el dataset en: {CSV_FILE}")

    open_meteo_cache = cargar_cache_meteo()

    # Lectura única del dataset desde disco
    df = pd.read_csv(
        CSV_FILE,
        usecols=['dt', 'AverageTemperature', 'City', 'Country', 'Latitude', 'Longitude']
    )
    df = df.dropna(subset=['AverageTemperature', 'dt'])
    df['year'] = df['dt'].str.slice(0, 4).astype(int)
    df = df[(df['year'] >= 1850) & (df['year'] <= 2013)]

    df_annual = (
        df.groupby(['City', 'Country', 'Latitude', 'Longitude', 'year'], as_index=False)['AverageTemperature']
        .mean()
    )
    df_annual['AverageTemperature'] = df_annual['AverageTemperature'].round(2)

    ciudades_coords = df[['City', 'Country', 'Latitude', 'Longitude']].drop_duplicates('City').copy()
    ciudades_coords['lat'] = ciudades_coords['Latitude'].apply(parse_coord)
    ciudades_coords['lon'] = ciudades_coords['Longitude'].apply(parse_coord)
    coords_dict = ciudades_coords.set_index('City')[['Country', 'lat', 'lon']].to_dict('index')

    meta_ciudades = {}
    todos_los_anos_reales = list(range(1850, CURRENT_YEAR + 1))

    for city, group in df_annual.groupby('City', sort=False):
        coord_info = coords_dict.get(city, {'Country': 'Unknown', 'lat': 0.0, 'lon': 0.0})
        lat = float(coord_info['lat'])
        lon = float(coord_info['lon'])
        group_sorted = group.sort_values('year')
        historial = dict(zip(group_sorted['year'].astype(int), group_sorted['AverageTemperature'].astype(float)))

        cache_key = f"{city}_{round(lat, 2)}_{round(lon, 2)}"
        if cache_key in open_meteo_cache:
            historial.update({int(k): float(v) for k, v in open_meteo_cache[cache_key].items()})

        x_prev = np.array([y for y in historial.keys() if y >= 1950], dtype=float)
        y_prev = np.array([historial[int(y)] for y in x_prev], dtype=float)
        if len(x_prev) >= 5:
            fit_prev = np.polyfit(x_prev, y_prev, deg=1)
            for y_fill in range(2014, CURRENT_YEAR + 1):
                if y_fill not in historial:
                    historial[y_fill] = round(float(fit_prev[1] + fit_prev[0] * y_fill), 2)

        x_all = np.array(sorted(historial.keys()), dtype=float)
        y_all = np.array([historial[int(y)] for y in x_all], dtype=float)
        mask_recent = x_all >= 1950
        coeffs = np.polyfit(x_all[mask_recent], y_all[mask_recent], deg=1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        calentamiento_decada = round(slope * 10, 3)

        temps_hist = [historial[int(y)] for y in x_all if y <= CURRENT_YEAR]
        media_hist = round(float(np.mean(temps_hist)), 2) if temps_hist else 0.0
        cdd_hist_base = round(float(np.mean([calcular_cdd(t) for t in temps_hist])), 2)

        temp_proy_2040 = round(float(intercept + slope * HORIZONTE_PROYECCION), 2)
        temp_actual = historial.get(CURRENT_YEAR, round(float(intercept + slope * CURRENT_YEAR), 2))
        delta_temp_2040 = round(temp_proy_2040 - temp_actual, 2)
        cdd_proy_2040 = calcular_cdd(temp_proy_2040)

        if cdd_hist_base > 0.01:
            pct_aumento_cdd = round(((cdd_proy_2040 - cdd_hist_base) / cdd_hist_base) * 100.0, 1)
        else:
            pct_aumento_cdd = round((cdd_proy_2040 * 10.0), 1)

        riesgo_info = clasificar_riesgo_energetico(calentamiento_decada, pct_aumento_cdd)

        tier_ciudad = obtener_tier_ciudad(city)

        meta_ciudades[city] = {
            "country": coord_info['Country'],
            "tier": tier_ciudad,
            "lat": lat,
            "lon": lon,
            "historial": historial,
            "slope": slope,
            "calentamiento_decada": calentamiento_decada,
            "intercept": intercept,
            "media_hist": media_hist,
            "baseline_anom": calcular_baseline_anomalia(historial),
            "cdd_hist_base": cdd_hist_base,
            "temp_actual": temp_actual,
            "temp_proy_2040": temp_proy_2040,
            "delta_temp_2040": delta_temp_2040,
            "cdd_proy_2040": cdd_proy_2040,
            "pct_aumento_cdd": pct_aumento_cdd,
            "riesgo": riesgo_info,
            "riesgo_compacto": {
                "nivel": riesgo_info["nivel"],
                "etiqueta": riesgo_info["etiqueta"],
                "hex": riesgo_info["hex"]
            }
        }

    METADATA_CIUDADES = meta_ciudades
    ANOS_HISTORICOS = todos_los_anos_reales
    PROYECCION_ANOS = list(range(CURRENT_YEAR + 1, HORIZONTE_PROYECCION + 1))

    # Consolidado global
    global_historial = {}
    for y in ANOS_HISTORICOS:
        vals = [data["historial"].get(y) for data in METADATA_CIUDADES.values() if data["historial"].get(y) is not None]
        if vals:
            global_historial[y] = round(float(np.mean(vals)), 2)

    GLOBAL_HISTORIAL = global_historial

    gx = np.array([y for y in GLOBAL_HISTORIAL.keys() if y >= 1950], dtype=float)
    gy = np.array([GLOBAL_HISTORIAL[int(y)] for y in gx], dtype=float)
    gcoeffs = np.polyfit(gx, gy, deg=1)
    g_slope = float(gcoeffs[0])
    g_intercept = float(gcoeffs[1])
    g_cal_dec = round(g_slope * 10, 3)
    g_temp_actual = GLOBAL_HISTORIAL.get(CURRENT_YEAR, round(g_intercept + g_slope * CURRENT_YEAR, 2))
    g_temp_2040 = round(g_intercept + g_slope * HORIZONTE_PROYECCION, 2)
    g_cdd_base = round(float(np.mean([calcular_cdd(t) for t in GLOBAL_HISTORIAL.values()])), 2)
    g_cdd_2040 = calcular_cdd(g_temp_2040)
    g_pct_cdd = round(((g_cdd_2040 - g_cdd_base) / g_cdd_base) * 100.0, 1) if g_cdd_base > 0.01 else 15.0

    GLOBAL_MODEL = {
        "country": "Global",
        "lat": 20.0,
        "lon": 0.0,
        "slope": g_slope,
        "intercept": g_intercept,
        "calentamiento_decada": g_cal_dec,
        "media_hist": round(float(np.mean(list(GLOBAL_HISTORIAL.values()))), 2),
        "cdd_hist_base": g_cdd_base,
        "temp_actual": g_temp_actual,
        "temp_proy_2040": g_temp_2040,
        "delta_temp_2040": round(g_temp_2040 - g_temp_actual, 2),
        "cdd_proy_2040": g_cdd_2040,
        "pct_aumento_cdd": g_pct_cdd,
        "riesgo": clasificar_riesgo_energetico(g_cal_dec, g_pct_cdd)
    }

    # 1. Pre-computar matriz de interpolación base para el mapa rasterizado de calor
    matriz_base = _precomputar_matriz_interpolacion_base()
    GLOBAL_CACHE["matriz_interpolacion_base"] = matriz_base

    # 2. Pre-cachear configuración (/api/config)
    ciudades_lista = [
        {
            "city": c,
            "country": data["country"],
            "tier": data["tier"],
            "lat": data["lat"],
            "lon": data["lon"],
            "riesgo": data["riesgo"]
        }
        for c, data in sorted(METADATA_CIUDADES.items())
    ]
    config_payload = {
        "current_year": CURRENT_YEAR,
        "horizonte_proyeccion": HORIZONTE_PROYECCION,
        "anos_historicos": ANOS_HISTORICOS,
        "min_ano": min(ANOS_HISTORICOS),
        "max_ano_hist": CURRENT_YEAR,
        "anos_proyeccion": PROYECCION_ANOS,
        "max_ano_proy": HORIZONTE_PROYECCION,
        "ciudades": ciudades_lista
    }
    GLOBAL_CACHE["config"] = config_payload
    GLOBAL_CACHE["bytes"]["config"] = json.dumps(config_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    # 3. Pre-computar índice rápido por ciudad en GLOBAL_CACHE["ciudades"]
    GLOBAL_CACHE["ciudades"].clear()
    GLOBAL_CACHE["ciudades_lower"].clear()
    GLOBAL_CACHE["bytes"]["cities"].clear()

    global_payload = construir_payload_ciudad("", GLOBAL_MODEL, GLOBAL_HISTORIAL)
    global_bytes = json.dumps(global_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    GLOBAL_CACHE["ciudades"][""] = global_payload
    GLOBAL_CACHE["ciudades"]["Global"] = global_payload
    GLOBAL_CACHE["ciudades_lower"][""] = ""
    GLOBAL_CACHE["ciudades_lower"]["global"] = "Global"
    GLOBAL_CACHE["bytes"]["cities"][""] = global_bytes
    GLOBAL_CACHE["bytes"]["cities"]["Global"] = global_bytes

    for city, meta in METADATA_CIUDADES.items():
        city_payload = construir_payload_ciudad(city, meta, meta["historial"])
        city_bytes = json.dumps(city_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        GLOBAL_CACHE["ciudades"][city] = city_payload
        GLOBAL_CACHE["ciudades_lower"][city.lower()] = city
        GLOBAL_CACHE["bytes"]["cities"][city] = city_bytes

    # 4. Pre-computar series temporales (1850 - 2050) y timeline térmico en GLOBAL_CACHE["series_temporales"]
    timeline_dict = {}
    GLOBAL_CACHE["bytes"]["years"].clear()
    rango_completo = ANOS_HISTORICOS + PROYECCION_ANOS

    for y in rango_completo:
        puntos_ano = []
        es_proyeccion = y > CURRENT_YEAR
        temps_y = []
        cdds_y = []

        for city, data in METADATA_CIUDADES.items():
            if not es_proyeccion:
                t = data["historial"].get(y)
                if t is None:
                    t = round(data["intercept"] + data["slope"] * y, 2)
            else:
                t = round(data["intercept"] + data["slope"] * y, 2)

            cdd_val = calcular_cdd(t)
            peso = calcular_peso_normalizado(t)
            temps_y.append(t)
            cdds_y.append(cdd_val)

            puntos_ano.append({
                "lat": data["lat"],
                "lon": data["lon"],
                "temperatura": t,
                "temp": t,
                "anomalia": round(t - data["baseline_anom"], 2),
                "cdd": cdd_val,
                "intensidad": peso,
                "city": city,
                "country": data["country"],
                "tier": data["tier"],
                "color": asignar_color_hex_termico(t),
                "riesgo": data["riesgo_compacto"]
            })

        timeline_dict[str(y)] = puntos_ano

        kpis_y = {
            "year": y,
            "current_year": CURRENT_YEAR,
            "es_proyeccion": es_proyeccion,
            "total_ciudades": len(puntos_ano),
            "temp_promedio": round(float(np.mean(temps_y)), 2) if temps_y else 0.0,
            "cdd_promedio": round(float(np.mean(cdds_y)), 2) if cdds_y else 0.0,
            "temp_maxima": round(float(np.max(temps_y)), 2) if temps_y else 0.0,
            "ciudad_mas_calida": max(puntos_ano, key=lambda x: x["temperatura"]) if puntos_ano else None
        }
        year_payload = {
            "year": y,
            "current_year": CURRENT_YEAR,
            "es_proyeccion": es_proyeccion,
            "kpis": kpis_y,
            "puntos": puntos_ano
        }
        GLOBAL_CACHE["bytes"]["years"][y] = json.dumps(year_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    series_temporales_payload = {
        "min_year": min(rango_completo),
        "max_year": max(rango_completo),
        "current_year": CURRENT_YEAR,
        "global_historico": global_payload["historico"],
        "global_proyeccion": global_payload["proyeccion"],
        "matriz_interpolacion_base": matriz_base,
        "timeline": timeline_dict
    }
    GLOBAL_CACHE["series_temporales"] = series_temporales_payload
    GLOBAL_CACHE["bytes"]["climate_data"] = json.dumps(series_temporales_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    # 5. Pre-cargar index.html en memoria RAM
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        GLOBAL_CACHE["index_html"] = index_path.read_text(encoding="utf-8")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[OK] GLOBAL_CACHE listo en {elapsed_ms:.1f} ms ({len(METADATA_CIUDADES)} ciudades, {len(rango_completo)} años en RAM).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejador de ciclo de vida FastAPI: carga única y pre-computación total en GLOBAL_CACHE (RAM)."""
    if not METADATA_CIUDADES:
        inicializar_dataset_en_memoria()
    yield
    GLOBAL_CACHE.clear()
    METADATA_CIUDADES.clear()


app = FastAPI(
    title="URBAN-HEAT GRID INTELLIGENCE — Plataforma de Diagnóstico y Predicción Térmica Urbana",
    lifespan=lifespan
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:8080,http://localhost:8080").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/Img/{filename:path}")
async def serve_project_img(filename: str):
    """Sirve archivos de Img con resolución automática de extensión (.png/.jpg) y sin caché obsoleta."""
    if not IMG_DIR.is_dir():
        raise HTTPException(status_code=404, detail="Carpeta Img no encontrada")

    candidate = (IMG_DIR / filename).resolve()
    if not candidate.is_file():
        stem = Path(filename).stem
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".ico"):
            alt = (IMG_DIR / f"{stem}{ext}").resolve()
            if alt.is_file():
                candidate = alt
                break

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Imagen '{filename}' no encontrada")

    return FileResponse(
        str(candidate),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )


def _build_dynamic_year_bytes(year: int) -> bytes:
    """Construye en memoria un año fuera del rango precalculado (1850-2050) sin tocar disco."""
    es_proyeccion = year > CURRENT_YEAR
    puntos = []
    temps = []
    cdds = []
    for city, data in METADATA_CIUDADES.items():
        temp = data["historial"].get(year) if not es_proyeccion else None
        if temp is None:
            temp = round(data["intercept"] + data["slope"] * year, 2)
        cdd_val = calcular_cdd(temp)
        temps.append(temp)
        cdds.append(cdd_val)
        puntos.append({
            "city": city,
            "country": data["country"],
            "tier": data["tier"],
            "lat": data["lat"],
            "lon": data["lon"],
            "temperatura": temp,
            "temp": temp,
            "anomalia": round(temp - data["baseline_anom"], 2),
            "cdd": cdd_val,
            "color": asignar_color_hex_termico(temp),
            "es_proyeccion": es_proyeccion,
            "riesgo": data["riesgo_compacto"]
        })
    payload = {
        "year": year,
        "current_year": CURRENT_YEAR,
        "es_proyeccion": es_proyeccion,
        "kpis": {
            "year": year,
            "current_year": CURRENT_YEAR,
            "es_proyeccion": es_proyeccion,
            "total_ciudades": len(puntos),
            "temp_promedio": round(float(np.mean(temps)), 2) if temps else 0.0,
            "cdd_promedio": round(float(np.mean(cdds)), 2) if cdds else 0.0,
            "temp_maxima": round(float(np.max(temps)), 2) if temps else 0.0,
            "ciudad_mas_calida": max(puntos, key=lambda x: x["temperatura"]) if puntos else None
        },
        "puntos": puntos
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Entrega index.html actualizado y sincroniza GLOBAL_CACHE en RAM."""
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        GLOBAL_CACHE["index_html"] = index_path.read_text(encoding="utf-8")
    html = GLOBAL_CACHE.get("index_html")
    if not html:
        raise HTTPException(status_code=404, detail="index.html no cargado en memoria")
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )


@app.get("/api/config")
async def get_config():
    """Lookup O(1) en GLOBAL_CACHE para configuración inicial y catálogo de ciudades (< 1 ms)."""
    if not METADATA_CIUDADES:
        inicializar_dataset_en_memoria()
    return Response(content=GLOBAL_CACHE["bytes"]["config"], media_type="application/json")


@app.get("/api/climate-data")
@app.get("/api/thermal-timeline")
async def get_climate_data(year: Optional[int] = Query(None)):
    """
    Lookup O(1) sobre GLOBAL_CACHE en RAM (< 1 ms):
    Retorna las series históricas/proyectadas (1850-2050), timeline anual y matriz base de interpolación,
    o el corte específico de un año si se pasa ?year=.
    """
    if year is not None:
        year_bytes = GLOBAL_CACHE["bytes"]["years"].get(year) or _build_dynamic_year_bytes(year)
        return Response(content=year_bytes, media_type="application/json")
    return Response(content=GLOBAL_CACHE["bytes"]["climate_data"], media_type="application/json")


@app.get("/api/heatmap-year")
async def get_heatmap_year(year: int = Query(...)):
    """Lookup O(1) en GLOBAL_CACHE para un año específico del mapa térmico."""
    year_bytes = GLOBAL_CACHE["bytes"]["years"].get(year) or _build_dynamic_year_bytes(year)
    return Response(content=year_bytes, media_type="application/json")


@app.get("/api/city/{name}")
async def get_city_by_name(name: str):
    """
    Lookup directo O(1) sobre GLOBAL_CACHE["bytes"]["cities"] (< 1 ms):
    Retorna coords, cdd_historico, metricas, serie histórica (1850-actual) y proyección (hasta 2050).
    """
    key = name.strip()
    cities_bytes = GLOBAL_CACHE["bytes"]["cities"]
    if key in cities_bytes:
        return Response(content=cities_bytes[key], media_type="application/json")

    canonical = GLOBAL_CACHE["ciudades_lower"].get(key.lower())
    if canonical is not None and canonical in cities_bytes:
        return Response(content=cities_bytes[canonical], media_type="application/json")

    raise HTTPException(status_code=404, detail=f"Ciudad '{name}' no encontrada en GLOBAL_CACHE")


@app.get("/api/city-detail")
async def get_city_detail(city: str = Query("")):
    """Lookup directo O(1) sobre GLOBAL_CACHE (retrocompatible con ?city=)."""
    key = city.strip()
    cities_bytes = GLOBAL_CACHE["bytes"]["cities"]
    if key in cities_bytes:
        return Response(content=cities_bytes[key], media_type="application/json")

    canonical = GLOBAL_CACHE["ciudades_lower"].get(key.lower())
    if canonical is not None and canonical in cities_bytes:
        return Response(content=cities_bytes[canonical], media_type="application/json")

    return Response(content=cities_bytes["Global"], media_type="application/json")


# ==============================================================================
# AGENTE METEOROLÓGICO INTELIGENTE: CACHÉ CON TTL (60 MIN) Y TELEMETRÍA GLOBAL
# ==============================================================================
class RiesgoSemaforo(BaseModel):
    nivel: str = Field(..., description="Nivel de criticidad: ALTO, MODERADO o BAJO")
    etiqueta: str = Field(..., description="Etiqueta semafórica del estado")
    color: str = Field(..., description="Color semafórico (red, yellow, green)")
    hex: str = Field(..., description="Código hexadecimal del semáforo")


class LiveWeatherItem(BaseModel):
    city: str = Field(..., description="Nombre de la metrópoli")
    country: str = Field(..., description="País")
    tier: int = Field(..., description="Jerarquía urbana: 1 (Global Tier 1) o 2 (Regional Tier 2)")
    lat: float = Field(..., description="Latitud decimal")
    lon: float = Field(..., description="Longitud decimal")
    temp: float = Field(..., description="Temperatura actual (°C)")
    current_temp: float = Field(..., description="Temperatura del día actual (°C)")
    temperatura: float = Field(..., description="Temperatura normalizada para el raster (°C)")
    anomalia: float = Field(..., description="Anomalía respecto a la línea base (°C)")
    cdd: float = Field(..., description="Cooling Degree Days del día actual (base 18.0 °C)")
    color: str = Field(..., description="Color hexadecimal GFS asignado por el agente")
    status: str = Field(..., description="Estado de sincronización (LIVE_OPEN_METEO o FALLBACK_PROJECTION)")
    riesgo: RiesgoSemaforo = Field(..., description="Estado semafórico asignado")


class AgentPointQueryResponse(BaseModel):
    lat: float = Field(..., description="Latitud consultada")
    lon: float = Field(..., description="Longitud consultada")
    temp: float = Field(..., description="Temperatura actual en el punto (°C)")
    humidity: int = Field(..., description="Humedad relativa a 2m (%)")
    anomalia: float = Field(..., description="Anomalía térmica estimada (°C)")
    cdd: float = Field(..., description="Cooling Degree Days (base 18.0 °C)")
    condition: str = Field(..., description="Condición bioclimática")
    status: str = Field(..., description="Estado operativo de red / telemetría")
    color: str = Field(..., description="Color hexadecimal asignado según espectro GFS")


# Estructura de caché del Agente Meteorológico indexada por clave de consulta (TTL: 60 minutos)
AGENT_CACHE: Dict[str, Dict[str, Any]] = {}
AGENT_CACHE_TTL: float = 3600.0  # 60 minutos en segundos

LIVE_WEATHER_CACHE: Dict[str, Any] = {
    "timestamp": 0.0,
    "ttl": 3600.0,  # 60 minutos
    "data": None
}


def _agent_cache_get(key: str) -> Optional[Any]:
    """Obtiene una entrada válida de AGENT_CACHE si no ha expirado el TTL de 60 minutos."""
    entry = AGENT_CACHE.get(key)
    if entry is not None and (time.time() - entry["timestamp"]) < AGENT_CACHE_TTL:
        return entry["data"]
    return None


def _agent_cache_set(key: str, data: Any) -> None:
    """Almacena una respuesta en AGENT_CACHE con marca de tiempo actual."""
    AGENT_CACHE[key] = {
        "timestamp": time.time(),
        "data": data
    }


def asignar_semaforo_cdd(cdd: float, current_temp: float) -> dict:
    """Asigna el estado semafórico del día en curso según el CDD (base 18.0 °C)."""
    if cdd >= 6.0 or current_temp >= 28.0:
        return {
            "nivel": "ALTO",
            "etiqueta": "ALTO RIESGO CRÍTICO",
            "color": "red",
            "hex": "#ef4444"
        }
    elif cdd >= 2.0 or current_temp >= 22.0:
        return {
            "nivel": "MODERADO",
            "etiqueta": "RIESGO MODERADO",
            "color": "yellow",
            "hex": "#f59e0b"
        }
    else:
        return {
            "nivel": "BAJO",
            "etiqueta": "ESTABLE / BAJO RIESGO",
            "color": "green",
            "hex": "#10b981"
        }


def obtener_clima_en_vivo() -> list:
    """
    Consulta la temperatura actual/media del día en curso para las 100 metrópolis
    mediante Open-Meteo Forecast API (con caché en AGENT_CACHE de 60 minutos).
    Si no hay conexión o falla la API, aplica fallback gracioso con la proyección del año actual.
    """
    cached = _agent_cache_get("live_cities")
    if cached is not None:
        return cached

    now = time.time()
    if LIVE_WEATHER_CACHE["data"] is not None and (now - LIVE_WEATHER_CACHE["timestamp"]) < LIVE_WEATHER_CACHE["ttl"]:
        _agent_cache_set("live_cities", LIVE_WEATHER_CACHE["data"])
        return LIVE_WEATHER_CACHE["data"]

    ciudades_ordenadas = sorted(METADATA_CIUDADES.keys())
    live_temps: Dict[str, float] = {}

    # Consulta por lotes de 50 coordenadas a Open-Meteo Forecast API (evita límites de longitud en URL)
    batch_size = 50
    for i in range(0, len(ciudades_ordenadas), batch_size):
        batch_cities = ciudades_ordenadas[i:i + batch_size]
        lats = ",".join(str(METADATA_CIUDADES[c]["lat"]) for c in batch_cities)
        lons = ",".join(str(METADATA_CIUDADES[c]["lon"]) for c in batch_cities)
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lats}&longitude={lons}"
            f"&current=temperature_2m&daily=temperature_2m_mean&hourly=temperature_2m"
            f"&forecast_days=1&timezone=auto"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "UrbanHeatGridIntelligence/2.0"})
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                items = payload if isinstance(payload, list) else [payload]
                for c_name, item in zip(batch_cities, items):
                    temp_val = None
                    if isinstance(item, dict):
                        cur = item.get("current") or {}
                        daily = item.get("daily") or {}
                        hourly = item.get("hourly") or {}
                        if cur.get("temperature_2m") is not None:
                            temp_val = float(cur["temperature_2m"])
                        elif daily.get("temperature_2m_mean") and daily["temperature_2m_mean"][0] is not None:
                            temp_val = float(daily["temperature_2m_mean"][0])
                        elif hourly.get("temperature_2m"):
                            valid_h = [float(x) for x in hourly["temperature_2m"] if x is not None]
                            if valid_h:
                                temp_val = float(np.mean(valid_h))
                    if temp_val is not None:
                        live_temps[c_name] = round(temp_val, 1)
        except Exception:
            # Fallback silencioso por lote en caso de timeout o falta de red
            pass

    resultado = []
    for city in ciudades_ordenadas:
        meta = METADATA_CIUDADES[city]
        if city in live_temps:
            current_temp = live_temps[city]
            status = "LIVE_OPEN_METEO"
        else:
            fallback_t = meta["historial"].get(
                CURRENT_YEAR,
                round(meta["intercept"] + meta["slope"] * CURRENT_YEAR, 2)
            )
            current_temp = round(float(fallback_t), 1)
            status = "FALLBACK_PROJECTION"

        cdd_val = round(max(0.0, current_temp - BASE_COMFORT_CDD), 2)
        semaforo = asignar_semaforo_cdd(cdd_val, current_temp)
        color_hex = asignar_color_hex_termico(current_temp)

        resultado.append({
            "city": city,
            "country": meta["country"],
            "tier": meta["tier"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "temp": current_temp,
            "current_temp": current_temp,
            "temperatura": current_temp,
            "anomalia": round(current_temp - meta["baseline_anom"], 2),
            "cdd": cdd_val,
            "color": color_hex,
            "status": status,
            "riesgo": semaforo
        })

    LIVE_WEATHER_CACHE["timestamp"] = now
    LIVE_WEATHER_CACHE["data"] = resultado
    _agent_cache_set("live_cities", resultado)
    return resultado


def consultar_punto_global_agente(lat: float, lon: float) -> dict:
    """
    Agente Meteorológico para consulta en cualquier coordenada del planeta.
    Consulta Open-Meteo Forecast API con caché en memoria (TTL 60 min) y fallback sinóptico.
    """
    lat_r = round(float(lat), 3)
    lon_norm = ((float(lon) + 180.0) % 360.0) - 180.0
    lon_r = round(lon_norm, 3)

    # Clave espacial discretizada a 0.05° (~5.5 km) para aprovechar AGENT_CACHE eficientemente
    cache_key = f"point:{round(lat_r, 2)}:{round(lon_r, 2)}"
    cached = _agent_cache_get(cache_key)
    if cached is not None:
        return cached

    # Línea base climatológica latitudinal para estimar anomalía local
    lat_rad = np.deg2rad(np.clip(lat_r, -85.0, 85.0))
    lon_rad = np.deg2rad(lon_r)
    base_sinoptica = float(
        -21.0 + 50.0 * (np.cos(lat_rad) ** 1.32)
        + 2.4 * np.sin(lon_rad * 2.0 + lat_rad * 1.5) * np.sin(lat_rad * 2.0)
        + 1.3 * np.cos(lon_rad * 3.0 - lat_rad * 1.2)
    )

    temp_val: Optional[float] = None
    humidity_val: Optional[int] = None

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat_r}&longitude={lon_r}"
        f"&current=temperature_2m,relative_humidity_2m"
        f"&timezone=auto"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "UrbanHeatGridIntelligence/2.0"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            cur = payload.get("current") or {}
            if cur.get("temperature_2m") is not None:
                temp_val = round(float(cur["temperature_2m"]), 1)
            if cur.get("relative_humidity_2m") is not None:
                humidity_val = int(round(float(cur["relative_humidity_2m"])))
    except Exception:
        pass

    if temp_val is None:
        # Fallback sinóptico ponderado con las ciudades registradas más cercanas
        suma_w = 0.0
        delta_w = 0.0
        for meta in METADATA_CIUDADES.values():
            d_lat = lat_r - meta["lat"]
            d_lon = abs(lon_r - meta["lon"])
            if d_lon > 180.0:
                d_lon = 360.0 - d_lon
            dist_sq = (d_lat * d_lat) + (d_lon * d_lon)
            if dist_sq < 225.0:
                w = 1.0 / (dist_sq + 0.25)
                suma_w += w
                delta_w += (meta["temp_actual"] - base_sinoptica) * w
        est_t = base_sinoptica + (delta_w / suma_w if suma_w > 0 else 0.0)
        temp_val = round(float(est_t), 1)

    if humidity_val is None:
        # Estimación higrométrica latitudinal de respaldo
        humidity_val = int(np.clip(round(74.0 - 22.0 * abs(np.sin(lat_rad * 1.8))), 22, 92))

    cdd_val = round(max(0.0, temp_val - BASE_COMFORT_CDD), 2)
    anomalia_val = round(temp_val - base_sinoptica, 2)
    color_hex = asignar_color_hex_termico(temp_val)

    if temp_val >= 32.0:
        condition = "Calor Extremo"
        status = "ESTRÉS TÉRMICO CRÍTICO"
    elif temp_val >= 24.0:
        condition = "Cálido Activo"
        status = "DEMANDA HVAC ALTA"
    elif temp_val >= 14.0:
        condition = "Confort Templado"
        status = "RED ESTABLE"
    elif temp_val >= 0.0:
        condition = "Fresco Moderado"
        status = "BAJA DEMANDA CDD"
    else:
        condition = "Gélido / Polar"
        status = "RÉGIMEN CRIOGÉNICO"

    result = {
        "lat": lat_r,
        "lon": lon_r,
        "temp": temp_val,
        "humidity": humidity_val,
        "anomalia": anomalia_val,
        "cdd": cdd_val,
        "condition": condition,
        "status": status,
        "color": color_hex
    }
    _agent_cache_set(cache_key, result)
    return result


@app.get("/api/live-weather", response_model=List[LiveWeatherItem], tags=["Live Weather"])
@app.get("/api/agent/live-cities", response_model=List[LiveWeatherItem], tags=["Weather Agent"])
async def get_live_weather():
    """
    Endpoint del Agente Meteorológico para ciudades registradas (/api/agent/live-cities y /api/live-weather).
    Retorna temperatura actual (Open-Meteo + caché 60 min), categoría tier (1 | 2), coordenadas, CDD y color hex.
    """
    return await asyncio.to_thread(obtener_clima_en_vivo)


@app.get("/api/agent/point-query", response_model=AgentPointQueryResponse, tags=["Weather Agent"])
async def agent_point_query(
    lat: float = Query(..., description="Latitud geográfica (-90 a 90)"),
    lon: float = Query(..., description="Longitud geográfica (-180 a 180)")
):
    """
    Consulta el clima en vivo de cualquier coordenada del planeta mediante el Agente Meteorológico
    (Open-Meteo Forecast API + caché en memoria AGENT_CACHE con TTL de 60 minutos).
    """
    return await asyncio.to_thread(consultar_punto_global_agente, lat, lon)


# ==============================================================================
# TAREA 1: AGENTE DE TELEMETRÍA CLIMÁTICA UNIVERSAL (TELEMETRY_CACHE + /api/agent/telemetry)
# ==============================================================================
TELEMETRY_CACHE: Dict[str, Dict[str, Any]] = {}
TELEMETRY_CACHE_TTL: float = 3600.0  # Validez de 60 minutos (3600 s)


class UniversalClimateTelemetryAgent:
    """
    Agente de Telemetría Climática Universal con caché en memoria (TTL 60 min),
    integración asíncrona con Open-Meteo Forecast/Geocoding API y respaldo físico-sinóptico local.
    """

    def __init__(self, cache_store: Dict[str, Dict[str, Any]], ttl_seconds: float = 3600.0):
        self.cache = cache_store
        self.ttl = ttl_seconds

    def _make_cache_key(self, lat: float, lon: float, city_name: str) -> str:
        lat_k = round(float(lat), 2)
        lon_k = round(((float(lon) + 180.0) % 360.0) - 180.0, 2)
        city_k = (city_name or "").strip().lower()
        return f"telemetry:{lat_k}:{lon_k}:{city_k}"

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self.cache.get(key)
        if entry is not None and (time.time() - entry["timestamp"]) < self.ttl:
            return entry["data"]
        return None

    def _set_cached(self, key: str, data: Dict[str, Any]) -> None:
        self.cache[key] = {
            "timestamp": time.time(),
            "data": data
        }

    @staticmethod
    def _fetch_open_meteo_sync(url: str, timeout: float = 4.0) -> Optional[Dict[str, Any]]:
        req = urllib.request.Request(url, headers={"User-Agent": "UrbanHeatGridIntelligence/3.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _resolve_nearest_city_meta(self, lat: float, lon: float, city_name: Optional[str]) -> tuple:
        """
        Busca si el nombre o las coordenadas corresponden a una metrópoli registrada
        o estima la línea base histórica local para cualquier coordenada del planeta.
        """
        clean_name = (city_name or "").strip()
        if clean_name and clean_name.lower() not in ("", "punto cartográfico", "global"):
            canonical = GLOBAL_CACHE["ciudades_lower"].get(clean_name.lower())
            if canonical and canonical in METADATA_CIUDADES:
                return canonical, METADATA_CIUDADES[canonical], True

        # Buscar metrópoli cercana dentro de ~0.35° (~38 km)
        best_city = None
        best_meta = None
        best_dist_sq = 0.1225  # 0.35^2
        for c_name, meta in METADATA_CIUDADES.items():
            d_lat = lat - meta["lat"]
            d_lon = abs(lon - meta["lon"])
            if d_lon > 180.0:
                d_lon = 360.0 - d_lon
            dist_sq = (d_lat * d_lat) + (d_lon * d_lon)
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_city = c_name
                best_meta = meta

        if best_city and best_meta:
            resolved_label = clean_name if clean_name and clean_name != "Punto Cartográfico" else best_city
            return resolved_label, best_meta, True

        resolved_label = clean_name if clean_name else "Punto Cartográfico"
        return resolved_label, None, False

    @staticmethod
    def _build_dynamic_action_plan(
        city_label: str,
        temp: float,
        apparent_temp: float,
        humidity: int,
        wind_speed: float,
        cdd: float,
        risk_level: str
    ) -> List[Dict[str, str]]:
        """Genera dinámicamente las 3 directrices del Plan de Acción (Infraestructura, Normativa, Eficiencia)."""
        if temp >= 33.0:
            return [
                {
                    "categoria": "Infraestructura",
                    "icono": "🛡️",
                    "titulo": f"Enfriamiento Forzado en Subestaciones ({city_label})",
                    "descripcion": (
                        f"Activar ventilación asistida y bancos de alivio térmico en transformadores HV ante "
                        f"{temp:.1f} °C (sensación {apparent_temp:.1f} °C) y {cdd:.1f} CDD/día para prevenir falla dieléctrica."
                    )
                },
                {
                    "categoria": "Normativa",
                    "icono": "📜",
                    "titulo": "Protocolo de Respuesta a la Demanda Crítica",
                    "descripcion": (
                        f"Restringir picos industriales en horas solares críticas y exigir consigna mínima de 24 °C "
                        f"en grandes consumidores comerciales ({risk_level})."
                    )
                },
                {
                    "categoria": "Eficiencia",
                    "icono": "⚡",
                    "titulo": "Despliegue de Techos Fríos y Almacenamiento Térmico",
                    "descripcion": (
                        f"Priorizar recubrimientos de alto albedo (SRI > 82) y pre-enfriamiento nocturno asistido "
                        f"(viento actual {wind_speed:.1f} km/h, HR {humidity}%)."
                    )
                }
            ]
        elif temp >= 26.0:
            return [
                {
                    "categoria": "Infraestructura",
                    "icono": "🌳",
                    "titulo": f"Corredores Verdes y Sombreamiento Urbano ({city_label})",
                    "descripcion": (
                        f"Incrementar cobertura arbórea perimetral en nodos de distribución para mitigar carga latente "
                        f"({temp:.1f} °C, HR {humidity}%, sensación {apparent_temp:.1f} °C)."
                    )
                },
                {
                    "categoria": "Normativa",
                    "icono": "📋",
                    "titulo": "Estándar de Aislamiento y Control Higrotérmico ASHRAE",
                    "descripcion": (
                        f"Actualizar envolventes térmicas y deshumidificación eficiente ante {cdd:.1f} CDD diarios "
                        f"bajo régimen {risk_level}."
                    )
                },
                {
                    "categoria": "Eficiencia",
                    "icono": "🔄",
                    "titulo": "Optimización de Chillers e Inversores HVAC",
                    "descripcion": (
                        f"Modular compresores de caudal variable (VRF) aprovechando ventilación natural "
                        f"({wind_speed:.1f} km/h) en horas valle."
                    )
                }
            ]
        else:
            return [
                {
                    "categoria": "Infraestructura",
                    "icono": "🏙️",
                    "titulo": f"Mantenimiento Preventivo de Red ({city_label})",
                    "descripcion": (
                        f"Condición térmica favorable ({temp:.1f} °C, CDD {cdd:.1f}). Ejecutar termografía infrarroja "
                        f"de rutina y balanceo de fases sin estrés térmico."
                    )
                },
                {
                    "categoria": "Normativa",
                    "icono": "🏛️",
                    "titulo": "Zonificación Bioclimática y Superficies Permeables",
                    "descripcion": (
                        f"Conservar el margen de resiliencia operativa ({risk_level}) mediante estándares de "
                        f"urbanismo pasivo y preservación de sumideros térmicos."
                    )
                },
                {
                    "categoria": "Eficiencia",
                    "icono": "🌬️",
                    "titulo": "Free-Cooling y Ventilación Cruzada Inteligente",
                    "descripcion": (
                        f"Maximizar enfriamiento gratuito por aire exterior ({wind_speed:.1f} km/h, HR {humidity}%) "
                        f"minimizando el consumo mecánico de climatización."
                    )
                }
            ]

    async def consultar_telemetria(
        self,
        lat: float,
        lon: float,
        city_name: Optional[str] = "Punto Cartográfico"
    ) -> Dict[str, Any]:
        lat_r = round(float(np.clip(lat, -90.0, 90.0)), 4)
        lon_norm = ((float(lon) + 180.0) % 360.0) - 180.0
        lon_r = round(lon_norm, 4)

        resolved_city, matched_meta, is_catalog_city = self._resolve_nearest_city_meta(lat_r, lon_r, city_name)
        cache_key = self._make_cache_key(lat_r, lon_r, resolved_city)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Línea base histórica local (de la metrópoli registrada o modelo físico-sinóptico latitudinal)
        lat_rad = np.deg2rad(np.clip(lat_r, -85.0, 85.0))
        lon_rad = np.deg2rad(lon_r)
        base_sinoptica = float(
            -21.0 + 50.0 * (np.cos(lat_rad) ** 1.32)
            + 2.4 * np.sin(lon_rad * 2.0 + lat_rad * 1.5) * np.sin(lat_rad * 2.0)
            + 1.3 * np.cos(lon_rad * 3.0 - lat_rad * 1.2)
        )
        base_hist = float(matched_meta["media_hist"]) if matched_meta else base_sinoptica

        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat_r}&longitude={lon_r}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m"
            f"&timezone=auto"
        )

        temp_val: Optional[float] = None
        apparent_val: Optional[float] = None
        humidity_val: Optional[int] = None
        wind_val: Optional[float] = None
        source_status = "LIVE_OPEN_METEO"

        try:
            payload = await asyncio.to_thread(self._fetch_open_meteo_sync, url, 4.0)
            if isinstance(payload, dict):
                cur = payload.get("current") or {}
                if cur.get("temperature_2m") is not None:
                    temp_val = round(float(cur["temperature_2m"]), 1)
                if cur.get("apparent_temperature") is not None:
                    apparent_val = round(float(cur["apparent_temperature"]), 1)
                if cur.get("relative_humidity_2m") is not None:
                    humidity_val = int(round(float(cur["relative_humidity_2m"])))
                if cur.get("wind_speed_10m") is not None:
                    wind_val = round(float(cur["wind_speed_10m"]), 1)
        except Exception:
            source_status = "FALLBACK_PROJECTION"

        # Respaldo local inteligente si no hay respuesta de red
        if temp_val is None:
            source_status = "FALLBACK_PROJECTION"
            if matched_meta is not None:
                temp_val = round(float(matched_meta["temp_actual"]), 1)
            else:
                suma_w = 0.0
                delta_w = 0.0
                for meta in METADATA_CIUDADES.values():
                    d_lat = lat_r - meta["lat"]
                    d_lon = abs(lon_r - meta["lon"])
                    if d_lon > 180.0:
                        d_lon = 360.0 - d_lon
                    dist_sq = (d_lat * d_lat) + (d_lon * d_lon)
                    if dist_sq < 225.0:
                        w = 1.0 / (dist_sq + 0.25)
                        suma_w += w
                        delta_w += (meta["temp_actual"] - base_sinoptica) * w
                temp_val = round(float(base_sinoptica + (delta_w / suma_w if suma_w > 0 else 0.0)), 1)

        if humidity_val is None:
            humidity_val = int(np.clip(round(72.0 - 20.0 * abs(np.sin(lat_rad * 1.8))), 20, 92))
        if wind_val is None:
            wind_val = round(float(11.5 + 6.0 * abs(np.cos(lat_rad * 2.2))), 1)
        if apparent_val is None:
            # Índice higrotérmico aproximado
            apparent_val = round(temp_val + (humidity_val - 50.0) * 0.04 - wind_val * 0.06, 1)

        # Cálculo de CDD diario base 18.0 °C (Estándar ASHRAE)
        cdd = max(0.0, round(temp_val - 18.0, 2))

        # Porcentaje de aumento relativo a la base histórica local
        if matched_meta is not None and matched_meta.get("pct_aumento_cdd") is not None:
            base_cdd_local = max(0.5, float(matched_meta.get("cdd_hist_base", 1.0)))
            if cdd > 0:
                pct_increase = round(((cdd - base_cdd_local) / base_cdd_local) * 100.0, 1)
            else:
                pct_increase = round(float(matched_meta["pct_aumento_cdd"]), 1)
        else:
            cdd_hist_local = max(0.5, round(max(0.0, base_hist - 18.0), 2))
            if cdd > 0:
                pct_increase = round(((cdd - cdd_hist_local) / cdd_hist_local) * 100.0, 1)
            else:
                pct_increase = round(((temp_val - base_hist) / max(abs(base_hist), 10.0)) * 100.0, 1)

        # Semáforo de red eléctrica según umbrales térmicos operativos
        if temp_val >= 33.0:
            risk_level = "[ALTO RIESGO CRÍTICO]"
            risk_desc = "Sobrecarga HV, riesgo de falla dieléctrica en transformadores"
        elif temp_val >= 26.0:
            risk_level = "[ESTRÉS MODERADO]"
            risk_desc = "Incremento sostenido de carga latente por climatización"
        else:
            risk_level = "[ESTABLE / BAJO RIESGO]"
            risk_desc = "Margen de resiliencia operativa óptimo"

        color_hex = asignar_color_hex_termico(temp_val)
        plan = self._build_dynamic_action_plan(
            resolved_city, temp_val, apparent_val, humidity_val, wind_val, cdd, risk_level
        )

        # Serie histórica y proyección para sincronizar también la pestaña "Curva Térmica" en cualquier coordenada
        if is_catalog_city and matched_meta is not None:
            city_payload = GLOBAL_CACHE["ciudades"].get(resolved_city) or construir_payload_ciudad(
                resolved_city, matched_meta, matched_meta["historial"]
            )
            historico_serie = city_payload.get("historico", [])
            proyeccion_serie = city_payload.get("proyeccion", [])
            delta_temp_2040 = matched_meta["delta_temp_2040"]
            calentamiento_decada = matched_meta["calentamiento_decada"]
        else:
            slope_est = 0.024 + 0.012 * abs(np.sin(lat_rad))
            calentamiento_decada = round(float(slope_est * 10.0), 3)
            delta_temp_2040 = round(float(slope_est * (HORIZONTE_PROYECCION - CURRENT_YEAR)), 2)
            historico_serie = []
            for y in range(1950, CURRENT_YEAR + 1, 2):
                t_y = round(temp_val - slope_est * (CURRENT_YEAR - y), 2)
                historico_serie.append({
                    "year": y,
                    "temperatura": t_y,
                    "cdd": calcular_cdd(t_y)
                })
            proyeccion_serie = []
            for y in range(CURRENT_YEAR + 1, HORIZONTE_PROYECCION + 1):
                t_p = round(temp_val + slope_est * (y - CURRENT_YEAR), 2)
                proyeccion_serie.append({
                    "year": y,
                    "prediccion": t_p,
                    "cdd_proy": calcular_cdd(t_p)
                })

        result = {
            "city": resolved_city,
            "lat": lat_r,
            "lon": lon_r,
            "temp": temp_val,
            "apparent_temp": apparent_val,
            "humidity": humidity_val,
            "wind_speed": wind_val,
            "cdd": cdd,
            "risk_level": risk_level,
            "risk_desc": risk_desc,
            "pct_increase": pct_increase,
            "color": color_hex,
            "plan": plan,
            "status": source_status,
            "delta_temp_2040": delta_temp_2040,
            "calentamiento_decada": calentamiento_decada,
            "historico": historico_serie,
            "proyeccion": proyeccion_serie
        }

        self._set_cached(cache_key, result)
        return result

    async def geocodificar_ciudad_universal(self, query: str) -> List[Dict[str, Any]]:
        """
        Delega la búsqueda de localidades directamente a geocoding_service.search(q)
        manteniendo compatibilidad con el formato de respuesta esperado por el frontend.
        """
        q = (query or "").strip()
        if not q:
            return []

        geo_items = await geocoding_service.search(q)
        resultados: List[Dict[str, Any]] = []
        for item in geo_items:
            name = str(item.get("name") or "")
            country = str(item.get("country") or "")
            display_name = str(item.get("display_name") or (f"{name}, {country}" if country else name))
            resultados.append(
                {
                    "city": name,
                    "name": name,
                    "display_name": display_name,
                    "country": country,
                    "admin1": "",
                    "lat": round(float(item.get("lat", 0.0)), 4),
                    "lon": round(float(item.get("lon", 0.0)), 4),
                    "type": str(item.get("type") or "city"),
                    "source": "NOMINATIM_SQLITE",
                }
            )

        if not resultados:
            q_lower = q.lower()
            for c_name, meta in METADATA_CIUDADES.items():
                if q_lower in c_name.lower() or q_lower in meta["country"].lower():
                    resultados.append(
                        {
                            "city": c_name,
                            "name": c_name,
                            "display_name": f"{c_name}, {meta['country']}",
                            "country": meta["country"],
                            "admin1": "",
                            "lat": round(float(meta["lat"]), 4),
                            "lon": round(float(meta["lon"]), 4),
                            "type": "city",
                            "source": "CATALOG",
                        }
                    )
                    if len(resultados) >= 8:
                        break

        return resultados[:8]


telemetry_agent = UniversalClimateTelemetryAgent(TELEMETRY_CACHE, TELEMETRY_CACHE_TTL)


@app.get("/api/agent/telemetry", tags=["Weather Agent"])
async def get_universal_agent_telemetry(
    lat: float = Query(..., description="Latitud geográfica (-90 a 90)"),
    lon: float = Query(..., description="Longitud geográfica (-180 a 180)"),
    city_name: Optional[str] = Query("Punto Cartográfico", description="Nombre de la ciudad o Punto Cartográfico")
):
    """
    Endpoint Universal de Telemetría Climática:
    Verifica TELEMETRY_CACHE (TTL 60 min), consulta Open-Meteo asíncronamente, calcula CDD diario base 18.0 °C,
    define el semáforo operativo de red eléctrica y genera dinámicamente las 3 directrices del Plan de Acción.
    """
    return await telemetry_agent.consultar_telemetria(lat=lat, lon=lon, city_name=city_name)


@app.get("/api/agent/geocode", tags=["Weather Agent"])
async def geocode_universal_city(
    q: str = Query(..., description="Nombre de cualquier ciudad del mundo")
):
    """Geocodifica cualquier ciudad del mundo delegando en geocoding_service.search(q)."""
    return await telemetry_agent.geocodificar_ciudad_universal(q)


# ==============================================================================
# TAREA 2: ENDPOINTS DE GEOCODIFICACIÓN GLOBAL (SQLite + Nominatim) Y MODO SISTEMA
# ==============================================================================
SYSTEM_STATE: Dict[str, str] = {"mode": os.getenv("SYSTEM_DEFAULT_MODE", "demo").strip().lower()}


@app.get("/api/geo/search", tags=["Global Geocoding"])
async def geo_search_places(
    q: str = Query(..., min_length=1, description="Ciudad, municipio o pueblo a geocodificar")
):
    """
    Busca localidades globales:
    - Si existe en la caché SQLite (data/geo_cache.db), retorna en < 2 ms.
    - Si no existe, consulta Nominatim OSM, persiste en SQLite y retorna la lista estructurada.
    """
    results = await geocoding_service.search(q)
    if results:
        return results

    # Respaldo instantáneo sobre el catálogo en memoria si no hay red hacia Nominatim
    q_norm = q.strip().lower()
    fallback_list: List[Dict[str, Any]] = []
    for c_name, meta in METADATA_CIUDADES.items():
        if q_norm in c_name.lower() or q_norm in meta["country"].lower():
            fallback_list.append(
                {
                    "name": c_name,
                    "display_name": f"{c_name}, {meta['country']}",
                    "lat": round(float(meta["lat"]), 4),
                    "lon": round(float(meta["lon"]), 4),
                    "type": "city",
                    "country": meta["country"],
                }
            )
            if len(fallback_list) >= 5:
                break
    return fallback_list


@app.get("/api/geo/reverse", tags=["Global Geocoding"])
async def geo_reverse_coordinates(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitud decimal"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitud decimal"),
):
    """Resuelve el topónimo (pueblo, municipio o ciudad) más cercano a las coordenadas enviadas."""
    return await geocoding_service.reverse(lat=lat, lon=lon)


@app.get("/api/system/mode", tags=["System"])
async def get_or_set_system_mode(
    mode: Optional[str] = Query(default=None, description="Opcional: cambiar modo ('demo' | 'live')")
):
    """Retorna (y opcionalmente actualiza) el modo activo del sistema: {'mode': 'demo' | 'live'}."""
    if mode is not None and mode.strip().lower() in ("demo", "live"):
        SYSTEM_STATE["mode"] = mode.strip().lower()
    return {"mode": SYSTEM_STATE["mode"]}


@app.get("/api/weather/live", tags=["Weather Live"])
async def get_live_weather(lat: float, lon: float):
    """
    Obtiene telemetría en tiempo real de Open-Meteo para cualquier coordenada.
    """
    try:
        data = await weather_service.obtener_clima_actual(lat, lon)
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fallo al conectar con Open-Meteo: {str(e)}")



# ==============================================================================
# ACOPLAMIENTO DEL MOTOR DE INFERENCIA ML (SINGLETON EN MEMORIA)
# ==============================================================================
inference_engine = UrbanHeatInferenceEngine()
engine = inference_engine


@app.get("/api/global/thermal-curve", tags=["Global Thermal Curve (ERA5 + ML)"])
async def get_global_thermal_curve(lat: float, lon: float):
    """
    Entrega el historial real observado (1940-2025) vía Open-Meteo ERA5 Reanalysis
    y la proyección ML (2026-2050) calculada por RandomForestRegressor + GradientBoosting
    para cualquier punto del planeta, sin tocar el dataset demo.
    """
    # 1. Obtener historial real ERA5 (1940-2025)
    historial = await obtener_historial_real_global(lat, lon)

    # Calcular baseline real (1961-1990)
    temps_base = [p["temp"] for p in historial if 1961 <= p["year"] <= 1990]
    baseline_temp = float(np.mean(temps_base)) if temps_base else (historial[-1]["temp"] - 0.8 if historial else 20.0)

    # 2. Proyectar con el RandomForestRegressor + GradientBoosting (2026 - 2050)
    proyeccion = []
    co2_estimado = 425.0
    for year in range(2026, 2051):
        co2_estimado += 2.3  # Incremento tendencial CO2
        pred = inference_engine.predecir_coordenada_libre(lat, lon, year, 7, co2_estimado, baseline_temp)
        proyeccion.append(
            {
                "year": year,
                "temp": pred["temp_predicha"],
                "sigma": pred["incertidumbre_sigma"],
                "ic_min": pred["intervalo_95"]["min"],
                "ic_max": pred["intervalo_95"]["max"],
                "cdd": pred["cdd"],
                "delta_t": pred["anomalia"],
                "estres_red": pred["nivel_estres"],
                "prob_critico": pred["probabilidades"]["critico"],
                "probabilidades": pred["probabilidades"],
                "plan_accion": pred["plan_accion"],
            }
        )

    return {
        "status": "success",
        "fuente_historica": "Open-Meteo ERA5 Reanalysis (Observación Real 1940-2025)",
        "fuente_proyeccion": "RandomForestRegressor + GradientBoosting (2026-2050)",
        "baseline_termico": round(baseline_temp, 2),
        "historial": historial,
        "proyeccion": proyeccion,
    }


@app.post("/api/predict", response_model=PredictionOutput, tags=["Machine Learning"])
async def predict_climate_scenario(data: PredictionInput):
    """
    Endpoint oficial de inferencia contractual (POST /api/predict).
    Ejecuta UrbanHeatInferenceEngine en memoria (< 10 ms) retornando temperatura proyectada,
    anomalía, demanda de refrigeración (CDD), semáforo de estrés eléctrico y métricas R²/RMSE.
    """
    try:
        return inference_engine.predecir(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno en motor de inferencia: {str(e)}")


@app.get("/api/model-metrics", tags=["Machine Learning"])
async def get_model_metrics():
    """Retorna las métricas oficiales de validación (80/20 split) del motor UrbanHeatInferenceEngine."""
    return {
        "r2": inference_engine.r2,
        "rmse": inference_engine.rmse,
        "metrics": inference_engine.metrics,
    }


class ExportReportRequest(BaseModel):
    city: str = Field(default="Global", description="Nombre de la ciudad o coordenada")
    country: Optional[str] = Field(default="Global")
    lat: Optional[float] = Field(default=0.0)
    lon: Optional[float] = Field(default=0.0)
    target_year: int = Field(default=2026)
    mode: str = Field(default="demo")
    observed_temp: Optional[float] = Field(default=None)
    apparent_temp: Optional[float] = Field(default=None)
    humidity: Optional[float] = Field(default=None)
    wet_bulb_temp: Optional[float] = Field(default=None)
    predicted_temp: Optional[float] = Field(default=None)
    sigma_temp: Optional[float] = Field(default=None)
    rango_min: Optional[float] = Field(default=None)
    rango_max: Optional[float] = Field(default=None)
    anomaly: Optional[float] = Field(default=None)
    cdd: Optional[float] = Field(default=None)
    pct_cdd: Optional[float] = Field(default=None)
    risk_level: Optional[str] = Field(default="MODERADO")
    grid_stress: Optional[str] = Field(default="Alerta")
    probabilities: Optional[Dict[str, float]] = Field(default_factory=dict)
    action_plan: Optional[List[Any]] = Field(default_factory=list)


@app.post("/api/export/report", tags=["Export"])
async def export_technical_report(payload: ExportReportRequest):
    """
    Genera y retorna como descarga adjunta la Ficha Técnica de Resiliencia Climática y de Red
    con los parámetros observados, la extrapolación del modelo en cascada y el plan de acción.
    """
    safe_city = "".join(ch if ch.isalnum() else "_" for ch in (payload.city or "Global")).strip("_") or "Global"
    filename = f"Ficha_Resiliencia_{safe_city}_{payload.target_year}.json"

    report_doc = {
        "sistema": "Urban-Heat Grid Intelligence v2.0",
        "fecha_generacion_utc": datetime.utcnow().isoformat() + "Z",
        "modo_operacion": payload.mode.upper(),
        "ubicacion": {
            "ciudad_localidad": payload.city,
            "pais": payload.country,
            "latitud": payload.lat,
            "longitud": payload.lon,
        },
        "telemetria_tiempo_real": {
            "temperatura_observada_c": payload.observed_temp,
            "sensacion_termica_c": payload.apparent_temp,
            "humedad_relativa_pct": payload.humidity,
            "temperatura_bulbo_humedo_stull_c": payload.wet_bulb_temp,
        },
        "inferencia_ml_cascada": {
            "horizonte_anio": payload.target_year,
            "temperatura_proyectada_c": payload.predicted_temp,
            "incertidumbre_inter_arbol_sigma_c": payload.sigma_temp,
            "intervalo_confianza_95_c": [payload.rango_min, payload.rango_max],
            "anomalia_termica_delta_c": payload.anomaly,
            "grados_dia_refrigeracion_cdd": payload.cdd,
            "incremento_demanda_hvac_pct": payload.pct_cdd,
            "semaforo_riesgo_red": payload.risk_level,
            "estado_operativo_red": payload.grid_stress,
            "probabilidades_riesgo_pct": payload.probabilities,
            "metricas_validacion_holdout_20pct": {
                "r2_test": inference_engine.r2,
                "rmse_test_c": inference_engine.rmse,
            },
        },
        "plan_accion_prescriptivo": payload.action_plan,
    }

    content_bytes = json.dumps(report_doc, ensure_ascii=False, indent=2).encode("utf-8")
    return Response(
        content=content_bytes,
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/export/report-image", tags=["Export"])
async def export_technical_report_image(payload: ExportReportRequest):
    """
    Renderiza la ficha ejecutiva de resiliencia climática y de red en formato PNG (300 DPI)
    utilizando el motor gráfico de pipeline/visual_auditor.py, guarda una copia en
    evidencias_finales/export_{ciudad}_{anio}.png y la retorna como descarga adjunta.
    """
    from pipeline.visual_auditor import renderizar_ficha_reporte_png

    buf, _out_path, ciudad, anio = await asyncio.to_thread(
        renderizar_ficha_reporte_png, payload.model_dump()
    )
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="Ficha_Resiliencia_{ciudad}_{anio}.png"',
            "Cache-Control": "no-store",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", 8080)),
        log_level=os.getenv("LOG_LEVEL", "info"),
        reload=False,
    )
