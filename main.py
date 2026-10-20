import os
import json
import urllib.request
from datetime import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Rutas de archivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "Temperaturas", "Global Land Temperatures", "GlobalLandTemperaturesByMajorCity.csv")
CACHE_FILE = os.path.join(BASE_DIR, "open_meteo_cache.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")

CURRENT_YEAR = datetime.now().year # Año actual dinámico (ej. 2026)
HORIZONTE_PROYECCION = 2050 # Cobertura de proyección extendida hasta 2050
BASE_COMFORT_CDD = 18.0 # Estándar ASHRAE (18°C)

app = FastAPI(title="ClimateAI: Analítica Predictiva para Climatización y Resiliencia Energética")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
    - DESERTICO: Alta radiación solar, aridez extrema o latitudes secas (Medio Oriente, Norte de México, Sahara, etc.)
    - COSTERO_TROPICAL: Zonas costeras/tropicales de alta humedad o islas (Caribe, Sudeste Asiático, India costera, etc.)
    - TEMPLADO_BOREAL: Europa Central/Norte, Norteamérica septentrional, zonas templadas frías en transición.
    """
    city_l = city.lower()
    country_l = country.lower()
    lat_abs = abs(lat)

    # Detección de Desértico / Árido
    desert_keywords = [
        "cairo", "baghdad", "riyadh", "dubai", "alexandria", "damascus", "amman", "jeddah",
        "kuwait", "doha", "casablanca", "tunis", "algiers", "hermosillo", "mexicali", "monterrey",
        "phoenix", "las vegas", "karachi", "lahore", "delhi", "jaipur", "ahmadabad"
    ]
    desert_countries = ["egypt", "iraq", "saudi arabia", "iran", "morocco", "algeria", "tunisia", "united arab emirates", "kuwait", "qatar"]

    if any(k in city_l for k in desert_keywords) or any(c in country_l for c in desert_countries):
        return "DESERTICO"

    # Detección de Templado / Boreal en transición
    if lat_abs >= 40.0:
        return "TEMPLADO_BOREAL"

    cold_transition_countries = ["germany", "united kingdom", "france", "russia", "poland", "canada", "ukraine", "japan", "sweden", "norway"]
    if any(c in country_l for c in cold_transition_countries):
        return "TEMPLADO_BOREAL"

    # Si es latitud tropical / intertropical o cálida costera
    if lat_abs <= 28.0 or temp_media >= 22.0:
        return "COSTERO_TROPICAL"

    return "TEMPLADO_BOREAL"

def generar_medidas_prescriptivas(riesgo_info: dict, delta_temp: float, pct_cdd: float, perfil_clima: str = "COSTERO_TROPICAL", city_name: str = "") -> list:
    """
    Genera el catálogo de 4 medidas prescriptivas diferenciadas y contextuales
    según el perfil geoclimático, el semáforo de riesgo y los valores dinámicos calculados.
    """
    nivel = riesgo_info["nivel"]

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
                "descripcion": f"Incentivos fiscales para migrar a equipos con compresores de velocidad variable certificados para operar hasta 52°C sin caída de eficiencia, gestionando la curva de demanda en horas pico de insolación.",
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
                "titulo": f"Protección Anticorrosiva & Resiliencia de Subestaciones Costeras",
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
    else: # TEMPLADO_BOREAL
        return [
            {
                "categoria": "1. Infraestructura Eléctrica",
                "titulo": f"Reconversión y Refuerzo de Redes para Picos Estivales",
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

# Estructuras en memoria optimizadas
METADATA_CIUDADES = {} 
ANOS_HISTORICOS = []
PROYECCION_ANOS = list(range(CURRENT_YEAR + 1, HORIZONTE_PROYECCION + 1))
GLOBAL_HISTORIAL = {}
GLOBAL_MODEL = {}

def cargar_cache_meteo() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def guardar_cache_meteo(cache: dict):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Error guardando caché: {e}")

OPEN_METEO_CACHE = cargar_cache_meteo()

def fetch_open_meteo_anual(city: str, lat: float, lon: float, start_year: int = 2014, end_year: int = CURRENT_YEAR) -> dict:
    cache_key = f"{city}_{round(lat, 2)}_{round(lon, 2)}"
    cached_data = OPEN_METEO_CACHE.get(cache_key, {})

    anos_faltantes = [y for y in range(start_year, end_year + 1) if str(y) not in cached_data]
    if not anos_faltantes:
        return {int(k): v for k, v in cached_data.items()}

    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={round(lat, 4)}&longitude={round(lon, 4)}"
            f"&start_date=2014-01-01&end_date={today_str}"
            f"&daily=temperature_2m_mean&timezone=auto"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "ClimateAI/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        times = data.get("daily", {}).get("time", [])
        temps = data.get("daily", {}).get("temperature_2m_mean", [])

        if times and temps:
            df_temp = pd.DataFrame({"time": times, "temp": temps})
            df_temp["temp"] = pd.to_numeric(df_temp["temp"], errors="coerce")
            df_temp = df_temp.dropna(subset=["temp"])
            df_temp["year"] = pd.to_datetime(df_temp["time"]).dt.year

            anuales = df_temp.groupby("year")["temp"].mean().round(2).to_dict()
            for y, val in anuales.items():
                cached_data[str(y)] = float(val)

            OPEN_METEO_CACHE[cache_key] = cached_data
            guardar_cache_meteo(OPEN_METEO_CACHE)

    except Exception as e:
        pass

    return {int(k): v for k, v in cached_data.items()}

def inicializar_dataset():
    """Lee Berkeley Earth, añade puente Open-Meteo y calcula indicadores ASHRAE y semáforo de riesgo."""
    global METADATA_CIUDADES, ANOS_HISTORICOS, GLOBAL_HISTORIAL, GLOBAL_MODEL, PROYECCION_ANOS
    print("[INIT] Cargando dataset ClimateAI...")

    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"No se encontró el dataset en: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=['AverageTemperature', 'dt'])
    df['year'] = pd.to_datetime(df['dt'], errors='coerce').dt.year
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    df = df[(df['year'] >= 1850) & (df['year'] <= 2013)]

    df_annual = df.groupby(['City', 'Country', 'Latitude', 'Longitude', 'year'])['AverageTemperature'].mean().round(2).reset_index()

    ciudades_coords = df[['City', 'Country', 'Latitude', 'Longitude']].drop_duplicates('City')
    ciudades_coords['lat'] = ciudades_coords['Latitude'].apply(parse_coord)
    ciudades_coords['lon'] = ciudades_coords['Longitude'].apply(parse_coord)
    coords_dict = ciudades_coords.set_index('City')[['Country', 'lat', 'lon']].to_dict('index')

    meta_ciudades = {}
    todos_los_anos_reales = set(range(1850, CURRENT_YEAR + 1))

    for city, group in df_annual.groupby('City'):
        coord_info = coords_dict.get(city, {'Country': 'Unknown', 'lat': 0.0, 'lon': 0.0})
        lat = coord_info['lat']
        lon = coord_info['lon']
        group_sorted = group.sort_values('year')
        historial = dict(zip(group_sorted['year'].astype(int), group_sorted['AverageTemperature'].astype(float)))

        cache_key = f"{city}_{round(lat, 2)}_{round(lon, 2)}"
        if cache_key in OPEN_METEO_CACHE:
            historial.update({int(k): v for k, v in OPEN_METEO_CACHE[cache_key].items()})

        x_prev = np.array([y for y in historial.keys() if y >= 1950], dtype=float)
        y_prev = np.array([historial[int(y)] for y in x_prev], dtype=float)
        if len(x_prev) >= 5:
            fit_prev = np.polyfit(x_prev, y_prev, deg=1)
            for y_fill in range(2014, CURRENT_YEAR + 1):
                if y_fill not in historial:
                    historial[y_fill] = round(float(fit_prev[1] + fit_prev[0] * y_fill), 2)

        # Regresión sobre la serie completa hasta CURRENT_YEAR
        x_all = np.array(sorted(historial.keys()), dtype=float)
        y_all = np.array([historial[int(y)] for y in x_all], dtype=float)
        mask_recent = x_all >= 1950
        coeffs = np.polyfit(x_all[mask_recent], y_all[mask_recent], deg=1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        calentamiento_decada = round(slope * 10, 3)

        # Cálculo de Indicadores ASHRAE y Energía
        # Media histórica base
        temps_hist = [historial[int(y)] for y in x_all if y <= CURRENT_YEAR]
        media_hist = round(float(np.mean(temps_hist)), 2) if temps_hist else 0.0
        cdd_hist_base = round(float(np.mean([calcular_cdd(t) for t in temps_hist])), 2)

        # Temperatura proyectada a 2040 y delta
        temp_proy_2040 = round(float(intercept + slope * HORIZONTE_PROYECCION), 2)
        temp_actual = historial.get(CURRENT_YEAR, round(float(intercept + slope * CURRENT_YEAR), 2))
        delta_temp_2040 = round(temp_proy_2040 - temp_actual, 2)
        cdd_proy_2040 = calcular_cdd(temp_proy_2040)

        # Porcentaje de incremento en CDD
        if cdd_hist_base > 0.01:
            pct_aumento_cdd = round(((cdd_proy_2040 - cdd_hist_base) / cdd_hist_base) * 100.0, 1)
        else:
            pct_aumento_cdd = round((cdd_proy_2040 * 10.0), 1)

        riesgo_info = clasificar_riesgo_energetico(calentamiento_decada, pct_aumento_cdd)

        meta_ciudades[city] = {
            "country": coord_info['Country'],
            "lat": lat,
            "lon": lon,
            "historial": historial,
            "slope": slope,
            "calentamiento_decada": calentamiento_decada,
            "intercept": intercept,
            "media_hist": media_hist,
            "cdd_hist_base": cdd_hist_base,
            "temp_actual": temp_actual,
            "temp_proy_2040": temp_proy_2040,
            "delta_temp_2040": delta_temp_2040,
            "cdd_proy_2040": cdd_proy_2040,
            "pct_aumento_cdd": pct_aumento_cdd,
            "riesgo": riesgo_info
        }

    METADATA_CIUDADES = meta_ciudades
    ANOS_HISTORICOS = sorted(list(todos_los_anos_reales))
    PROYECCION_ANOS = list(range(CURRENT_YEAR + 1, HORIZONTE_PROYECCION + 1))

    # Promedio global consolidado
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

    print(f"[OK] ClimateAI inicializado. {len(METADATA_CIUDADES)} ciudades clasificadas con semáforo energético ASHRAE.")

inicializar_dataset()

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html no encontrado")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/config")
def get_config():
    """Retorna lista de ciudades, años reales, horizonte y semáforos de riesgo."""
    ciudades_lista = [
        {
            "city": c,
            "country": data["country"],
            "lat": data["lat"],
            "lon": data["lon"],
            "riesgo": data["riesgo"]
        }
        for c, data in sorted(METADATA_CIUDADES.items())
    ]
    return JSONResponse(content={
        "current_year": CURRENT_YEAR,
        "horizonte_proyeccion": HORIZONTE_PROYECCION,
        "anos_historicos": ANOS_HISTORICOS,
        "min_ano": min(ANOS_HISTORICOS),
        "max_ano_hist": CURRENT_YEAR,
        "anos_proyeccion": PROYECCION_ANOS,
        "max_ano_proy": HORIZONTE_PROYECCION,
        "ciudades": ciudades_lista
    })

@app.get("/api/thermal-timeline")
def get_thermal_timeline():
    """
    Retorna la cuadrícula térmica completa indexada por año { "1850": [[lat, lon, peso], ...], ... }
    con intensidades normalizadas calculadas según las directrices:
    - Temperaturas frías (< 10°C): 0.1 - 0.3
    - Temperaturas templadas (10°C - 25°C): 0.4 - 0.6
    - Temperaturas extremas (> 28°C / alto CDD): 0.7 - 1.0
    """
    timeline_dict = {}
    rango_completo = ANOS_HISTORICOS + PROYECCION_ANOS

    def calcular_peso_normalizado(temp: float) -> float:
        if temp < 10.0:
            # Rango 0.1 a 0.35
            return round(max(0.1, min(0.35, 0.1 + (temp - (-5.0)) / 15.0 * 0.25)), 2)
        elif temp <= 26.0:
            # Rango 0.36 a 0.68
            return round(0.36 + (temp - 10.0) / 16.0 * 0.32, 2)
        else:
            # Rango 0.70 a 1.00
            return round(min(1.0, 0.70 + (temp - 26.0) / 14.0 * 0.30), 2)

    for y in rango_completo:
        puntos_ano = []
        es_proyeccion = y > CURRENT_YEAR

        for city, data in METADATA_CIUDADES.items():
            if not es_proyeccion:
                t = data["historial"].get(y)
                if t is None:
                    t = round(data["intercept"] + data["slope"] * y, 2)
            else:
                t = round(data["intercept"] + data["slope"] * y, 2)

            peso = calcular_peso_normalizado(t)
            puntos_ano.append({
                "lat": data["lat"],
                "lon": data["lon"],
                "temperatura": t,          # Clave canónica usada por el frontend (igual que /api/heatmap-year)
                "temp": t,                 # Alias retrocompatible
                "cdd": calcular_cdd(t),
                "intensidad": peso,
                "city": city,
                "country": data["country"],
                "riesgo": data["riesgo"]
            })

        timeline_dict[str(y)] = puntos_ano

    return JSONResponse(content={
        "min_year": min(rango_completo),
        "max_year": max(rango_completo),
        "current_year": CURRENT_YEAR,
        "timeline": timeline_dict
    })

@app.get("/api/heatmap-year")
def get_heatmap_year(year: int = Query(...)):
    """Retorna las ciudades con su temperatura, CDD e indicador de semáforo para el año seleccionado."""
    es_proyeccion = year > CURRENT_YEAR
    puntos = []

    for city, data in METADATA_CIUDADES.items():
        if not es_proyeccion:
            temp = data["historial"].get(year)
            if temp is None:
                temp = round(data["intercept"] + data["slope"] * year, 2)
        else:
            temp = round(data["intercept"] + data["slope"] * year, 2)

        cdd_val = calcular_cdd(temp)

        puntos.append({
            "city": city,
            "country": data["country"],
            "lat": data["lat"],
            "lon": data["lon"],
            "temperatura": temp,
            "cdd": cdd_val,
            "es_proyeccion": es_proyeccion,
            "riesgo": data["riesgo"]
        })

    temps = [p["temperatura"] for p in puntos if p["temperatura"] is not None]
    cdds = [p["cdd"] for p in puntos if p["cdd"] is not None]

    kpis = {
        "year": year,
        "current_year": CURRENT_YEAR,
        "es_proyeccion": es_proyeccion,
        "total_ciudades": len(puntos),
        "temp_promedio": round(float(np.mean(temps)), 2) if temps else 0.0,
        "cdd_promedio": round(float(np.mean(cdds)), 2) if cdds else 0.0,
        "temp_maxima": round(float(np.max(temps)), 2) if temps else 0.0,
        "ciudad_mas_calida": max(puntos, key=lambda x: x["temperatura"]) if puntos else None
    }

    return JSONResponse(content={
        "year": year,
        "current_year": CURRENT_YEAR,
        "es_proyeccion": es_proyeccion,
        "kpis": kpis,
        "puntos": puntos
    })

@app.get("/api/city-detail")
def get_city_detail(city: str = Query(...)):
    """
    Retorna la ficha técnica ejecutiva completa de resiliencia energética:
    - Indicadores ASHRAE (CDD histórico vs 2040, % aumento climatización).
    - Semáforo de riesgo y nivel de estrés de transformadores.
    - Medidas prescriptivas y decisiones operativas urbanas.
    - Serie dual de temperatura y evolución CDD año con año.
    """
    if city not in METADATA_CIUDADES:
        meta = {
            "country": "Global",
            "lat": 20.0,
            "lon": 0.0,
            "slope": GLOBAL_MODEL["slope"],
            "intercept": GLOBAL_MODEL["intercept"],
            "calentamiento_decada": GLOBAL_MODEL["calentamiento_decada"],
            "media_hist": GLOBAL_MODEL["media_hist"],
            "cdd_hist_base": GLOBAL_MODEL["cdd_hist_base"],
            "temp_actual": GLOBAL_MODEL["temp_actual"],
            "temp_proy_2040": GLOBAL_MODEL["temp_proy_2040"],
            "delta_temp_2040": GLOBAL_MODEL["delta_temp_2040"],
            "cdd_proy_2040": GLOBAL_MODEL["cdd_proy_2040"],
            "pct_aumento_cdd": GLOBAL_MODEL["pct_aumento_cdd"],
            "riesgo": GLOBAL_MODEL["riesgo"]
        }
        historial_dict = GLOBAL_HISTORIAL.copy()
    else:
        meta = METADATA_CIUDADES[city]
        meteo_data = fetch_open_meteo_anual(city, meta["lat"], meta["lon"], start_year=2014, end_year=CURRENT_YEAR)
        if meteo_data:
            meta["historial"].update(meteo_data)
        historial_dict = meta["historial"]

    # Histórico con CDD anual
    serie_historica = []
    for y in sorted(historial_dict.keys()):
        if y <= CURRENT_YEAR:
            t_val = historial_dict[y]
            serie_historica.append({
                "year": y,
                "temperatura": t_val,
                "cdd": calcular_cdd(t_val)
            })

    # Proyección con CDD anual
    serie_proyeccion = []
    slope = meta["slope"]
    intercept = meta["intercept"]

    for y in PROYECCION_ANOS:
        pred_val = round(intercept + slope * y, 2)
        diff_years = y - CURRENT_YEAR
        serie_proyeccion.append({
            "year": y,
            "prediccion": pred_val,
            "cdd_proy": calcular_cdd(pred_val),
            "lim_sup": round(pred_val + (0.04 * diff_years), 2),
            "lim_inf": round(pred_val - (0.04 * diff_years), 2)
        })

    # Medidas prescriptivas adaptadas al nivel de riesgo y perfil geoclimático de la urbe
    perfil_clima = clasificar_perfil_geoclimatico(
        city,
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
        city if city else "Global"
    )

    return JSONResponse(content={
        "city": city,
        "country": meta["country"],
        "lat": meta["lat"],
        "lon": meta["lon"],
        "perfil_clima": perfil_clima,
        "current_year": CURRENT_YEAR,
        "horizonte": HORIZONTE_PROYECCION,
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
    })

if __name__ == "__main__":
    import signal
    import sys
    import threading
    import webbrowser
    import uvicorn
    import socket

    HOST = "127.0.0.1"

    # Seleccionar puerto disponible (8080 o 8000)
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((HOST, port)) == 0

    PORT = 8080
    if is_port_in_use(8080) and not is_port_in_use(8000):
        PORT = 8000

    URL = f"http://localhost:{PORT}"

    def signal_handler(sig, frame):
        print("\n\n[ClimateAI] Cerrando servidor limpiamente y liberando puertos...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    try:
        signal.signal(signal.SIGTERM, signal_handler)
    except (AttributeError, ValueError):
        pass

    def open_browser():
        try:
            print(f"[ClimateAI] Abriendo navegador predeterminado en {URL} ...")
            webbrowser.open_new(URL)
        except Exception as e:
            print(f"[ClimateAI] Aviso al abrir navegador: {e}")

    # Abrir el navegador en hilo independiente tras breve retardo de arranque
    threading.Timer(1.2, open_browser).start()

    print(f"\n========================================================")
    print(f" ClimateAI: Servidor iniciado en {URL}")
    print(f" Presiona Ctrl + C para detener y liberar el puerto.")
    print(f"========================================================\n")

    # Ejecución en primer plano para detenerse limpiamente al cerrar la consola o VS Code
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)

