"""
URBAN-HEAT GRID INTELLIGENCE — Servicio GIS de Nodos Jerárquicos Progresivos por Viewport (LOD)
Archivo: services/viewport_cities.py
Provee renderizado jerárquico progresivo por nivel de zoom (LOD) para el Modo Live Global:
  - zoom <= 5: ~200 capitales y megaciudades globales en memoria (GLOBAL_MAJOR_CITIES, máx. 55 nodos).
  - zoom 6-7: Ciudades principales vía Overpass API (place="city", máx. 65 nodos) + GLOBAL_MAJOR_CITIES.
  - zoom 8-10: Ciudades y cabeceras municipales vía Overpass API (place~"^(city|town)$", máx. 75 nodos).
  - zoom >= 11: Ciudades, municipios, pueblos y comunidades vía Overpass API (place~"^(city|town|village|hamlet)$", máx. 85 nodos).
Incluye decluttering espacial por celda proporcional al zoom, caché persistente SQLite no bloqueante
(viewport_nodes_cache) y asignación térmica batch (Open-Meteo en 2026 + motor ML en cascada).
"""
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pipeline.model import _ENGINE

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DB_PATH: Path = BASE_DIR / "data" / "geo_cache.db"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_DB_LOCK = threading.Lock()
_L1_VIEWPORT_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_LIVE_TEMP_CACHE: Dict[Tuple[float, float], Tuple[float, float]] = {}  # (lat_005, lon_005) -> (temp_c, ts_epoch)
LIVE_TEMP_TTL_SEC = 900.0  # 15 minutos

TIER_PRIORITY = {
    "megacity": 0,
    "city": 1,
    "town": 2,
    "village": 3,
    "hamlet": 4,
}

# Catálogo base en memoria de ~200 capitales, megaciudades y centros regionales estratégicos del mundo
GLOBAL_MAJOR_CITIES: List[Dict[str, Any]] = [
    # Norteamérica y México / Península de Yucatán / Centroamérica / Caribe
    {"name": "Mexico City", "lat": 19.4326, "lon": -99.1332, "tier": "megacity"},
    {"name": "Guadalajara", "lat": 20.6597, "lon": -103.3496, "tier": "city"},
    {"name": "Monterrey", "lat": 25.6866, "lon": -100.3161, "tier": "city"},
    {"name": "Puebla", "lat": 19.0414, "lon": -98.2063, "tier": "city"},
    {"name": "Tijuana", "lat": 32.5149, "lon": -117.0382, "tier": "city"},
    {"name": "Cancún", "lat": 21.1619, "lon": -86.8515, "tier": "city"},
    {"name": "Mérida", "lat": 20.9674, "lon": -89.5926, "tier": "city"},
    {"name": "Tulum", "lat": 20.2114, "lon": -87.4654, "tier": "town"},
    {"name": "Playa del Carmen", "lat": 20.6296, "lon": -87.0739, "tier": "city"},
    {"name": "Cozumel", "lat": 20.5083, "lon": -86.9458, "tier": "town"},
    {"name": "Valladolid", "lat": 20.6896, "lon": -88.2022, "tier": "town"},
    {"name": "Progreso", "lat": 21.2812, "lon": -89.6652, "tier": "town"},
    {"name": "Chetumal", "lat": 18.5001, "lon": -88.2961, "tier": "city"},
    {"name": "Campeche", "lat": 19.8301, "lon": -90.5349, "tier": "city"},
    {"name": "Puerto Morelos", "lat": 20.8477, "lon": -86.8756, "tier": "town"},
    {"name": "Isla Mujeres", "lat": 21.2322, "lon": -86.7341, "tier": "town"},
    {"name": "Bacalar", "lat": 18.6783, "lon": -88.3924, "tier": "town"},
    {"name": "Felipe Carrillo Puerto", "lat": 19.5775, "lon": -88.0453, "tier": "town"},
    {"name": "Tizimín", "lat": 21.1456, "lon": -88.1514, "tier": "town"},
    {"name": "Izamal", "lat": 20.9333, "lon": -89.0167, "tier": "town"},
    {"name": "Motul", "lat": 21.0958, "lon": -89.2833, "tier": "town"},
    {"name": "Tekax", "lat": 20.2058, "lon": -89.2856, "tier": "town"},
    {"name": "Ciudad del Carmen", "lat": 18.6414, "lon": -91.8297, "tier": "city"},
    {"name": "Oaxaca", "lat": 17.0732, "lon": -96.7266, "tier": "city"},
    {"name": "León", "lat": 21.1221, "lon": -101.6826, "tier": "city"},
    {"name": "Querétaro", "lat": 20.5888, "lon": -100.3899, "tier": "city"},
    {"name": "Chihuahua", "lat": 28.6353, "lon": -106.0889, "tier": "city"},
    {"name": "Hermosillo", "lat": 29.0729, "lon": -110.9559, "tier": "city"},
    {"name": "Veracruz", "lat": 19.1738, "lon": -96.1342, "tier": "city"},
    {"name": "Villahermosa", "lat": 17.9895, "lon": -92.9475, "tier": "city"},
    {"name": "Acapulco", "lat": 16.8531, "lon": -99.8237, "tier": "city"},
    {"name": "Culiacán", "lat": 24.8091, "lon": -107.3940, "tier": "city"},
    {"name": "San Luis Potosí", "lat": 22.1565, "lon": -100.9855, "tier": "city"},
    {"name": "Torreón", "lat": 25.5428, "lon": -103.4068, "tier": "city"},
    {"name": "Tuxtla Gutiérrez", "lat": 16.7516, "lon": -93.1029, "tier": "city"},
    {"name": "La Paz", "lat": 24.1426, "lon": -110.3128, "tier": "city"},
    {"name": "New York", "lat": 40.7128, "lon": -74.0060, "tier": "megacity"},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "tier": "megacity"},
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298, "tier": "megacity"},
    {"name": "Houston", "lat": 29.7604, "lon": -95.3698, "tier": "megacity"},
    {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740, "tier": "megacity"},
    {"name": "Miami", "lat": 25.7617, "lon": -80.1918, "tier": "megacity"},
    {"name": "Dallas", "lat": 32.7767, "lon": -96.7970, "tier": "megacity"},
    {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194, "tier": "megacity"},
    {"name": "Seattle", "lat": 47.6062, "lon": -122.3321, "tier": "megacity"},
    {"name": "Denver", "lat": 39.7392, "lon": -104.9903, "tier": "megacity"},
    {"name": "Atlanta", "lat": 33.7490, "lon": -84.3880, "tier": "megacity"},
    {"name": "Washington D.C.", "lat": 38.9072, "lon": -77.0369, "tier": "megacity"},
    {"name": "Boston", "lat": 42.3601, "lon": -71.0589, "tier": "megacity"},
    {"name": "Toronto", "lat": 43.6532, "lon": -79.3832, "tier": "megacity"},
    {"name": "Montreal", "lat": 45.5017, "lon": -73.5673, "tier": "megacity"},
    {"name": "Vancouver", "lat": 49.2827, "lon": -123.1207, "tier": "megacity"},
    {"name": "Calgary", "lat": 51.0447, "lon": -114.0719, "tier": "megacity"},
    {"name": "Ottawa", "lat": 45.4215, "lon": -75.6972, "tier": "megacity"},
    {"name": "Havana", "lat": 23.1136, "lon": -82.3666, "tier": "megacity"},
    {"name": "Santo Domingo", "lat": 18.4861, "lon": -69.9312, "tier": "megacity"},
    {"name": "San Juan", "lat": 18.4655, "lon": -66.1057, "tier": "megacity"},
    {"name": "Guatemala City", "lat": 14.6349, "lon": -90.5069, "tier": "megacity"},
    {"name": "San Salvador", "lat": 13.6929, "lon": -89.2182, "tier": "megacity"},
    {"name": "Tegucigalpa", "lat": 14.0723, "lon": -87.1921, "tier": "megacity"},
    {"name": "Managua", "lat": 12.1150, "lon": -86.2362, "tier": "megacity"},
    {"name": "San José", "lat": 9.9281, "lon": -84.0907, "tier": "megacity"},
    {"name": "Panama City", "lat": 8.9824, "lon": -79.5199, "tier": "megacity"},
    {"name": "Kingston", "lat": 17.9712, "lon": -76.7936, "tier": "megacity"},
    # Sudamérica
    {"name": "Bogotá", "lat": 4.7110, "lon": -74.0721, "tier": "megacity"},
    {"name": "Medellín", "lat": 6.2442, "lon": -75.5812, "tier": "megacity"},
    {"name": "Cali", "lat": 3.4516, "lon": -76.5320, "tier": "megacity"},
    {"name": "Barranquilla", "lat": 10.9685, "lon": -74.7813, "tier": "megacity"},
    {"name": "Caracas", "lat": 10.4806, "lon": -66.9036, "tier": "megacity"},
    {"name": "Quito", "lat": -0.1807, "lon": -78.4678, "tier": "megacity"},
    {"name": "Guayaquil", "lat": -2.1894, "lon": -79.8891, "tier": "megacity"},
    {"name": "Lima", "lat": -12.0464, "lon": -77.0428, "tier": "megacity"},
    {"name": "Arequipa", "lat": -16.4090, "lon": -71.5375, "tier": "megacity"},
    {"name": "La Paz (Bolivia)", "lat": -16.4897, "lon": -68.1193, "tier": "megacity"},
    {"name": "Santa Cruz", "lat": -17.7833, "lon": -63.1821, "tier": "megacity"},
    {"name": "Santiago", "lat": -33.4489, "lon": -70.6693, "tier": "megacity"},
    {"name": "Buenos Aires", "lat": -34.6037, "lon": -58.3816, "tier": "megacity"},
    {"name": "Córdoba", "lat": -31.4201, "lon": -64.1888, "tier": "megacity"},
    {"name": "Montevideo", "lat": -34.9011, "lon": -56.1645, "tier": "megacity"},
    {"name": "Asunción", "lat": -25.2637, "lon": -57.5759, "tier": "megacity"},
    {"name": "São Paulo", "lat": -23.5505, "lon": -46.6333, "tier": "megacity"},
    {"name": "Rio de Janeiro", "lat": -22.9068, "lon": -43.1729, "tier": "megacity"},
    {"name": "Brasília", "lat": -15.7975, "lon": -47.8919, "tier": "megacity"},
    {"name": "Salvador", "lat": -12.9777, "lon": -38.5016, "tier": "megacity"},
    {"name": "Fortaleza", "lat": -3.7172, "lon": -38.5433, "tier": "megacity"},
    {"name": "Belo Horizonte", "lat": -19.9167, "lon": -43.9345, "tier": "megacity"},
    {"name": "Manaus", "lat": -3.1190, "lon": -60.0217, "tier": "megacity"},
    {"name": "Recife", "lat": -8.0476, "lon": -34.8770, "tier": "megacity"},
    {"name": "Porto Alegre", "lat": -30.0346, "lon": -51.2177, "tier": "megacity"},
    # Europa
    {"name": "London", "lat": 51.5074, "lon": -0.1278, "tier": "megacity"},
    {"name": "Manchester", "lat": 53.4808, "lon": -2.2426, "tier": "megacity"},
    {"name": "Dublin", "lat": 53.3498, "lon": -6.2603, "tier": "megacity"},
    {"name": "Paris", "lat": 48.8566, "lon": 2.3522, "tier": "megacity"},
    {"name": "Lyon", "lat": 45.7640, "lon": 4.8357, "tier": "megacity"},
    {"name": "Marseille", "lat": 43.2965, "lon": 5.3698, "tier": "megacity"},
    {"name": "Madrid", "lat": 40.4168, "lon": -3.7038, "tier": "megacity"},
    {"name": "Barcelona", "lat": 41.3874, "lon": 2.1686, "tier": "megacity"},
    {"name": "Lisbon", "lat": 38.7223, "lon": -9.1393, "tier": "megacity"},
    {"name": "Rome", "lat": 41.9028, "lon": 12.4964, "tier": "megacity"},
    {"name": "Milan", "lat": 45.4642, "lon": 9.1900, "tier": "megacity"},
    {"name": "Naples", "lat": 40.8518, "lon": 14.2681, "tier": "megacity"},
    {"name": "Berlin", "lat": 52.5200, "lon": 13.4050, "tier": "megacity"},
    {"name": "Munich", "lat": 48.1351, "lon": 11.5820, "tier": "megacity"},
    {"name": "Frankfurt", "lat": 50.1109, "lon": 8.6821, "tier": "megacity"},
    {"name": "Hamburg", "lat": 53.5511, "lon": 9.9937, "tier": "megacity"},
    {"name": "Amsterdam", "lat": 52.3676, "lon": 4.9041, "tier": "megacity"},
    {"name": "Brussels", "lat": 50.8503, "lon": 4.3517, "tier": "megacity"},
    {"name": "Zurich", "lat": 47.3769, "lon": 8.5417, "tier": "megacity"},
    {"name": "Vienna", "lat": 48.2082, "lon": 16.3738, "tier": "megacity"},
    {"name": "Prague", "lat": 50.0755, "lon": 14.4378, "tier": "megacity"},
    {"name": "Warsaw", "lat": 52.2297, "lon": 21.0122, "tier": "megacity"},
    {"name": "Budapest", "lat": 47.4979, "lon": 19.0402, "tier": "megacity"},
    {"name": "Bucharest", "lat": 44.4268, "lon": 26.1025, "tier": "megacity"},
    {"name": "Athens", "lat": 37.9838, "lon": 23.7275, "tier": "megacity"},
    {"name": "Istanbul", "lat": 41.0082, "lon": 28.9784, "tier": "megacity"},
    {"name": "Ankara", "lat": 39.9334, "lon": 32.8597, "tier": "megacity"},
    {"name": "Stockholm", "lat": 59.3293, "lon": 18.0686, "tier": "megacity"},
    {"name": "Oslo", "lat": 59.9139, "lon": 10.7522, "tier": "megacity"},
    {"name": "Copenhagen", "lat": 55.6761, "lon": 12.5683, "tier": "megacity"},
    {"name": "Helsinki", "lat": 60.1699, "lon": 24.9384, "tier": "megacity"},
    {"name": "Kyiv", "lat": 50.4501, "lon": 30.5234, "tier": "megacity"},
    {"name": "Moscow", "lat": 55.7558, "lon": 37.6173, "tier": "megacity"},
    {"name": "Saint Petersburg", "lat": 59.9311, "lon": 30.3609, "tier": "megacity"},
    {"name": "Belgrade", "lat": 44.7866, "lon": 20.4489, "tier": "megacity"},
    {"name": "Sofia", "lat": 42.6977, "lon": 23.3219, "tier": "megacity"},
    # Medio Oriente y Asia Central
    {"name": "Cairo", "lat": 30.0444, "lon": 31.2357, "tier": "megacity"},
    {"name": "Alexandria", "lat": 31.2001, "lon": 29.9187, "tier": "megacity"},
    {"name": "Baghdad", "lat": 33.3152, "lon": 44.3661, "tier": "megacity"},
    {"name": "Riyadh", "lat": 24.7136, "lon": 46.6753, "tier": "megacity"},
    {"name": "Jeddah", "lat": 21.4858, "lon": 39.1925, "tier": "megacity"},
    {"name": "Dubai", "lat": 25.2048, "lon": 55.2708, "tier": "megacity"},
    {"name": "Abu Dhabi", "lat": 24.4539, "lon": 54.3773, "tier": "megacity"},
    {"name": "Doha", "lat": 25.2854, "lon": 51.5310, "tier": "megacity"},
    {"name": "Kuwait City", "lat": 29.3759, "lon": 47.9774, "tier": "megacity"},
    {"name": "Muscat", "lat": 23.5880, "lon": 58.3829, "tier": "megacity"},
    {"name": "Tehran", "lat": 35.6892, "lon": 51.3890, "tier": "megacity"},
    {"name": "Tel Aviv", "lat": 32.0853, "lon": 34.7818, "tier": "megacity"},
    {"name": "Amman", "lat": 31.9454, "lon": 35.9284, "tier": "megacity"},
    {"name": "Beirut", "lat": 33.8938, "lon": 35.5018, "tier": "megacity"},
    {"name": "Damascus", "lat": 33.5138, "lon": 36.2765, "tier": "megacity"},
    {"name": "Baku", "lat": 40.4093, "lon": 49.8671, "tier": "megacity"},
    {"name": "Tashkent", "lat": 41.2995, "lon": 69.2401, "tier": "megacity"},
    {"name": "Almaty", "lat": 43.2220, "lon": 76.8512, "tier": "megacity"},
    {"name": "Astana", "lat": 51.1694, "lon": 71.4491, "tier": "megacity"},
    # África
    {"name": "Casablanca", "lat": 33.5731, "lon": -7.5898, "tier": "megacity"},
    {"name": "Algiers", "lat": 36.7538, "lon": 3.0588, "tier": "megacity"},
    {"name": "Tunis", "lat": 36.8065, "lon": 10.1815, "tier": "megacity"},
    {"name": "Tripoli", "lat": 32.8872, "lon": 13.1913, "tier": "megacity"},
    {"name": "Khartoum", "lat": 15.5007, "lon": 32.5599, "tier": "megacity"},
    {"name": "Addis Ababa", "lat": 9.0320, "lon": 38.7469, "tier": "megacity"},
    {"name": "Nairobi", "lat": -1.2921, "lon": 36.8219, "tier": "megacity"},
    {"name": "Dar es Salaam", "lat": -6.7924, "lon": 39.2083, "tier": "megacity"},
    {"name": "Kampala", "lat": 0.3476, "lon": 32.5825, "tier": "megacity"},
    {"name": "Lagos", "lat": 6.5244, "lon": 3.3792, "tier": "megacity"},
    {"name": "Abuja", "lat": 9.0765, "lon": 7.3986, "tier": "megacity"},
    {"name": "Accra", "lat": 5.6037, "lon": -0.1870, "tier": "megacity"},
    {"name": "Abidjan", "lat": 5.3600, "lon": -4.0083, "tier": "megacity"},
    {"name": "Dakar", "lat": 14.7167, "lon": -17.4677, "tier": "megacity"},
    {"name": "Kinshasa", "lat": -4.4419, "lon": 15.2663, "tier": "megacity"},
    {"name": "Luanda", "lat": -8.8390, "lon": 13.2894, "tier": "megacity"},
    {"name": "Johannesburg", "lat": -26.2041, "lon": 28.0473, "tier": "megacity"},
    {"name": "Cape Town", "lat": -33.9249, "lon": 18.4241, "tier": "megacity"},
    {"name": "Durban", "lat": -29.8587, "lon": 31.0218, "tier": "megacity"},
    {"name": "Harare", "lat": -17.8252, "lon": 31.0335, "tier": "megacity"},
    {"name": "Maputo", "lat": -25.9692, "lon": 32.5732, "tier": "megacity"},
    {"name": "Antananarivo", "lat": -18.8792, "lon": 47.5079, "tier": "megacity"},
    # Asia del Sur, Este y Sudeste Asiático
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011, "tier": "megacity"},
    {"name": "Lahore", "lat": 31.5204, "lon": 74.3587, "tier": "megacity"},
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479, "tier": "megacity"},
    {"name": "Kabul", "lat": 34.5553, "lon": 69.2075, "tier": "megacity"},
    {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090, "tier": "megacity"},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "tier": "megacity"},
    {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "tier": "megacity"},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "tier": "megacity"},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "tier": "megacity"},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "tier": "megacity"},
    {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "tier": "megacity"},
    {"name": "Dhaka", "lat": 23.8103, "lon": 90.4125, "tier": "megacity"},
    {"name": "Colombo", "lat": 6.9271, "lon": 79.8612, "tier": "megacity"},
    {"name": "Kathmandu", "lat": 27.7172, "lon": 85.3240, "tier": "megacity"},
    {"name": "Yangon", "lat": 16.8409, "lon": 96.1735, "tier": "megacity"},
    {"name": "Bangkok", "lat": 13.7563, "lon": 100.5018, "tier": "megacity"},
    {"name": "Kuala Lumpur", "lat": 3.1390, "lon": 101.6869, "tier": "megacity"},
    {"name": "Singapore", "lat": 1.3521, "lon": 103.8198, "tier": "megacity"},
    {"name": "Jakarta", "lat": -6.2088, "lon": 106.8456, "tier": "megacity"},
    {"name": "Surabaya", "lat": -7.2575, "lon": 112.7521, "tier": "megacity"},
    {"name": "Manila", "lat": 14.5995, "lon": 120.9842, "tier": "megacity"},
    {"name": "Ho Chi Minh City", "lat": 10.8231, "lon": 106.6297, "tier": "megacity"},
    {"name": "Hanoi", "lat": 21.0278, "lon": 105.8342, "tier": "megacity"},
    {"name": "Phnom Penh", "lat": 11.5564, "lon": 104.9282, "tier": "megacity"},
    {"name": "Beijing", "lat": 39.9042, "lon": 116.4074, "tier": "megacity"},
    {"name": "Shanghai", "lat": 31.2304, "lon": 121.4737, "tier": "megacity"},
    {"name": "Guangzhou", "lat": 23.1291, "lon": 113.2644, "tier": "megacity"},
    {"name": "Shenzhen", "lat": 22.5431, "lon": 114.0579, "tier": "megacity"},
    {"name": "Chengdu", "lat": 30.5728, "lon": 104.0668, "tier": "megacity"},
    {"name": "Chongqing", "lat": 29.4316, "lon": 106.9123, "tier": "megacity"},
    {"name": "Wuhan", "lat": 30.5928, "lon": 114.3055, "tier": "megacity"},
    {"name": "Xi'an", "lat": 34.3416, "lon": 108.9398, "tier": "megacity"},
    {"name": "Hong Kong", "lat": 22.3193, "lon": 114.1694, "tier": "megacity"},
    {"name": "Taipei", "lat": 25.0330, "lon": 121.5654, "tier": "megacity"},
    {"name": "Seoul", "lat": 37.5665, "lon": 126.9780, "tier": "megacity"},
    {"name": "Busan", "lat": 35.1796, "lon": 129.0756, "tier": "megacity"},
    {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503, "tier": "megacity"},
    {"name": "Osaka", "lat": 34.6937, "lon": 135.5023, "tier": "megacity"},
    {"name": "Sapporo", "lat": 43.0618, "lon": 141.3545, "tier": "megacity"},
    {"name": "Ulaanbaatar", "lat": 47.8864, "lon": 106.9057, "tier": "megacity"},
    # Oceanía
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093, "tier": "megacity"},
    {"name": "Melbourne", "lat": -37.8136, "lon": 144.9631, "tier": "megacity"},
    {"name": "Brisbane", "lat": -27.4705, "lon": 153.0260, "tier": "megacity"},
    {"name": "Perth", "lat": -31.9505, "lon": 115.8605, "tier": "megacity"},
    {"name": "Auckland", "lat": -36.8485, "lon": 174.7633, "tier": "megacity"},
    {"name": "Wellington", "lat": -41.2866, "lon": 174.7756, "tier": "megacity"},
    {"name": "Port Moresby", "lat": -9.4438, "lon": 147.1803, "tier": "megacity"},
    {"name": "Suva", "lat": -18.1416, "lon": 178.4419, "tier": "megacity"},
]


def _init_viewport_cache_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _DB_LOCK:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS viewport_nodes_cache (
                    tile_key TEXT PRIMARY KEY,
                    nodes_json TEXT,
                    timestamp DATETIME
                )
                """
            )
            conn.commit()


def _read_viewport_cache_sync(tile_key: str) -> Optional[List[Dict[str, Any]]]:
    _init_viewport_cache_db()
    with _DB_LOCK:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            row = conn.execute(
                "SELECT nodes_json FROM viewport_nodes_cache WHERE tile_key = ?",
                (tile_key,),
            ).fetchone()
            if row and row[0]:
                try:
                    data = json.loads(row[0])
                    if isinstance(data, list) and len(data) > 0:
                        return data
                except Exception:
                    pass
    return None


def _write_viewport_cache_sync(tile_key: str, nodes: List[Dict[str, Any]]) -> None:
    if not nodes:
        return
    _init_viewport_cache_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    payload_str = json.dumps(nodes, ensure_ascii=False)
    with _DB_LOCK:
        with sqlite3.connect(DB_PATH, timeout=10.0) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO viewport_nodes_cache (tile_key, nodes_json, timestamp) VALUES (?, ?, ?)",
                (tile_key, payload_str, now_iso),
            )
            conn.commit()


def _make_tile_key(south: float, west: float, north: float, east: float, zoom: int) -> str:
    z = max(1, min(18, int(zoom)))
    precision = 0 if z <= 5 else (1 if z <= 8 else 2)
    return (
        f"z{z}:"
        f"{round(float(south), precision)}:"
        f"{round(float(west), precision)}:"
        f"{round(float(north), precision)}:"
        f"{round(float(east), precision)}"
    )


def _in_bbox(lat: float, lon: float, south: float, west: float, north: float, east: float) -> bool:
    if not (south <= lat <= north):
        return False
    if west <= east:
        return west <= lon <= east
    # Cruce del antimeridiano
    return lon >= west or lon <= east


def _declutter_nodes(nodes: List[Dict[str, Any]], zoom: int, max_nodes: int) -> List[Dict[str, Any]]:
    """
    Aplica decluttering espacial por celda de grilla proporcional al zoom:
    cell_size = 180.0 / (2 ** zoom), reteniendo el nodo de mayor jerarquía por celda.
    """
    z = max(1, min(18, int(zoom)))
    cell_size = max(0.002, 180.0 / (2 ** z))

    ordered = sorted(
        nodes,
        key=lambda n: (
            TIER_PRIORITY.get(str(n.get("tier", "city")).lower(), 2),
            -float(n.get("population", 0) or 0),
        ),
    )

    grid_seen: Dict[Tuple[int, int], Dict[str, Any]] = {}
    name_seen = set()

    for node in ordered:
        lat = float(node["lat"])
        lon = float(node["lon"])
        name_key = str(node["name"]).strip().lower()
        if not name_key or name_key in name_seen:
            continue
        cell = (
            int(math.floor((lat + 90.0) / cell_size)),
            int(math.floor((lon + 180.0) / cell_size)),
        )
        if cell not in grid_seen:
            grid_seen[cell] = {
                "name": str(node["name"]),
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "tier": str(node.get("tier", "city")),
            }
            name_seen.add(name_key)
            if len(grid_seen) >= max_nodes:
                break

    return list(grid_seen.values())


async def _query_overpass_nodes(
    south: float,
    west: float,
    north: float,
    east: float,
    place_filter: str,
    limit: int,
) -> List[Dict[str, Any]]:
    s_clamp = max(-85.0, min(85.0, float(south)))
    n_clamp = max(-85.0, min(85.0, float(north)))
    w_clamp = max(-180.0, min(180.0, float(west)))
    e_clamp = max(-180.0, min(180.0, float(east)))
    if s_clamp >= n_clamp or w_clamp >= e_clamp:
        return []

    query = f'[out:json][timeout:5];node["place"{place_filter}]({s_clamp:.4f},{w_clamp:.4f},{n_clamp:.4f},{e_clamp:.4f});out body {int(limit)};'
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            payload = resp.json()
            elements = payload.get("elements") or []
            parsed: List[Dict[str, Any]] = []
            for el in elements:
                tags = el.get("tags") or {}
                name = tags.get("name:es") or tags.get("name:en") or tags.get("name")
                lat = el.get("lat")
                lon = el.get("lon")
                if not name or lat is None or lon is None:
                    continue
                place_type = str(tags.get("place") or "city").lower()
                pop_raw = str(tags.get("population") or "0").replace(",", "").strip()
                pop_val = int(pop_raw) if pop_raw.isdigit() else 0
                parsed.append(
                    {
                        "name": str(name),
                        "lat": round(float(lat), 4),
                        "lon": round(float(lon), 4),
                        "tier": place_type,
                        "population": pop_val,
                    }
                )
            return parsed
    except Exception:
        return []


def _classify_risk_from_temp(temp_c: float) -> str:
    if temp_c >= 31.5:
        return "ALTO RIESGO CRÍTICO"
    if temp_c >= 25.0:
        return "RIESGO MODERADO"
    return "BAJO RIESGO"


def _infer_nodes_ml_sync(
    nodes_to_infer: List[Dict[str, Any]],
    yr: int,
    co2_est: float,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for node in nodes_to_infer:
        lat = float(node["lat"])
        lon = float(node["lon"])
        base_lat = max(-10.0, 28.5 - 0.45 * abs(lat))
        ml_res = _ENGINE.predecir_coordenada_libre(lat, lon, yr, 7, co2_est, base_lat)
        results.append(
            {
                "name": node["name"],
                "lat": lat,
                "lon": lon,
                "tier": node["tier"],
                "temp": round(float(ml_res["temp_predicha"]), 2),
                "risk_level": str(ml_res["nivel_estres"]),
            }
        )
    return results


async def _assign_temperatures_to_nodes(
    topology_nodes: List[Dict[str, Any]],
    year: int,
) -> List[Dict[str, Any]]:
    if not topology_nodes:
        return []

    yr = int(year or 2026)
    co2_est = 425.0 + (yr - 2026) * 2.2 if yr >= 2026 else max(280.0, 425.0 - (2026 - yr) * 1.4)

    # Caso 1: Año en curso (2026) -> Batch Open-Meteo Forecast API + caché RAM TTL 15 min + fallback ML
    if yr == 2026:
        now = time.time()
        missing_indices: List[int] = []
        temps_by_idx: Dict[int, float] = {}

        for idx, node in enumerate(topology_nodes):
            grid_key = (round(float(node["lat"]) * 20.0) / 20.0, round(float(node["lon"]) * 20.0) / 20.0)
            cached = _LIVE_TEMP_CACHE.get(grid_key)
            if cached is not None and (now - cached[1]) < LIVE_TEMP_TTL_SEC:
                temps_by_idx[idx] = cached[0]
            else:
                missing_indices.append(idx)

        if missing_indices:
            lats_str = ",".join(str(topology_nodes[i]["lat"]) for i in missing_indices)
            lons_str = ",".join(str(topology_nodes[i]["lon"]) for i in missing_indices)
            params = {
                "latitude": lats_str,
                "longitude": lons_str,
                "current": "temperature_2m",
                "timezone": "auto",
            }
            try:
                async with httpx.AsyncClient(timeout=4.0) as client:
                    resp = await client.get(OPEN_METEO_FORECAST_URL, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    items = data if isinstance(data, list) else [data]
                    for idx_node, item in zip(missing_indices, items):
                        cur = (item.get("current") or {}) if isinstance(item, dict) else {}
                        t_val = cur.get("temperature_2m")
                        if t_val is not None:
                            t_float = round(float(t_val), 2)
                            temps_by_idx[idx_node] = t_float
                            n_ref = topology_nodes[idx_node]
                            g_key = (
                                round(float(n_ref["lat"]) * 20.0) / 20.0,
                                round(float(n_ref["lon"]) * 20.0) / 20.0,
                            )
                            _LIVE_TEMP_CACHE[g_key] = (t_float, now)
            except Exception:
                pass

        unassigned_indices = [i for i in range(len(topology_nodes)) if i not in temps_by_idx]
        fallback_by_idx: Dict[int, Dict[str, Any]] = {}
        if unassigned_indices:
            unassigned_nodes = [topology_nodes[i] for i in unassigned_indices]
            inferred_list = await asyncio.to_thread(_infer_nodes_ml_sync, unassigned_nodes, 2026, 425.0)
            for idx_u, inf_item in zip(unassigned_indices, inferred_list):
                fallback_by_idx[idx_u] = inf_item

        enriched: List[Dict[str, Any]] = []
        for idx, node in enumerate(topology_nodes):
            lat = float(node["lat"])
            lon = float(node["lon"])
            if idx in temps_by_idx:
                temp_val = temps_by_idx[idx]
                risk_lbl = _classify_risk_from_temp(temp_val)
                enriched.append(
                    {
                        "name": node["name"],
                        "lat": lat,
                        "lon": lon,
                        "tier": node["tier"],
                        "temp": temp_val,
                        "risk_level": risk_lbl,
                    }
                )
            else:
                enriched.append(fallback_by_idx[idx])
        return enriched

    # Caso 2: Histórico o Proyección Futura (year != 2026) -> Evaluación ML en hilo separado (no bloquea event loop)
    return await asyncio.to_thread(_infer_nodes_ml_sync, topology_nodes, yr, co2_est)


async def obtener_nodos_viewport(
    south: float,
    west: float,
    north: float,
    east: float,
    zoom: int = 4,
    year: int = 2026,
) -> Dict[str, Any]:
    """
    Retorna la jerarquía progresiva de nodos urbanos dentro del encuadre (bounding box)
    según el nivel de zoom (LOD), con decluttering espacial, caché SQLite y temperatura asignada.
    """
    z = max(1, min(18, int(zoom)))
    if z <= 5:
        tier_label = "megacity"
        max_nodes = 55
    elif z in (6, 7):
        tier_label = "city"
        max_nodes = 65
    elif z in (8, 9, 10):
        tier_label = "city_town"
        max_nodes = 75
    else:
        tier_label = "local_settlement"
        max_nodes = 85

    tile_key = _make_tile_key(south, west, north, east, z)
    base_in_box = [
        c for c in GLOBAL_MAJOR_CITIES if _in_bbox(c["lat"], c["lon"], south, west, north, east)
    ]

    # 1. Consultar topología del cuadrante en caché L1 (RAM) o L2 (SQLite viewport_nodes_cache)
    topology_nodes = _L1_VIEWPORT_CACHE.get(tile_key)
    if topology_nodes is None:
        topology_nodes = await asyncio.to_thread(_read_viewport_cache_sync, tile_key)
        if topology_nodes is not None:
            if base_in_box:
                topology_nodes = _declutter_nodes(topology_nodes + base_in_box, zoom=z, max_nodes=max_nodes)
            _L1_VIEWPORT_CACHE[tile_key] = topology_nodes

    # 2. Si no está en caché, construir jerarquía progresiva según zoom
    if topology_nodes is None:

        if z <= 5:
            candidates = base_in_box
        elif z in (6, 7):
            overpass_nodes = await _query_overpass_nodes(
                south, west, north, east, place_filter='="city"', limit=65
            )
            candidates = base_in_box + overpass_nodes
        elif z in (8, 9, 10):
            overpass_nodes = await _query_overpass_nodes(
                south, west, north, east, place_filter='~"^(city|town)$"', limit=75
            )
            candidates = overpass_nodes + base_in_box
        else:
            overpass_nodes = await _query_overpass_nodes(
                south, west, north, east, place_filter='~"^(city|town|village|hamlet)$"', limit=85
            )
            candidates = overpass_nodes + base_in_box

        topology_nodes = _declutter_nodes(candidates, zoom=z, max_nodes=max_nodes)

        if topology_nodes:
            _L1_VIEWPORT_CACHE[tile_key] = topology_nodes
            await asyncio.to_thread(_write_viewport_cache_sync, tile_key, topology_nodes)

    # 3. Asignar temperaturas (Open-Meteo batch para 2026 o inferencia ML en cascada para otro año)
    nodes_with_temps = await _assign_temperatures_to_nodes(topology_nodes, year=int(year))

    return {
        "status": "success",
        "zoom_tier": tier_label,
        "count": len(nodes_with_temps),
        "nodes": nodes_with_temps,
    }
