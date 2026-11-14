"""
URBAN-HEAT GRID INTELLIGENCE — Servicio Climático Global en Tiempo Real (Open-Meteo API)
Archivo: services/weather_api.py
Consulta asíncrona con httpx (timeout 5.0s) y cálculo psicrométrico de Temperatura
de Bulbo Húmedo (Fórmula empírica de Stull, 2011) y carga latente de enfriamiento.
"""
import time
from typing import Any, Dict, Tuple

import httpx
import numpy as np

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CACHE_TTL_SECONDS = 600.0  # 10 minutos de caché en memoria por coordenada


class GlobalWeatherService:
    """
    Cliente asíncrono para telemetría meteorológica global vía Open-Meteo API.
    Calcula Temperatura de Bulbo Húmedo (Stull) y métricas de carga latente para la red eléctrica.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout
        self._cache: Dict[Tuple[float, float], Tuple[float, Dict[str, Any]]] = {}

    @staticmethod
    def calcular_bulbo_humedo_stull(temp_c: float, rh_pct: float) -> float:
        """
        Calcula la temperatura de bulbo húmedo simplificada (Twb en °C) mediante
        la fórmula empírica de Roland Stull (2011).
        """
        t = float(temp_c)
        rh = max(1.0, min(100.0, float(rh_pct)))
        twb = (
            t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
            + np.arctan(t + rh)
            - np.arctan(rh - 1.676331)
            + 0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh)
            - 4.686035
        )
        return round(float(twb), 1)

    async def obtener_clima_actual(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Consulta Open-Meteo Forecast API con timeout de 5.0s y retorna
        telemetría actual junto con temperatura de bulbo húmedo (Twb).
        """
        lat_f = round(max(-90.0, min(90.0, float(lat))), 4)
        lon_f = round(((float(lon) + 180.0) % 360.0) - 180.0, 4)
        cache_key = (round(lat_f, 2), round(lon_f, 2))

        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached is not None and (now - cached[0]) < WEATHER_CACHE_TTL_SECONDS:
            return cached[1]

        params = {
            "latitude": lat_f,
            "longitude": lon_f,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(OPEN_METEO_FORECAST_URL, params=params)
                r.raise_for_status()
                data = r.json()
        except httpx.RequestError as exc:
            raise RuntimeError(f"Error de red hacia Open-Meteo ({exc.__class__.__name__}): {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"HTTP {exc.response.status_code} desde Open-Meteo") from exc

        curr = data.get("current") or {}
        daily = data.get("daily") or {}

        t = float(curr.get("temperature_2m", 25.0))
        rh = float(curr.get("relative_humidity_2m", 60.0))
        app_t = float(curr.get("apparent_temperature", t))
        wind_kmh = float(curr.get("wind_speed_10m", 0.0))
        is_day = int(curr.get("is_day", 1))
        weather_code = int(curr.get("weather_code", 0))

        daily_max_list = daily.get("temperature_2m_max") or []
        daily_min_list = daily.get("temperature_2m_min") or []
        t_max = float(daily_max_list[0]) if daily_max_list and daily_max_list[0] is not None else round(t + 3.5, 1)
        t_min = float(daily_min_list[0]) if daily_min_list and daily_min_list[0] is not None else round(t - 4.5, 1)

        twb = self.calcular_bulbo_humedo_stull(t, rh)
        cdd_live = round(max(0.0, t - 18.0), 2)
        latent_cooling_index = round(
            max(0.0, (twb - 16.0) * 1.4 + (rh / 100.0) * max(0.0, t - 18.0)),
            2,
        )

        result: Dict[str, Any] = {
            "temperatura": round(t, 1),
            "sensacion": round(app_t, 1),
            "humedad": int(round(rh, 0)),
            "twb": round(twb, 1),
            "viento_kmh": round(wind_kmh, 1),
            "latitude": float(data.get("latitude", lat_f)),
            "longitude": float(data.get("longitude", lon_f)),
            "timezone": str(data.get("timezone", "UTC")),
            "timestamp": str(curr.get("time", "")),
            "temperature_2m": round(t, 1),
            "relative_humidity_2m": round(rh, 1),
            "apparent_temperature": round(app_t, 1),
            "wet_bulb_temperature": round(twb, 1),
            "is_day": is_day,
            "weather_code": weather_code,
            "wind_speed_10m": round(wind_kmh, 1),
            "temperature_2m_max": round(t_max, 1),
            "temperature_2m_min": round(t_min, 1),
            "cdd_daily": cdd_live,
            "latent_cooling_index": latent_cooling_index,
        }

        if len(self._cache) >= 1024:
            self._cache.clear()
        self._cache[cache_key] = (now, result)
        return result


weather_service = GlobalWeatherService()
