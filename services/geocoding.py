"""
URBAN-HEAT GRID INTELLIGENCE — Servicio de Geocodificación Global Asíncrono
Archivo: services/geocoding.py
Integra OpenStreetMap Nominatim (Search + Reverse) con persistencia local SQLite (data/geo_cache.db)
y priorización de cabeceras urbanas pobladas (city, town, village, suburb, hamlet) sobre polígonos administrativos.
"""
import asyncio
import json
import sqlite3
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DB_PATH: Path = BASE_DIR / "data" / "geo_cache.db"
NOMINATIM_USER_AGENT = "UrbanHeatGridIntelligence/2.0 (contact: academic_project)"

URBAN_PLACE_TYPES = {"city", "town", "village", "suburb", "hamlet", "borough", "quarter", "neighbourhood"}
ADMIN_BOUNDARY_TYPES = {"administrative", "municipality", "county", "state", "region", "boundary"}


class GlobalGeocodingService:
    """
    Servicio singleton de geocodificación directa e inversa con caché persistente SQLite,
    espejo L1 en RAM y priorización espacial de cabeceras urbanas frente a polígonos municipales.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.Lock()
        self._l1_search: Dict[str, List[Dict[str, Any]]] = {}
        self._l1_reverse: Dict[str, Dict[str, Any]] = {}
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._db_lock:
            with self._get_conn() as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS places_cache (
                        query TEXT PRIMARY KEY,
                        name TEXT,
                        display_name TEXT,
                        lat REAL,
                        lon REAL,
                        place_type TEXT,
                        country TEXT,
                        results_json TEXT,
                        timestamp DATETIME
                    )
                    """
                )
                cols = {row["name"] for row in conn.execute("PRAGMA table_info(places_cache)").fetchall()}
                if "results_json" not in cols:
                    conn.execute("ALTER TABLE places_cache ADD COLUMN results_json TEXT")
                # Purgar entradas antiguas donde el top 1 quedó guardado como polígono 'administrative' o aldea homónima
                conn.execute(
                    "DELETE FROM places_cache WHERE (place_type IN ('administrative', 'village') OR query = 'tulum') AND query NOT LIKE 'rev:%'"
                )
                conn.commit()

    @staticmethod
    def _normalize_query(q: str) -> str:
        return " ".join((q or "").strip().lower().split())

    @staticmethod
    def _make_reverse_key(lat: float, lon: float) -> str:
        return f"rev:{round(float(lat), 3):.3f},{round(float(lon), 3):.3f}"

    @staticmethod
    def _urban_priority_rank(item: Dict[str, Any], norm_query: str = "") -> Tuple[int, int, float]:
        """
        Ordena los resultados priorizando:
        1. Coincidencia exacta del nombre con la consulta.
        2. Jerarquía urbana poblada: (city, town) > (village, suburb, borough) > (hamlet) > (municipality, administrative).
        3. Importancia geográfica de Nominatim (mayor importancia primero).
        """
        p_type = str(item.get("type") or "").lower()
        p_class = str(item.get("class") or "").lower()
        name_norm = str(item.get("name") or "").strip().lower()
        q_head = norm_query.split(",")[0].strip() if norm_query else ""

        exact_name_penalty = 0 if (q_head and name_norm == q_head) else (1 if (q_head and q_head in name_norm) else 2)

        if p_type in ("city", "town"):
            type_rank = 0
        elif p_type in ("village", "suburb", "borough", "city_district"):
            type_rank = 1
        elif p_type in ("hamlet", "quarter", "neighbourhood") or p_class == "place":
            type_rank = 2
        elif p_type in ADMIN_BOUNDARY_TYPES or p_class == "boundary":
            type_rank = 3
        else:
            type_rank = 2

        importance = float(item.get("importance") or 0.0)
        return (exact_name_penalty, type_rank, -importance)

    def _read_from_sqlite(self, key: str) -> Optional[List[Dict[str, Any]]]:
        with self._db_lock:
            with self._get_conn() as conn:
                row = conn.execute(
                    """
                    SELECT query, name, display_name, lat, lon, place_type, country, results_json
                    FROM places_cache
                    WHERE query = ?
                    """,
                    (key,),
                ).fetchone()
                if row is None:
                    return None
                if row["results_json"]:
                    try:
                        parsed = json.loads(row["results_json"])
                        if isinstance(parsed, list) and parsed:
                            return parsed
                    except Exception:
                        pass
                return [
                    {
                        "name": row["name"],
                        "display_name": row["display_name"],
                        "lat": float(row["lat"]),
                        "lon": float(row["lon"]),
                        "type": row["place_type"] or "place",
                        "country": row["country"] or "",
                    }
                ]

    def _write_to_sqlite(self, key: str, items: List[Dict[str, Any]]) -> None:
        if not items:
            return
        top = items[0]
        now_iso = datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(items, ensure_ascii=False)
        with self._db_lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO places_cache
                    (query, name, display_name, lat, lon, place_type, country, results_json, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        str(top.get("name", "")),
                        str(top.get("display_name", "")),
                        float(top.get("lat", 0.0)),
                        float(top.get("lon", 0.0)),
                        str(top.get("type", "place")),
                        str(top.get("country", "")),
                        payload_json,
                        now_iso,
                    ),
                )
                conn.commit()

    @staticmethod
    def _fetch_nominatim_json(url: str, timeout: float = 5.0) -> Any:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": NOMINATIM_USER_AGENT,
                "Accept": "application/json",
                "Accept-Language": "es,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _parse_nominatim_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        lat_raw = item.get("lat")
        lon_raw = item.get("lon")
        if lat_raw is None or lon_raw is None:
            return None

        address = item.get("address") or {}
        name = (
            item.get("name")
            or address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("suburb")
            or address.get("hamlet")
            or address.get("municipality")
            or address.get("county")
            or address.get("state")
            or (str(item.get("display_name", "")).split(",")[0].strip())
            or "Punto Geográfico"
        )
        display_name = str(item.get("display_name") or name)

        raw_type = str(item.get("type") or "").lower()
        addr_type = str(item.get("addresstype") or "").lower()
        raw_class = str(item.get("class") or "").lower()

        # Si Nominatim etiqueta la relación como 'administrative', usar 'addresstype' ('town', 'city', 'village', etc.)
        if raw_type in ("administrative", "boundary", "") and addr_type:
            place_type = addr_type
        else:
            place_type = raw_type or addr_type or "place"

        country = str(address.get("country") or "")
        importance = float(item.get("importance") or 0.0)

        return {
            "name": str(name),
            "display_name": display_name,
            "lat": round(float(lat_raw), 4),
            "lon": round(float(lon_raw), 4),
            "type": place_type,
            "class": raw_class,
            "country": country,
            "importance": round(importance, 4),
        }

    async def search(self, query: str) -> List[Dict[str, Any]]:
        norm_q = self._normalize_query(query)
        if not norm_q:
            return []

        # 1. Consulta en espejo L1 RAM (< 0.05 ms)
        if norm_q in self._l1_search:
            return self._l1_search[norm_q]

        # 2. Consulta en SQLite local (< 1.5 ms)
        cached_db = await asyncio.to_thread(self._read_from_sqlite, norm_q)
        if cached_db is not None:
            self._l1_search[norm_q] = cached_db
            return cached_db

        # 3. Consulta externa a OpenStreetMap Nominatim API con addressdetails=1 y limit=10
        encoded_q = urllib.parse.quote(query.strip())
        url = f"https://nominatim.openstreetmap.org/search?q={encoded_q}&format=json&addressdetails=1&limit=10"

        parsed_candidates: List[Dict[str, Any]] = []
        try:
            raw_data = await asyncio.to_thread(self._fetch_nominatim_json, url, 5.0)
            if isinstance(raw_data, list):
                for item in raw_data:
                    parsed = self._parse_nominatim_item(item)
                    if parsed is not None:
                        parsed_candidates.append(parsed)
        except Exception:
            parsed_candidates = []

        # Priorizar cabeceras urbanas (town, city, village, suburb, hamlet) sobre polígonos municipales/administrativos
        parsed_candidates.sort(key=lambda it: self._urban_priority_rank(it, norm_q))

        # Deduplicar entradas redundantes conservando la cabecera urbana prioritaria
        results: List[Dict[str, Any]] = []
        seen_labels = set()
        for cand in parsed_candidates:
            dedup_key = (cand["name"].lower(), cand["display_name"].lower())
            if dedup_key in seen_labels:
                continue
            seen_labels.add(dedup_key)
            results.append(
                {
                    "name": cand["name"],
                    "display_name": cand["display_name"],
                    "lat": cand["lat"],
                    "lon": cand["lon"],
                    "type": cand["type"],
                    "country": cand["country"],
                }
            )
            if len(results) >= 5:
                break

        if results:
            self._l1_search[norm_q] = results
            await asyncio.to_thread(self._write_to_sqlite, norm_q, results)

        return results

    async def reverse(self, lat: float, lon: float) -> Dict[str, Any]:
        lat_f = round(float(lat), 4)
        lon_f = round(((float(lon) + 180.0) % 360.0) - 180.0, 4)
        rev_key = self._make_reverse_key(lat_f, lon_f)

        if rev_key in self._l1_reverse:
            return self._l1_reverse[rev_key]

        cached_db = await asyncio.to_thread(self._read_from_sqlite, rev_key)
        if cached_db:
            top = cached_db[0]
            self._l1_reverse[rev_key] = top
            return top

        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat_f}&lon={lon_f}&format=json&addressdetails=1"
        fallback = {
            "name": f"Coord ({lat_f:.2f}°, {lon_f:.2f}°)",
            "display_name": f"Coordenada Global ({lat_f:.4f}, {lon_f:.4f})",
            "lat": lat_f,
            "lon": lon_f,
            "type": "coordinate",
            "country": "Global",
        }

        try:
            raw_data = await asyncio.to_thread(self._fetch_nominatim_json, url, 4.5)
            parsed = self._parse_nominatim_item(raw_data)
            if parsed is not None:
                clean_parsed = {
                    "name": parsed["name"],
                    "display_name": parsed["display_name"],
                    "lat": lat_f,
                    "lon": lon_f,
                    "type": parsed["type"],
                    "country": parsed["country"],
                }
                self._l1_reverse[rev_key] = clean_parsed
                await asyncio.to_thread(self._write_to_sqlite, rev_key, [clean_parsed])
                return clean_parsed
        except Exception:
            pass

        return fallback


geocoding_service = GlobalGeocodingService()
