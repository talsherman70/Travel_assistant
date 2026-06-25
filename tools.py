"""
External tool integrations: Open-Meteo (weather) and Overpass API/OpenStreetMap (attractions).

All failures produce a result with .error set rather than raising exceptions.
The response generator checks .succeeded and handles errors gracefully.
"""

from __future__ import annotations

from datetime import datetime

import httpx

from models import (
    AttractionItem,
    AttractionsResult,
    IntentExtraction,
    TripContext,
    WeatherDay,
    WeatherResult,
)

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_ATTRACTIONS_RADIUS_M = 3_000  # 3 km radius — Overpass default
_ATTRACTIONS_LIMIT = 10
_HTTP_TIMEOUT = 10.0  # seconds


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M")


def _geocode(city: str) -> tuple[float, float] | None:
    """Return (lat, lon) for a city name via Open-Meteo geocoding, or None on failure."""
    try:
        r = httpx.get(
            _GEOCODE_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        return results[0]["latitude"], results[0]["longitude"]
    except Exception:
        return None


def get_weather(destination: str, start_date: str | None = None) -> WeatherResult:
    """Fetch a 7-day weather forecast for destination via Open-Meteo (no key required)."""
    coords = _geocode(destination)
    if coords is None:
        return WeatherResult(
            destination=destination,
            retrieved_at=_now(),
            error=f"Could not geocode '{destination}' — check spelling or try a nearby major city.",
        )

    lat, lon = coords
    try:
        r = httpx.get(
            _FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
                "forecast_days": 7,
                "timezone": "auto",
            },
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        daily = data.get("daily", {})

        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [None] * len(dates))
        lows = daily.get("temperature_2m_min", [None] * len(dates))
        precips = daily.get("precipitation_sum", [None] * len(dates))
        wcodes = daily.get("weathercode", [None] * len(dates))

        forecast = [
            WeatherDay(
                date=dates[i],
                temp_high_c=highs[i],
                temp_low_c=lows[i],
                condition=_wmo_condition(wcodes[i]),
                precipitation_mm=precips[i],
            )
            for i in range(len(dates))
        ]

        return WeatherResult(
            destination=destination,
            retrieved_at=_now(),
            forecast=forecast,
        )
    except httpx.HTTPStatusError as e:
        return WeatherResult(
            destination=destination,
            retrieved_at=_now(),
            error=f"Weather API returned {e.response.status_code}",
        )
    except Exception as e:
        return WeatherResult(
            destination=destination,
            retrieved_at=_now(),
            error=f"Weather API error: {e}",
        )


def get_attractions(destination: str) -> AttractionsResult:
    """Fetch top attractions for destination via Overpass API (OpenStreetMap). No key required."""
    import asyncio
    from services.overpass_service import search_places

    coords = _geocode(destination)
    if coords is None:
        return AttractionsResult(
            destination=destination,
            retrieved_at=_now(),
            error=f"Could not geocode '{destination}' — check spelling or try a nearby major city.",
        )

    lat, lon = coords
    try:
        places = asyncio.run(search_places(lat, lon, radius=_ATTRACTIONS_RADIUS_M, limit=_ATTRACTIONS_LIMIT))
    except Exception as e:
        return AttractionsResult(
            destination=destination,
            retrieved_at=_now(),
            error=f"Attractions error: {e}",
        )

    if not places:
        return AttractionsResult(
            destination=destination,
            retrieved_at=_now(),
            error=f"No attractions found near '{destination}'.",
        )

    items = [
        AttractionItem(
            name=p["name"],
            kinds=[p["category"]] if p.get("category") else [],
            description=p.get("address"),
        )
        for p in places
    ]

    return AttractionsResult(
        destination=destination,
        retrieved_at=_now(),
        attractions=items,
    )


class ToolRouter:
    """
    Routes tool calls based on IntentExtraction flags.
    Caches results per (destination, date) key to avoid redundant API calls.
    Cache is invalidated by ContextManager when destination or dates change.
    """

    def __init__(self, cache: dict | None = None):
        # Shared cache reference; ContextManager clears it on dest/date change
        self._cache: dict = cache if cache is not None else {}

    def route(
        self,
        extraction: IntentExtraction,
        trip_context: TripContext,
    ) -> tuple[WeatherResult | None, AttractionsResult | None]:
        weather = None
        attractions = None

        if extraction.needs_weather and trip_context.destination:
            cache_key = f"weather:{trip_context.destination.lower()}:{trip_context.start_date}"
            weather = self._cache.get(cache_key)
            if weather is None:
                weather = get_weather(trip_context.destination, trip_context.start_date)
                self._cache[cache_key] = weather

        if extraction.needs_attractions and trip_context.destination:
            cache_key = f"attractions:{trip_context.destination.lower()}"
            attractions = self._cache.get(cache_key)
            if attractions is None:
                attractions = get_attractions(trip_context.destination)
                self._cache[cache_key] = attractions

        return weather, attractions


# WMO weather interpretation codes → human-readable string
def _wmo_condition(code: int | None) -> str | None:
    if code is None:
        return None
    _WMO: dict[int, str] = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "foggy (rime)",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "heavy drizzle",
        61: "light rain",
        63: "moderate rain",
        65: "heavy rain",
        71: "light snow",
        73: "moderate snow",
        75: "heavy snow",
        77: "snow grains",
        80: "light showers",
        81: "moderate showers",
        82: "heavy showers",
        85: "light snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with hail",
        99: "thunderstorm with heavy hail",
    }
    return _WMO.get(code, f"weather code {code}")
