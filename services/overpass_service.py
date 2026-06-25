"""
Overpass API (OpenStreetMap) places search service.

No API key required. Uses the public Overpass API endpoint.
Async HTTP via httpx.AsyncClient.
"""

from __future__ import annotations

import httpx

# Public Overpass mirrors tried in order; first success wins.
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
_HTTP_TIMEOUT = 25.0
_USER_AGENT = "TravelAssistant/1.0 (student project; educational use)"

# Simple user query → OSM tag mapping
_QUERY_TO_TAG: dict[str, tuple[str, str]] = {
    "restaurant":  ("amenity",  "restaurant"),
    "cafe":        ("amenity",  "cafe"),
    "bar":         ("amenity",  "bar"),
    "pub":         ("amenity",  "pub"),
    "hotel":       ("tourism",  "hotel"),
    "museum":      ("tourism",  "museum"),
    "attraction":  ("tourism",  "attraction"),
    "park":        ("leisure",  "park"),
    "supermarket": ("shop",     "supermarket"),
    "pharmacy":    ("amenity",  "pharmacy"),
}

# Broad fallback when query is None or unknown
_FALLBACK_TAGS: list[tuple[str, str]] = [
    ("tourism", "attraction"),
    ("tourism", "museum"),
    ("amenity", "restaurant"),
    ("amenity", "cafe"),
    ("leisure", "park"),
]


def _build_query(lat: float, lon: float, tags: list[tuple[str, str]], radius: int, limit: int) -> str:
    """Build an Overpass QL query for the given tags, radius, and limit."""
    lines: list[str] = []
    for key, value in tags:
        f = f'["{key}"="{value}"](around:{radius},{lat},{lon})'
        lines.append(f"  node{f};")
        lines.append(f"  way{f};")
        lines.append(f"  relation{f};")
    body = "\n".join(lines)
    return f"[out:json][timeout:25];\n(\n{body}\n);\nout center tags {limit};"


def _parse_element(element: dict) -> dict | None:
    """Parse one Overpass element into a clean place dict. Returns None if unnamed."""
    tags = element.get("tags", {})
    name = tags.get("name", "").strip()
    if not name:
        return None

    elem_type = element.get("type", "node")

    if elem_type == "node":
        lat = element.get("lat")
        lon = element.get("lon")
    else:
        center = element.get("center", {})
        lat = center.get("lat")
        lon = center.get("lon")

    category = (
        tags.get("tourism")
        or tags.get("amenity")
        or tags.get("leisure")
        or tags.get("shop")
        or "place"
    )

    addr_parts = [
        tags.get("addr:housenumber", ""),
        tags.get("addr:street", ""),
        tags.get("addr:city", ""),
    ]
    address = ", ".join(p for p in addr_parts if p) or None

    return {
        "id": element.get("id"),
        "type": elem_type,
        "name": name,
        "category": category,
        "lat": lat,
        "lon": lon,
        "address": address,
        "tags": tags,
    }


async def search_places(
    lat: float,
    lon: float,
    query: str | None = None,
    radius: int = 3000,
    limit: int = 10,
) -> list[dict]:
    """Search for places near coordinates using the Overpass API.

    Args:
        lat:    Latitude of the search center.
        lon:    Longitude of the search center.
        query:  Simple category string (e.g. "museum", "restaurant").
                Unknown or None falls back to a broad tourist-relevant set.
        radius: Search radius in metres. Default 3000.
        limit:  Maximum results returned. Default 10.

    Returns:
        List of place dicts (id, type, name, category, lat, lon, address, tags).
        Empty list on timeout, HTTP error, malformed response, or no named results.
    """
    if query:
        tag = _QUERY_TO_TAG.get(query.strip().lower())
        tags = [tag] if tag else _FALLBACK_TAGS
    else:
        tags = _FALLBACK_TAGS

    overpass_query = _build_query(lat, lon, tags, radius, limit)

    data: dict | None = None
    for endpoint in _OVERPASS_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.post(
                    endpoint,
                    data={"data": overpass_query},
                    headers={"User-Agent": _USER_AGENT},
                )
                resp.raise_for_status()
                data = resp.json()
                break  # success — stop trying mirrors
        except httpx.TimeoutException:
            print(f"[overpass] {endpoint} timed out — trying next mirror")
            continue
        except httpx.HTTPStatusError as e:
            print(f"[overpass] {endpoint} HTTP {e.response.status_code} — trying next mirror")
            continue
        except Exception as e:
            print(f"[overpass] {endpoint} error: {e} — trying next mirror")
            continue

    if data is None:
        return []

    elements = data.get("elements", [])
    results: list[dict] = []
    for element in elements:
        place = _parse_element(element)
        if place is not None:
            results.append(place)
        if len(results) >= limit:
            break

    return results
