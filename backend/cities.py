from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from backend.config import CITY_DIVERSITY_DECAY_KM

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from backend.scoring import CellScorePoint

GRID_DEGREES = 10 / 60
GRID_HALF_DEGREES = GRID_DEGREES / 2
MAX_LATITUDE_INDEX = 1079
MAX_LONGITUDE_INDEX = 2159
EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True, slots=True)
class CityCandidate:
    """One user-facing place that can inherit a climate-cell score."""

    name: str
    country_code: str
    lat: float
    lon: float
    cell_lat: float
    cell_lon: float


class CityScorePoint(TypedDict):
    """JSON score payload for the ranked city list."""

    name: str
    country_code: str
    flag: str
    score: float


@dataclass(frozen=True, slots=True)
class RankedCityCandidate:
    """City plus its current ranking state during diversified selection."""

    city: CityCandidate
    score: float


@dataclass(frozen=True, slots=True)
class CityRankingCache:
    """Compact city ranking inputs for the optimized scoring path."""

    cities: tuple[CityCandidate, ...]
    climate_indexes: NDArray[np.int32]
    latitude_radians: NDArray[np.float32]
    longitude_radians: NDArray[np.float32]
    cosine_latitudes: NDArray[np.float32]
    flags: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject malformed ranking arrays before they reach the hot path."""
        city_count = len(self.cities)

        if self.climate_indexes.shape != (city_count,):
            msg = "climate_indexes must align with cities"
            raise ValueError(msg)

        if self.latitude_radians.shape != (city_count,):
            msg = "latitude_radians must align with cities"
            raise ValueError(msg)

        if self.longitude_radians.shape != (city_count,):
            msg = "longitude_radians must align with cities"
            raise ValueError(msg)

        if self.cosine_latitudes.shape != (city_count,):
            msg = "cosine_latitudes must align with cities"
            raise ValueError(msg)

        if len(self.flags) != city_count:
            msg = "flags must align with cities"
            raise ValueError(msg)

    @classmethod
    def from_cities(
        cls,
        cities: tuple[CityCandidate, ...],
        climate_indexes: NDArray[np.int32],
    ) -> CityRankingCache:
        """Build the vector-friendly ranking cache from resolved cities."""
        latitude_radians = np.radians(np.array([city.lat for city in cities], dtype=np.float32)).astype(
            np.float32,
            copy=False,
        )
        longitude_radians = np.radians(np.array([city.lon for city in cities], dtype=np.float32)).astype(
            np.float32,
            copy=False,
        )
        return cls(
            cities=cities,
            climate_indexes=climate_indexes,
            latitude_radians=latitude_radians,
            longitude_radians=longitude_radians,
            cosine_latitudes=np.cos(latitude_radians).astype(np.float32, copy=False),
            flags=tuple(country_flag(city.country_code) for city in cities),
        )


STUB_CITY_CANDIDATES: tuple[CityCandidate, ...] = (
    CityCandidate(name="San Francisco", country_code="US", lat=37.5, lon=-122.0, cell_lat=37.5, cell_lon=-122.0),
    CityCandidate(name="Rome", country_code="IT", lat=41.9, lon=12.5, cell_lat=41.9, cell_lon=12.5),
    CityCandidate(name="Cape Town", country_code="ZA", lat=-33.9, lon=18.4, cell_lat=-33.9, cell_lon=18.4),
    CityCandidate(name="Tokyo", country_code="JP", lat=35.7, lon=139.7, cell_lat=35.7, cell_lon=139.7),
    CityCandidate(name="Rio de Janeiro", country_code="BR", lat=-22.9, lon=-43.2, cell_lat=-22.9, cell_lon=-43.2),
)


def rank_city_scores(
    city_catalog: tuple[CityCandidate, ...],
    cell_scores: list[CellScorePoint],
    *,
    limit: int,
    diversity_decay_km: float = CITY_DIVERSITY_DECAY_KM,
) -> list[CityScorePoint]:
    """Project scored grid cells onto cities and diversify the shortlist by region."""
    score_by_cell = {
        (round(score_point["lat"], 4), round(score_point["lon"], 4)): score_point["score"]
        for score_point in cell_scores
    }
    remaining: list[RankedCityCandidate] = []
    ranked: list[CityScorePoint] = []

    for city in city_catalog:
        score = score_by_cell.get((round(city.cell_lat, 4), round(city.cell_lon, 4)))

        if score is None:
            continue

        remaining.append(RankedCityCandidate(city=city, score=score))

    while remaining and len(ranked) < limit:
        winner = max(remaining, key=lambda candidate: candidate.score)
        ranked.append(
            {
                "name": winner.city.name,
                "country_code": winner.city.country_code,
                "flag": country_flag(winner.city.country_code),
                "score": round(winner.score, 4),
            }
        )
        remaining = [
            RankedCityCandidate(
                city=candidate.city,
                score=apply_regional_penalty(
                    candidate.score,
                    winner.city,
                    candidate.city,
                    winner.score,
                    decay_km=diversity_decay_km,
                ),
            )
            for candidate in remaining
            if candidate.city != winner.city
        ]

    return ranked


def rank_indexed_city_scores(
    city_catalog: CityRankingCache,
    cell_scores: NDArray[np.float32],
    *,
    limit: int,
    diversity_decay_km: float = CITY_DIVERSITY_DECAY_KM,
) -> list[CityScorePoint]:
    """Rank cities by direct climate-matrix lookup instead of lat/lon joins."""
    if not city_catalog.cities:
        return []

    scores = cell_scores[city_catalog.climate_indexes].astype(np.float32, copy=True)
    active = np.ones(scores.shape, dtype=bool)
    ranked: list[CityScorePoint] = []

    while active.any() and len(ranked) < limit:
        winner_index = int(np.argmax(np.where(active, scores, -1.0), axis=None))
        winner_city = city_catalog.cities[winner_index]
        winner_score = float(scores[winner_index])
        ranked.append(
            {
                "name": winner_city.name,
                "country_code": winner_city.country_code,
                "flag": city_catalog.flags[winner_index],
                "score": round(winner_score, 4),
            }
        )

        distance_km = _haversine_distance_vector_km(city_catalog, winner_index)
        penalty = winner_score * np.exp(-distance_km / diversity_decay_km)
        scores = np.where(active, np.maximum(0.0, scores * (1.0 - penalty)), scores)
        active[winner_index] = False

    return ranked


def apply_regional_penalty(
    score: float,
    center: CityCandidate,
    candidate: CityCandidate,
    center_score: float,
    *,
    decay_km: float = CITY_DIVERSITY_DECAY_KM,
) -> float:
    """Exponentially suppress cities that cluster too close to a stronger regional center."""
    distance_km = haversine_distance_km(center.lat, center.lon, candidate.lat, candidate.lon)
    penalty = center_score * math.exp(-distance_km / decay_km)
    return max(0.0, score * (1 - penalty))


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Measure great-circle distance so regional suppression is geography-based."""
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    half_chord = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(half_chord))


def _haversine_distance_vector_km(
    city_catalog: CityRankingCache,
    winner_index: int,
) -> NDArray[np.float32]:
    """Vectorized great-circle distance for one regional-center update."""
    latitude_radians = city_catalog.latitude_radians[winner_index]
    longitude_radians = city_catalog.longitude_radians[winner_index]
    cosine_latitude = city_catalog.cosine_latitudes[winner_index]
    delta_lat = city_catalog.latitude_radians - latitude_radians
    delta_lon = city_catalog.longitude_radians - longitude_radians
    half_chord = (
        np.sin(delta_lat / 2.0) ** 2 + cosine_latitude * city_catalog.cosine_latitudes * np.sin(delta_lon / 2.0) ** 2
    )
    return (2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(half_chord))).astype(np.float32, copy=False)


def snap_city_to_cell_key(city: CityCandidate) -> tuple[float, float]:
    """Snap a city location to the nearest native 10-arcminute WorldClim cell center."""
    latitude_index = round((90 - GRID_HALF_DEGREES - city.lat) / GRID_DEGREES)
    longitude_index = round((city.lon - (-180 + GRID_HALF_DEGREES)) / GRID_DEGREES)
    latitude_index = min(max(latitude_index, 0), MAX_LATITUDE_INDEX)
    longitude_index = min(max(longitude_index, 0), MAX_LONGITUDE_INDEX)

    return (
        round(90 - GRID_HALF_DEGREES - latitude_index * GRID_DEGREES, 4),
        round(-180 + GRID_HALF_DEGREES + longitude_index * GRID_DEGREES, 4),
    )


def coordinate_key(latitude: float, longitude: float) -> int:
    """Encode a rounded lat/lon pair into one sortable integer key."""
    latitude_key = round(latitude * 10_000) + 900_000
    longitude_key = round(longitude * 10_000) + 1_800_000
    return latitude_key * 4_000_000 + longitude_key


def country_flag(country_code: str) -> str:
    """Convert ISO alpha-2 country codes into regional indicator flag emoji."""
    return "".join(chr(0x1F1E6 + ord(letter) - ord("A")) for letter in country_code.upper())
