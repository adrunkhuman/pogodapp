from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from backend.config import CITY_DIVERSITY_DECAY_KM

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from backend.scoring import CellScorePoint

_EUROPE = frozenset(
    [
        "AD",
        "AL",
        "AT",
        "BA",
        "BE",
        "BG",
        "BY",
        "CH",
        "CY",
        "CZ",
        "DE",
        "DK",
        "EE",
        "ES",
        "FI",
        "FR",
        "GB",
        "GI",
        "GR",
        "HR",
        "HU",
        "IE",
        "IS",
        "IT",
        "LI",
        "LT",
        "LU",
        "LV",
        "MC",
        "MD",
        "ME",
        "MK",
        "MT",
        "NL",
        "NO",
        "PL",
        "PT",
        "RO",
        "RS",
        "RU",
        "SE",
        "SI",
        "SK",
        "SM",
        "UA",
        "VA",
        "XK",
    ]
)
_ASIA = frozenset(
    [
        "AE",
        "AF",
        "AM",
        "AZ",
        "BD",
        "BH",
        "BN",
        "BT",
        "CN",
        "GE",
        "ID",
        "IL",
        "IN",
        "IQ",
        "IR",
        "JO",
        "JP",
        "KG",
        "KH",
        "KP",
        "KR",
        "KW",
        "KZ",
        "LA",
        "LB",
        "LK",
        "MM",
        "MN",
        "MO",
        "MV",
        "MY",
        "NP",
        "OM",
        "PH",
        "PK",
        "PS",
        "QA",
        "SA",
        "SG",
        "SY",
        "TH",
        "TJ",
        "TL",
        "TM",
        "TR",
        "TW",
        "UZ",
        "VN",
        "YE",
    ]
)
_AFRICA = frozenset(
    [
        "AO",
        "BF",
        "BI",
        "BJ",
        "BW",
        "CD",
        "CF",
        "CG",
        "CI",
        "CM",
        "CV",
        "DJ",
        "DZ",
        "EG",
        "EH",
        "ER",
        "ET",
        "GA",
        "GH",
        "GM",
        "GN",
        "GQ",
        "GW",
        "KE",
        "KM",
        "LR",
        "LS",
        "LY",
        "MA",
        "MG",
        "ML",
        "MR",
        "MU",
        "MW",
        "MZ",
        "NA",
        "NE",
        "NG",
        "RW",
        "SC",
        "SD",
        "SL",
        "SN",
        "SO",
        "SS",
        "ST",
        "SZ",
        "TD",
        "TG",
        "TN",
        "TZ",
        "UG",
        "ZA",
        "ZM",
        "ZW",
    ]
)
_NORTH_AMERICA = frozenset(
    [
        "AG",
        "BB",
        "BS",
        "BZ",
        "CA",
        "CR",
        "CU",
        "DM",
        "DO",
        "GD",
        "GT",
        "HN",
        "HT",
        "JM",
        "KN",
        "LC",
        "MX",
        "NI",
        "PA",
        "TT",
        "US",
        "VC",
    ]
)
_SOUTH_AMERICA = frozenset(["AR", "BO", "BR", "CL", "CO", "EC", "GF", "GY", "PE", "PY", "SR", "UY", "VE"])
_OCEANIA = frozenset(["AU", "FJ", "FM", "KI", "MH", "NR", "NZ", "PG", "PW", "SB", "TO", "TV", "VU", "WS"])

_CONTINENT_LOOKUP: tuple[tuple[str, frozenset[str]], ...] = (
    ("Europe", _EUROPE),
    ("Asia", _ASIA),
    ("Africa", _AFRICA),
    ("North America", _NORTH_AMERICA),
    ("South America", _SOUTH_AMERICA),
    ("Oceania", _OCEANIA),
)

EUROPE_ASIA_LONGITUDE_SPLIT = 60.0


def continent_of(country_code: str, lon: float | None = None) -> str:
    """Map an ISO 3166-1 alpha-2 code to its continent name.

    Russia spans both Europe and Asia; use longitude to keep Siberian cities out
    of the Europe sidebar bucket.
    """
    if country_code == "RU" and lon is not None:
        return "Asia" if lon >= EUROPE_ASIA_LONGITUDE_SPLIT else "Europe"

    for continent, codes in _CONTINENT_LOOKUP:
        if country_code in codes:
            return continent
    return "Other"


GRID_DEGREES = 5 / 60
GRID_HALF_DEGREES = GRID_DEGREES / 2
MAX_LATITUDE_INDEX = 2159
MAX_LONGITUDE_INDEX = 4319
EARTH_RADIUS_KM = 6371.0
POPULATION_TIE_SCORE_WINDOW = 0.015
FULL_DIVERSITY_RANKS = 15
DIVERSITY_TAPER_RANKS = 20
MIN_DIVERSITY_STRENGTH = 0.2


@dataclass(frozen=True, slots=True)
class CityCandidate:
    """One user-facing place that can inherit a climate-cell score."""

    name: str
    country_code: str
    lat: float
    lon: float
    cell_lat: float
    cell_lon: float
    population: int = 0


class CityScorePoint(TypedDict):
    """JSON score payload for the ranked city list and map markers.

    `lat`/`lon` place the city marker and focus ring. `probe_lat`/`probe_lon`
    point at the snapped climate cell used for `/probe` scoring.
    """

    name: str
    continent: str
    country_code: str
    flag: str
    score: float
    lat: float
    lon: float
    probe_lat: float
    probe_lon: float


@dataclass(frozen=True, slots=True)
class RankedCityCandidate:
    """City plus its current ranking state during diversified selection."""

    city: CityCandidate
    score: float


@dataclass(frozen=True, slots=True)
class RegionalPenaltyCenter:
    """Winner context for one diversity-suppression step."""

    city: CityCandidate
    score: float
    strength: float = 1.0


def city_score_point(city: CityCandidate, score: float, *, flag: str | None = None) -> CityScorePoint:
    """Build the API score shape for one ranked city."""
    return {
        "name": city.name,
        "continent": continent_of(city.country_code, city.lon),
        "country_code": city.country_code,
        "flag": flag or country_flag(city.country_code),
        "score": round(score, 4),
        "lat": city.lat,
        "lon": city.lon,
        "probe_lat": city.cell_lat,
        "probe_lon": city.cell_lon,
    }


def _candidate_population(city: CityCandidate) -> int:
    return max(city.population, 0)


def _select_population_biased_winner(remaining: list[RankedCityCandidate]) -> RankedCityCandidate:
    """Prefer larger population centers when effective scores are nearly tied."""
    best_score = max(candidate.score for candidate in remaining)
    score_floor = best_score - POPULATION_TIE_SCORE_WINDOW
    near_tied = [candidate for candidate in remaining if candidate.score >= score_floor]
    return max(near_tied, key=lambda candidate: (_candidate_population(candidate.city), candidate.score))


def _select_population_biased_winner_index(
    city_catalog: CityRankingCache,
    scores: NDArray[np.float32],
    active: NDArray[np.bool],
) -> int:
    """Prefer larger population centers when effective scores are nearly tied."""
    active_scores = np.where(active, scores, -1.0)
    best_score = float(active_scores.max())
    near_tied = np.flatnonzero(active & (scores >= best_score - POPULATION_TIE_SCORE_WINDOW))
    return max(
        near_tied.tolist(),
        key=lambda index: (_candidate_population(city_catalog.cities[index]), float(scores[index])),
    )


@dataclass(frozen=True, slots=True)
class CityRankingCache:
    """Compact city ranking inputs, all positionally aligned with climate-matrix rows."""

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
        winner = _select_population_biased_winner(remaining)
        penalty_center = RegionalPenaltyCenter(winner.city, winner.score, _diversity_strength_for_rank(len(ranked)))
        ranked.append(city_score_point(winner.city, winner.score))
        remaining = [
            RankedCityCandidate(
                city=candidate.city,
                score=apply_regional_penalty(
                    candidate.score,
                    penalty_center,
                    candidate.city,
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
    """Rank cities from climate-matrix-aligned scores while keeping regional diversity suppression."""
    if not city_catalog.cities:
        return []

    scores = cell_scores[city_catalog.climate_indexes].astype(np.float32, copy=True)
    active = np.ones(scores.shape, dtype=bool)
    ranked: list[CityScorePoint] = []

    while active.any() and len(ranked) < limit:
        winner_index = _select_population_biased_winner_index(city_catalog, scores, active)
        winner_city = city_catalog.cities[winner_index]
        winner_score = float(scores[winner_index])
        diversity_strength = _diversity_strength_for_rank(len(ranked))
        ranked.append(city_score_point(winner_city, winner_score, flag=city_catalog.flags[winner_index]))

        distance_km = _haversine_distance_vector_km(city_catalog, winner_index)
        penalty = winner_score * diversity_strength * np.exp(-distance_km / diversity_decay_km)
        scores = np.where(active, np.maximum(0.0, scores * (1.0 - penalty)), scores)
        active[winner_index] = False

    return ranked


def _diversity_strength_for_rank(rank_index: int) -> float:
    """Taper regional spreading once the shortlist already covers the main hot zones."""
    if rank_index < FULL_DIVERSITY_RANKS:
        return 1.0

    taper_progress = min((rank_index - FULL_DIVERSITY_RANKS + 1) / DIVERSITY_TAPER_RANKS, 1.0)
    return 1.0 - (1.0 - MIN_DIVERSITY_STRENGTH) * taper_progress


def apply_regional_penalty(
    score: float,
    center: RegionalPenaltyCenter,
    candidate: CityCandidate,
    *,
    decay_km: float = CITY_DIVERSITY_DECAY_KM,
) -> float:
    """Exponentially suppress cities that cluster too close to a stronger regional center."""
    distance_km = haversine_distance_km(center.city.lat, center.city.lon, candidate.lat, candidate.lon)
    penalty = center.score * center.strength * math.exp(-distance_km / decay_km)
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
    """Vectorized winner-to-all distances for one diversity-penalty update."""
    latitude_radians = city_catalog.latitude_radians[winner_index]
    longitude_radians = city_catalog.longitude_radians[winner_index]
    cosine_latitude = city_catalog.cosine_latitudes[winner_index]
    delta_lat = city_catalog.latitude_radians - latitude_radians
    delta_lon = city_catalog.longitude_radians - longitude_radians
    half_chord = (
        np.sin(delta_lat / 2.0) ** 2 + cosine_latitude * city_catalog.cosine_latitudes * np.sin(delta_lon / 2.0) ** 2
    )
    return (2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(half_chord))).astype(np.float32, copy=False)


def snap_city_to_cell_key(city: CityCandidate, *, grid_degrees: float = GRID_DEGREES) -> tuple[float, float]:
    """Snap a city location to the nearest WorldClim cell center for one grid size."""
    grid_half_degrees = grid_degrees / 2
    max_latitude_index = round(180 / grid_degrees) - 1
    max_longitude_index = round(360 / grid_degrees) - 1
    latitude_index = round((90 - grid_half_degrees - city.lat) / grid_degrees)
    longitude_index = round((city.lon - (-180 + grid_half_degrees)) / grid_degrees)
    latitude_index = min(max(latitude_index, 0), max_latitude_index)
    longitude_index = min(max(longitude_index, 0), max_longitude_index)

    return (
        round(90 - grid_half_degrees - latitude_index * grid_degrees, 4),
        round(-180 + grid_half_degrees + longitude_index * grid_degrees, 4),
    )


def coordinate_key(latitude: float, longitude: float) -> int:
    """Encode a rounded lat/lon pair into one sortable integer key."""
    latitude_key = round(latitude * 10_000) + 900_000
    longitude_key = round(longitude * 10_000) + 1_800_000
    return latitude_key * 4_000_000 + longitude_key


def country_flag(country_code: str) -> str:
    """Convert ISO alpha-2 country codes into regional indicator flag emoji."""
    return "".join(chr(0x1F1E6 + ord(letter) - ord("A")) for letter in country_code.upper())
