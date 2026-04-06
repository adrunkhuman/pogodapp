from dataclasses import dataclass

CITY_DIVERSITY_DECAY_KM = 500.0
# population=0 means a pre-migration DB row; don't filter it out.
RANKING_MIN_POPULATION = 30_000
WEB_MERCATOR_MAX_LATITUDE = 85.051129

PREFERENCE_FIELD_NAMES: tuple[str, ...] = (
    "preferred_day_temperature",
    "summer_heat_limit",
    "winter_cold_limit",
    "dryness_preference",
    "sunshine_preference",
)


@dataclass(frozen=True, slots=True)
class PreferenceField:
    """Canonical UI contract for a single climate preference field."""

    name: str
    label: str
    minimum: int
    maximum: int
    step: int
    value: int
    description: str
    low_label: str
    high_label: str


@dataclass(frozen=True, slots=True)
class MapProjection:
    """Shared map projection settings for frontend rendering and heatmap output."""

    name: str
    image_corners: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    max_render_latitude: float | None = None


MAP_PROJECTION = MapProjection(
    name="mercator",
    image_corners=(
        (-180.0, WEB_MERCATOR_MAX_LATITUDE),
        (180.0, WEB_MERCATOR_MAX_LATITUDE),
        (180.0, -WEB_MERCATOR_MAX_LATITUDE),
        (-180.0, -WEB_MERCATOR_MAX_LATITUDE),
    ),
    max_render_latitude=WEB_MERCATOR_MAX_LATITUDE,
)


# These ranges define the current UI and `/score` contract.
# The scoring still runs on monthly mean temperature until the dataset grows
# dedicated high/low temperature normals.
DEFAULT_PREFERENCES: tuple[PreferenceField, ...] = (
    PreferenceField(
        name="preferred_day_temperature",
        label="Typical day",
        minimum=5,
        maximum=35,
        step=1,
        value=22,
        description="What should a normal daytime temperature feel like most of the year?",
        low_label="cool",
        high_label="warm",
    ),
    PreferenceField(
        name="summer_heat_limit",
        label="Summer limit",
        minimum=18,
        maximum=42,
        step=1,
        value=30,
        description="How hot can warmer months get before the place starts feeling too hot?",
        low_label="avoid heat",
        high_label="heat is fine",
    ),
    PreferenceField(
        name="winter_cold_limit",
        label="Winter limit",
        minimum=-15,
        maximum=20,
        step=1,
        value=5,
        description="How cold can cooler months get before the place starts feeling too cold?",
        low_label="handle cold",
        high_label="keep it mild",
    ),
    PreferenceField(
        name="dryness_preference",
        label="Dryness",
        minimum=0,
        maximum=100,
        step=5,
        value=60,
        description="Do you want mostly dry weather, or is regular rain acceptable?",
        low_label="rain is okay",
        high_label="prefer dry",
    ),
    PreferenceField(
        name="sunshine_preference",
        label="Sunshine",
        minimum=0,
        maximum=100,
        step=5,
        value=60,
        description="How important are bright, sunny skies?",
        low_label="clouds are fine",
        high_label="need sun",
    ),
)
