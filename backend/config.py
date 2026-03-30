from dataclasses import dataclass

PREFERENCE_FIELD_NAMES: tuple[str, ...] = (
    "ideal_temperature",
    "cold_tolerance",
    "heat_tolerance",
    "rain_sensitivity",
    "sun_preference",
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


# Higher sensitivity means a stronger score penalty.
# `sun_preference` stays user-facing while later scoring maps it onto cloud-cover data.
# Exact scoring curves stay unresolved until the scoring issue lands.
DEFAULT_PREFERENCES: tuple[PreferenceField, ...] = (
    PreferenceField(
        name="ideal_temperature",
        label="Ideal temperature (C)",
        minimum=-10,
        maximum=35,
        step=1,
        value=22,
    ),
    PreferenceField(
        name="cold_tolerance",
        label="Cold tolerance",
        minimum=0,
        maximum=15,
        step=1,
        value=7,
    ),
    PreferenceField(
        name="heat_tolerance",
        label="Heat tolerance",
        minimum=0,
        maximum=15,
        step=1,
        value=5,
    ),
    PreferenceField(
        name="rain_sensitivity",
        label="Rain sensitivity",
        minimum=0,
        maximum=100,
        step=5,
        value=55,
    ),
    PreferenceField(
        name="sun_preference",
        label="Sun preference",
        minimum=0,
        maximum=100,
        step=5,
        value=60,
    ),
)
