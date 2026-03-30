from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreferenceField:
    """Default UI configuration for a single preference control."""

    name: str
    label: str
    minimum: int
    maximum: int
    step: int
    value: int


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
        name="rain_tolerance",
        label="Rain tolerance",
        minimum=0,
        maximum=100,
        step=5,
        value=35,
    ),
    PreferenceField(
        name="cloud_tolerance",
        label="Cloud tolerance",
        minimum=0,
        maximum=100,
        step=5,
        value=40,
    ),
)
