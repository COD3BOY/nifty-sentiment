"""Source registry with auto-discovery via @register_source decorator."""

from sources.base import DataSource

_REGISTRY: dict[str, type[DataSource]] = {}


def register_source(cls: type[DataSource]) -> type[DataSource]:
    """Decorator to register a DataSource implementation."""
    _REGISTRY[cls.name.fget(None)] = cls  # type: ignore[union-attr]
    return cls


def get_registry() -> dict[str, type[DataSource]]:
    return dict(_REGISTRY)


def discover_sources() -> None:
    """Import all source modules to trigger @register_source decorators."""
    from sources import (  # noqa: F401
        crude_oil,
        fii_dii,
        gift_nifty,
        global_markets,
        reddit,
        vix,
        youtube,
        zerodha_pulse,
    )
