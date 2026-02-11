"""Algorithm registry with auto-discovery via @register_algorithm decorator."""

from algorithms.base import TradingAlgorithm

_REGISTRY: dict[str, type[TradingAlgorithm]] = {}


def register_algorithm(cls: type[TradingAlgorithm]) -> type[TradingAlgorithm]:
    """Decorator to register a TradingAlgorithm implementation."""
    _REGISTRY[cls.name] = cls
    return cls


def get_algorithm_registry() -> dict[str, type[TradingAlgorithm]]:
    return dict(_REGISTRY)


def discover_algorithms() -> None:
    """Import all algorithm modules to trigger @register_algorithm decorators."""
    from algorithms import (  # noqa: F401
        sentinel,
        institutional,
    )
