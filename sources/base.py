"""Abstract base class for all data sources."""

from abc import ABC, abstractmethod

from core.models import SentimentScore


class DataSource(ABC):
    """Base class that all sentiment data sources must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this source."""
        ...

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Category of this source (e.g., 'news', 'market', 'social')."""
        ...

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    async def fetch_sentiment(self) -> SentimentScore:
        """Fetch data and return a normalized sentiment score."""
        ...

    async def is_available(self) -> bool:
        """Check if this source is currently available (API keys set, etc)."""
        return True
