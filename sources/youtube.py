"""YouTube financial advisor transcript analysis."""

import logging

from analyzers.claude_analyzer import analyze_text
from core.config import get_source_config
from core.models import SentimentScore
from sources import register_source
from sources.base import DataSource

logger = logging.getLogger(__name__)


@register_source
class YouTubeSource(DataSource):
    @property
    def name(self) -> str:
        return "youtube"

    @property
    def source_type(self) -> str:
        return "social"

    async def fetch_sentiment(self) -> SentimentScore:
        cfg = get_source_config("youtube")

        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return SentimentScore(
                source_name=self.name, score=0.0, confidence=0.0,
                explanation="youtube-transcript-api not installed",
            )

        from core.config import get_env
        api_key = get_env("YOUTUBE_API_KEY")
        channels = cfg.get("channels", [])

        if not channels:
            return SentimentScore(
                source_name=self.name, score=0.0, confidence=0.0,
                explanation="No YouTube channels configured",
            )

        # Fetch recent video transcripts and analyze
        all_text = []
        videos_analyzed = 0

        for channel in channels:
            try:
                # Search for recent videos from the channel
                import requests
                if api_key:
                    resp = requests.get(
                        "https://www.googleapis.com/youtube/v3/search",
                        params={
                            "part": "snippet",
                            "q": f"{channel} market today",
                            "type": "video",
                            "maxResults": 1,
                            "order": "date",
                            "key": api_key,
                        },
                        timeout=10,
                    )
                    data = resp.json()
                    items = data.get("items", [])
                else:
                    items = []

                for item in items:
                    video_id = item["id"]["videoId"]
                    try:
                        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "hi"])
                        text = " ".join(entry["text"] for entry in transcript[:100])
                        all_text.append(f"[{channel}]: {text}")
                        videos_analyzed += 1
                    except Exception as e:
                        logger.debug(f"No transcript for {video_id}: {e}")

            except Exception as e:
                logger.warning(f"YouTube search failed for {channel}: {e}")

        if not all_text:
            return SentimentScore(
                source_name=self.name, score=0.0, confidence=0.0,
                explanation="No YouTube transcripts available",
            )

        combined_text = "\n\n".join(all_text)
        result = await analyze_text(combined_text, context="YouTube financial advisor commentary")

        return SentimentScore(
            source_name=self.name,
            score=max(-1.0, min(1.0, result.overall_score)),
            confidence=result.confidence * 0.8,  # discount for noise
            explanation=result.summary,
            raw_data={"videos_analyzed": videos_analyzed, "channels": channels},
            bullish_factors=result.bullish_factors,
            bearish_factors=result.bearish_factors,
        )

    async def is_available(self) -> bool:
        try:
            import youtube_transcript_api  # noqa: F401
            return True
        except ImportError:
            return False
