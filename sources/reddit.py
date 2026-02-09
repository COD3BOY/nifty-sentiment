"""r/IndianStreetBets sentiment via PRAW + Claude."""

import logging

from analyzers.claude_analyzer import analyze_text
from core.config import get_env, get_source_config
from core.models import SentimentScore
from sources import register_source
from sources.base import DataSource

logger = logging.getLogger(__name__)


@register_source
class RedditSource(DataSource):
    @property
    def name(self) -> str:
        return "reddit"

    @property
    def source_type(self) -> str:
        return "social"

    async def fetch_sentiment(self) -> SentimentScore:
        cfg = get_source_config("reddit")
        subreddit_name = cfg.get("subreddit", "IndianStreetBets")
        post_limit = cfg.get("post_limit", 25)
        confidence_discount = cfg.get("confidence_discount", 0.7)

        try:
            import praw
        except ImportError:
            return SentimentScore(
                source_name=self.name, score=0.0, confidence=0.0,
                explanation="praw not installed",
            )

        client_id = get_env("REDDIT_CLIENT_ID")
        client_secret = get_env("REDDIT_CLIENT_SECRET")
        user_agent = get_env("REDDIT_USER_AGENT")

        if not client_id or not client_secret:
            return SentimentScore(
                source_name=self.name, score=0.0, confidence=0.0,
                explanation="Reddit API credentials not configured",
            )

        try:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
            subreddit = reddit.subreddit(subreddit_name)
            posts = list(subreddit.hot(limit=post_limit))

            if not posts:
                return SentimentScore(
                    source_name=self.name, score=0.0, confidence=0.0,
                    explanation=f"No posts found in r/{subreddit_name}",
                )

            # Combine post titles and top comments
            text_parts = []
            for post in posts:
                text_parts.append(f"Title: {post.title}")
                if post.selftext:
                    text_parts.append(f"Body: {post.selftext[:200]}")

            combined = "\n".join(text_parts)
            result = await analyze_text(
                combined,
                context=f"Reddit r/{subreddit_name} retail investor discussion",
            )

            return SentimentScore(
                source_name=self.name,
                score=max(-1.0, min(1.0, result.overall_score)),
                confidence=result.confidence * confidence_discount,
                explanation=result.summary,
                raw_data={
                    "subreddit": subreddit_name,
                    "posts_analyzed": len(posts),
                },
                bullish_factors=result.bullish_factors,
                bearish_factors=result.bearish_factors,
            )

        except Exception as e:
            logger.error(f"Reddit fetch failed: {e}")
            return SentimentScore(
                source_name=self.name, score=0.0, confidence=0.0,
                explanation=f"Error: {e}",
            )

    async def is_available(self) -> bool:
        try:
            import praw  # noqa: F401
            return bool(get_env("REDDIT_CLIENT_ID"))
        except ImportError:
            return False
