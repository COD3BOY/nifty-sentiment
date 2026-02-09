"""Claude API sentiment analyzer with structured outputs."""

import logging

import anthropic

from core.config import get_env, load_config
from core.models import ClaudeSentimentResponse

logger = logging.getLogger(__name__)

SENTIMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overall_score": {
            "type": "number",
            "description": "Overall sentiment score from -1.0 (very bearish) to 1.0 (very bullish)",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the analysis from 0.0 to 1.0",
        },
        "headline_sentiments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "headline": {"type": "string"},
                    "score": {"type": "number"},
                    "reasoning": {"type": "string"},
                },
                "required": ["headline", "score", "reasoning"],
            },
        },
        "bullish_factors": {
            "type": "array",
            "items": {"type": "string"},
        },
        "bearish_factors": {
            "type": "array",
            "items": {"type": "string"},
        },
        "summary": {"type": "string"},
    },
    "required": [
        "overall_score",
        "confidence",
        "headline_sentiments",
        "bullish_factors",
        "bearish_factors",
        "summary",
    ],
}


async def analyze_headlines(headlines: list[str]) -> ClaudeSentimentResponse:
    """Analyze news headlines for NIFTY market sentiment using Claude."""
    config = load_config()
    api_key = get_env("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    model = config.get("claude", {}).get("model", "claude-sonnet-4-5-20250929")
    max_tokens = config.get("claude", {}).get("max_tokens", 1024)

    headlines_text = "\n".join(f"- {h}" for h in headlines)
    prompt = f"""Analyze these Indian financial market news headlines and determine the overall sentiment for NIFTY 50 (Indian stock market index) for today's trading session.

Headlines:
{headlines_text}

Consider:
- Direct impact on Indian markets (policy changes, RBI decisions, earnings)
- Indirect impact (global cues, FII flows, sector-specific news)
- Market psychology and momentum signals

Score from -1.0 (very bearish) to 1.0 (very bullish). Be calibrated - most days are near 0, only use extremes for truly significant events."""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": SENTIMENT_SCHEMA,
            },
        },
    )

    import json
    result = json.loads(response.content[0].text)
    return ClaudeSentimentResponse(**result)


async def analyze_text(text: str, context: str = "general market") -> ClaudeSentimentResponse:
    """Analyze arbitrary text for market sentiment."""
    config = load_config()
    api_key = get_env("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    model = config.get("claude", {}).get("model", "claude-sonnet-4-5-20250929")
    max_tokens = config.get("claude", {}).get("max_tokens", 1024)

    prompt = f"""Analyze the following {context} content and determine the sentiment for NIFTY 50 (Indian stock market).

Content:
{text[:3000]}

Score from -1.0 (very bearish) to 1.0 (very bullish). Be calibrated."""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": SENTIMENT_SCHEMA,
            },
        },
    )

    import json
    result = json.loads(response.content[0].text)
    return ClaudeSentimentResponse(**result)
