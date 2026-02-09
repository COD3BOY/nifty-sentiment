"""FII/DII institutional flows via nsepython."""

import logging
from datetime import datetime

from core.config import get_source_config
from core.models import FIIDIIData, SentimentScore
from sources import register_source
from sources.base import DataSource

logger = logging.getLogger(__name__)


@register_source
class FIIDIISource(DataSource):
    @property
    def name(self) -> str:
        return "fii_dii"

    @property
    def source_type(self) -> str:
        return "institutional"

    async def fetch_sentiment(self) -> SentimentScore:
        cfg = get_source_config("fii_dii")
        significant_flow_cr = cfg.get("significant_flow_cr", 2000)
        fii_weight = cfg.get("fii_weight", 0.7)
        dii_weight = cfg.get("dii_weight", 0.3)

        try:
            from nsepython import nse_fiidii
            data = nse_fiidii()

            if data is None or data.empty:
                return SentimentScore(
                    source_name=self.name,
                    score=0.0,
                    confidence=0.0,
                    explanation="No FII/DII data available",
                )

            # nsepython returns rows with columns: category, date, buyValue, sellValue, netValue
            # Separate rows for "FII/FPI" and "DII"
            fii_net = 0.0
            dii_net = 0.0
            for _, row in data.iterrows():
                cat = str(row.get("category", "")).upper()
                net = float(row.get("netValue", 0))
                if "FII" in cat or "FPI" in cat:
                    fii_net = net
                elif "DII" in cat:
                    dii_net = net

            flow_data = FIIDIIData(
                date=datetime.utcnow(),
                fii_net=fii_net,
                dii_net=dii_net,
            )

            # Combined signal: FII buying is bullish, DII buying is moderately bullish
            combined_flow = fii_net * fii_weight + dii_net * dii_weight
            score = max(-1.0, min(1.0, combined_flow / significant_flow_cr))

            bullish = []
            bearish = []
            if fii_net > 100:
                bullish.append(f"FII net buying: ₹{fii_net:.0f} Cr")
            elif fii_net < -100:
                bearish.append(f"FII net selling: ₹{fii_net:.0f} Cr")
            if dii_net > 100:
                bullish.append(f"DII net buying: ₹{dii_net:.0f} Cr")
            elif dii_net < -100:
                bearish.append(f"DII net selling: ₹{dii_net:.0f} Cr")

            return SentimentScore(
                source_name=self.name,
                score=score,
                confidence=0.8,
                explanation=f"FII net: ₹{fii_net:.0f} Cr, DII net: ₹{dii_net:.0f} Cr",
                raw_data=flow_data.model_dump(mode="json"),
                bullish_factors=bullish,
                bearish_factors=bearish,
            )

        except ImportError:
            logger.warning("nsepython not installed")
            return SentimentScore(
                source_name=self.name, score=0.0, confidence=0.0,
                explanation="nsepython not installed",
            )
        except Exception as e:
            logger.error(f"FII/DII fetch failed: {e}")
            return SentimentScore(
                source_name=self.name, score=0.0, confidence=0.0,
                explanation=f"Error: {e}",
            )
