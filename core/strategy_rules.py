"""Machine-readable map of all 11 strategies' scoring rules.

Each entry lists every threshold used by the strategy evaluator in
trade_strategies.py. This is included in the Claude criticizer prompt so it
knows exactly what signals each strategy checks, and serves as the parameter
registry for future automated tuning.
"""

STRATEGY_RULES: dict[str, dict] = {
    "Short Straddle": {
        "direction": "Neutral",
        "type": "credit",
        "scoring_rules": [
            {"signal": "atm_iv", "condition": "< 15", "score_change": "+25", "description": "Low IV favorable for selling"},
            {"signal": "atm_iv", "condition": "15-20", "score_change": "+15", "description": "Moderate IV"},
            {"signal": "atm_iv", "condition": ">= 20", "score_change": "-10", "description": "High IV unfavorable"},
            {"signal": "rsi", "condition": "40-60", "score_change": "+20", "description": "Neutral RSI zone"},
            {"signal": "rsi", "condition": "30-70", "score_change": "+5", "description": "Moderate RSI"},
            {"signal": "rsi", "condition": "outside 30-70", "score_change": "-15", "description": "Extreme RSI"},
            {"signal": "spot_vs_max_pain", "condition": "< 0.3%", "score_change": "+20", "description": "Spot within 0.3% of max pain"},
            {"signal": "spot_vs_max_pain", "condition": "0.3-0.8%", "score_change": "+10", "description": "Spot near max pain"},
            {"signal": "spot_vs_max_pain", "condition": "> 0.8%", "score_change": "-10", "description": "Spot far from max pain"},
            {"signal": "pcr", "condition": "0.8-1.2", "score_change": "+15", "description": "Balanced PCR"},
            {"signal": "pcr", "condition": "outside 0.8-1.2", "score_change": "-5", "description": "Skewed PCR"},
            {"signal": "bb_width_pct", "condition": "< 1.0%", "score_change": "+10", "description": "Tight Bollinger Bands"},
        ],
    },
    "Short Strangle": {
        "direction": "Neutral",
        "type": "credit",
        "scoring_rules": [
            {"signal": "atm_iv", "condition": "< 18", "score_change": "+20", "description": "Low IV favorable"},
            {"signal": "atm_iv", "condition": "18-22", "score_change": "+10", "description": "Moderate IV"},
            {"signal": "support_resistance", "condition": "both present", "score_change": "+15", "description": "Strong support and resistance"},
            {"signal": "rsi", "condition": "35-65", "score_change": "+15", "description": "No extreme momentum"},
            {"signal": "spot_in_range", "condition": "between support-resistance", "score_change": "+15", "description": "Spot within band"},
            {"signal": "bb_width_pct", "condition": "< 1.2%", "score_change": "+10", "description": "Compressed BB"},
        ],
    },
    "Long Straddle": {
        "direction": "Neutral",
        "type": "debit",
        "scoring_rules": [
            {"signal": "bb_width_pct", "condition": "< 0.8%", "score_change": "+25", "description": "BB squeeze — breakout expected"},
            {"signal": "bb_width_pct", "condition": "0.8-1.0%", "score_change": "+15", "description": "Moderate BB compression"},
            {"signal": "iv_skew", "condition": "abs > 3", "score_change": "+20", "description": "Significant IV imbalance"},
            {"signal": "atm_iv", "condition": "> 18", "score_change": "+10", "description": "Elevated IV implies movement"},
            {"signal": "atm_iv", "condition": "<= 18", "score_change": "-10", "description": "Low IV — no movement expected"},
            {"signal": "rsi", "condition": "45-55", "score_change": "+10", "description": "Coiled RSI ready to break"},
        ],
    },
    "Long Strangle": {
        "direction": "Neutral",
        "type": "debit",
        "scoring_rules": [
            {"signal": "bb_width_pct", "condition": "< 1.0%", "score_change": "+20", "description": "Breakout setup"},
            {"signal": "atm_iv", "condition": "< 16", "score_change": "+20", "description": "Cheap premiums for longs"},
            {"signal": "atm_iv", "condition": "16-20", "score_change": "+10", "description": "Moderate premiums"},
            {"signal": "iv_skew", "condition": "abs > 2", "score_change": "+10", "description": "Directional pressure building"},
        ],
    },
    "Bull Put Spread": {
        "direction": "Bullish",
        "type": "credit",
        "scoring_rules": [
            {"signal": "supertrend_direction", "condition": "== 1", "score_change": "+15", "description": "Supertrend bullish"},
            {"signal": "spot_vs_vwap", "condition": "spot > vwap", "score_change": "+15", "description": "Spot above VWAP"},
            {"signal": "ema_alignment", "condition": "ema_9 > ema_21 > ema_50", "score_change": "+10", "description": "Bullish EMA alignment"},
            {"signal": "pcr", "condition": "> 1.2", "score_change": "+15", "description": "Heavy put writing (bullish)"},
            {"signal": "pcr", "condition": "1.0-1.2", "score_change": "+5", "description": "Moderate put writing"},
            {"signal": "rsi", "condition": "< 70", "score_change": "+5", "description": "Not overbought"},
            {"signal": "rsi", "condition": ">= 70", "score_change": "-10", "description": "Overbought — risk of reversal"},
        ],
    },
    "Bear Call Spread": {
        "direction": "Bearish",
        "type": "credit",
        "scoring_rules": [
            {"signal": "supertrend_direction", "condition": "== -1", "score_change": "+15", "description": "Supertrend bearish"},
            {"signal": "spot_vs_vwap", "condition": "spot < vwap", "score_change": "+15", "description": "Spot below VWAP"},
            {"signal": "ema_alignment", "condition": "ema_9 < ema_21 < ema_50", "score_change": "+10", "description": "Bearish EMA alignment"},
            {"signal": "pcr", "condition": "< 0.7", "score_change": "+15", "description": "Low put writing (bearish)"},
            {"signal": "pcr", "condition": "0.7-1.0", "score_change": "+5", "description": "Moderate call bias"},
            {"signal": "rsi", "condition": "> 30", "score_change": "+5", "description": "Not oversold"},
            {"signal": "rsi", "condition": "<= 30", "score_change": "-10", "description": "Oversold — risk of bounce"},
        ],
    },
    "Bull Call Spread": {
        "direction": "Bullish",
        "type": "debit",
        "scoring_rules": [
            {"signal": "ema_alignment", "condition": "ema_9 > ema_21 > ema_50", "score_change": "+20", "description": "Bullish EMA alignment"},
            {"signal": "supertrend_direction", "condition": "== 1", "score_change": "+15", "description": "Supertrend bullish"},
            {"signal": "spot_vs_vwap", "condition": "spot > vwap", "score_change": "+10", "description": "Spot above VWAP"},
            {"signal": "rsi", "condition": "50-70", "score_change": "+15", "description": "Bullish momentum, not overbought"},
            {"signal": "rsi", "condition": ">= 70", "score_change": "-10", "description": "Overbought"},
        ],
    },
    "Bear Put Spread": {
        "direction": "Bearish",
        "type": "debit",
        "scoring_rules": [
            {"signal": "ema_alignment", "condition": "ema_9 < ema_21 < ema_50", "score_change": "+20", "description": "Bearish EMA alignment"},
            {"signal": "supertrend_direction", "condition": "== -1", "score_change": "+15", "description": "Supertrend bearish"},
            {"signal": "spot_vs_vwap", "condition": "spot < vwap", "score_change": "+10", "description": "Spot below VWAP"},
            {"signal": "rsi", "condition": "30-50", "score_change": "+15", "description": "Bearish momentum, not oversold"},
            {"signal": "rsi", "condition": "<= 30", "score_change": "-10", "description": "Oversold"},
        ],
    },
    "Iron Condor": {
        "direction": "Neutral",
        "type": "credit",
        "scoring_rules": [
            {"signal": "rsi", "condition": "40-60", "score_change": "+20", "description": "Perfectly neutral RSI"},
            {"signal": "rsi", "condition": "35-65", "score_change": "+10", "description": "Near-neutral RSI"},
            {"signal": "bb_width_pct", "condition": "< 1.0%", "score_change": "+15", "description": "Tight range"},
            {"signal": "atm_iv", "condition": "< 18", "score_change": "+15", "description": "Moderate IV good for selling"},
            {"signal": "pcr", "condition": "0.85-1.15", "score_change": "+10", "description": "Balanced PCR"},
            {"signal": "spot_in_range", "condition": "between support-resistance", "score_change": "+10", "description": "Spot in range"},
        ],
    },
    "Long Call (CE)": {
        "direction": "Bullish",
        "type": "debit",
        "scoring_rules": [
            {"signal": "all_bullish", "condition": "ema_bullish + supertrend_1 + spot>vwap", "score_change": "+30", "description": "All signals aligned bullish"},
            {"signal": "ema_alignment", "condition": "ema_9 > ema_21 > ema_50", "score_change": "+10", "description": "Bullish EMA (partial)"},
            {"signal": "supertrend_direction", "condition": "== 1", "score_change": "+10", "description": "Supertrend bullish (partial)"},
            {"signal": "spot_vs_vwap", "condition": "spot > vwap", "score_change": "+5", "description": "Above VWAP (partial)"},
            {"signal": "rsi", "condition": "< 70", "score_change": "+10", "description": "Room to run higher"},
            {"signal": "rsi", "condition": ">= 70", "score_change": "-15", "description": "Overbought"},
            {"signal": "pcr", "condition": "> 1.2", "score_change": "+10", "description": "Bullish options flow"},
        ],
    },
    "Long Put (PE)": {
        "direction": "Bearish",
        "type": "debit",
        "scoring_rules": [
            {"signal": "all_bearish", "condition": "ema_bearish + supertrend_-1 + spot<vwap", "score_change": "+30", "description": "All signals aligned bearish"},
            {"signal": "ema_alignment", "condition": "ema_9 < ema_21 < ema_50", "score_change": "+10", "description": "Bearish EMA (partial)"},
            {"signal": "supertrend_direction", "condition": "== -1", "score_change": "+10", "description": "Supertrend bearish (partial)"},
            {"signal": "spot_vs_vwap", "condition": "spot < vwap", "score_change": "+5", "description": "Below VWAP (partial)"},
            {"signal": "rsi", "condition": "> 30", "score_change": "+10", "description": "Room to fall further"},
            {"signal": "rsi", "condition": "<= 30", "score_change": "-15", "description": "Oversold"},
            {"signal": "pcr", "condition": "< 0.7", "score_change": "+10", "description": "Bearish options flow"},
        ],
    },
}
