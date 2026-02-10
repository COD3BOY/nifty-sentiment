"""Claude API trade criticizer with structured outputs.

Analyzes every closed trade and produces structured insights for future learning.
Follows the same pattern as analyzers/claude_analyzer.py.
"""

import json
import logging

import anthropic

from core.config import get_env, load_config
from core.criticizer_models import ParameterRecommendation, TradeCritique
from core.paper_trading_models import TradeRecord
from core.strategy_rules import STRATEGY_RULES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON Schema for structured output
# ---------------------------------------------------------------------------

TRADE_CRITIQUE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overall_grade": {
            "type": "string",
            "enum": ["excellent", "good", "acceptable", "poor", "terrible"],
            "description": "Overall grade for this trade",
        },
        "pnl_assessment": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "outcome_quality": {
                    "type": "string",
                    "description": "Assessment of the P&L outcome (e.g., 'strong profit', 'small loss within expectations')",
                },
                "was_exit_timing_good": {
                    "type": "boolean",
                    "description": "Whether the exit timing was appropriate",
                },
                "risk_reward_actual": {
                    "type": "string",
                    "description": "Actual risk/reward achieved vs. what was expected",
                },
            },
            "required": ["outcome_quality", "was_exit_timing_good", "risk_reward_actual"],
        },
        "entry_signal_analysis": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "signals_that_worked": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Signals that correctly predicted the outcome",
                },
                "signals_that_failed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Signals that were incorrect or misleading",
                },
                "signals_missed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Important signals that were not considered",
                },
            },
            "required": ["signals_that_worked", "signals_that_failed", "signals_missed"],
        },
        "strategy_fitness": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "was_right_strategy": {
                    "type": "boolean",
                    "description": "Whether this was the best strategy for the market conditions",
                },
                "better_strategy": {
                    "type": "string",
                    "description": "A better strategy that could have been used, or 'none' if the chosen strategy was appropriate",
                },
                "market_regime_match": {
                    "type": "string",
                    "description": "How well the strategy matched the actual market regime (e.g., 'trending', 'range-bound')",
                },
            },
            "required": ["was_right_strategy", "better_strategy", "market_regime_match"],
        },
        "parameter_recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "strategy_name": {"type": "string"},
                    "parameter_name": {"type": "string", "description": "Must match a param_key from the scoring rules (e.g., 'iv_low_threshold', 'rsi_neutral_low')"},
                    "current_value": {"type": "number"},
                    "recommended_value": {"type": "number"},
                    "confidence": {"type": "number", "description": "0.0 to 1.0"},
                    "reasoning": {"type": "string"},
                    "condition": {"type": "string", "description": "e.g., 'when RSI > 55'"},
                },
                "required": [
                    "strategy_name", "parameter_name", "current_value",
                    "recommended_value", "confidence", "reasoning", "condition",
                ],
            },
        },
        "patterns_observed": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Recurring patterns noticed in this trade",
        },
        "risk_management_notes": {
            "type": "string",
            "description": "Assessment of stop-loss and profit-target settings",
        },
        "summary": {
            "type": "string",
            "description": "2-3 sentence executive summary of the critique",
        },
    },
    "required": [
        "overall_grade", "pnl_assessment", "entry_signal_analysis",
        "strategy_fitness", "parameter_recommendations", "patterns_observed",
        "risk_management_notes", "summary",
    ],
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert NIFTY options analyst reviewing completed paper trades.

Your job is to critically analyze a closed trade and provide:
1. An overall grade (excellent/good/acceptable/poor/terrible)
2. Assessment of entry signals — which worked, which failed, which were missed
3. Whether the chosen strategy was optimal for the market conditions
4. Conservative parameter recommendations (if any)
5. Patterns observed for future learning

IMPORTANT GUIDELINES:
- Be conservative with parameter recommendations. A single trade should NOT cause dramatic threshold shifts.
- Only recommend parameter changes with confidence > 0.7 if you are very sure.
- Reference actual indicator values from the entry/exit contexts in your analysis.
- Consider the full trade lifecycle: entry conditions, how the trade evolved (max favorable, max drawdown), and exit conditions.
- Compare entry vs exit market states to understand what changed during the hold period.
- When recommending parameter changes, use the `param_key` field from the scoring rules as the `parameter_name` in your recommendations. This ensures the system can automatically apply your suggestions. Only recommend changes to parameters that have a `param_key` defined.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def criticize_trade(
    record: TradeRecord,
    recent_performance: list[dict] | None = None,
) -> TradeCritique:
    """Analyze a closed trade and return a structured critique.

    Args:
        record: The closed trade record with entry/exit contexts.
        recent_performance: Optional list of recent trade dicts for this strategy
                           (last 10 trades' win/loss + grades).
    """
    config = load_config()
    api_key = get_env("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    model = config.get("claude", {}).get("model", "claude-sonnet-4-5-20250929")
    max_tokens = config.get("claude", {}).get("max_tokens", 4096)

    # Build the user prompt
    strategy_name = record.strategy
    rules = STRATEGY_RULES.get(strategy_name, {})

    # Format entry/exit contexts
    entry_ctx = record.entry_context or {}
    exit_ctx = record.exit_context or {}

    # Duration
    duration_mins = 0
    if record.entry_time and record.exit_time:
        duration_mins = int((record.exit_time - record.entry_time).total_seconds() / 60)

    # Recent performance summary
    perf_text = "No recent performance data available."
    if recent_performance:
        wins = sum(1 for t in recent_performance if t.get("overall_grade") in ("excellent", "good"))
        losses = sum(1 for t in recent_performance if t.get("overall_grade") in ("poor", "terrible"))
        perf_text = f"Last {len(recent_performance)} trades for {strategy_name}: {wins} good/excellent, {losses} poor/terrible"

    prompt = f"""Analyze this completed NIFTY paper trade:

## Strategy: {strategy_name}
### Current Scoring Rules
{json.dumps(rules, indent=2)}

### Entry Context (Market State at Entry)
{json.dumps(entry_ctx, indent=2)}

### Exit Context (Market State at Exit)
{json.dumps(exit_ctx, indent=2)}

### Trade Outcome
- Direction Bias: {record.direction_bias}
- Confidence: {record.confidence}
- Strategy Score at Entry: {record.score}
- Realized P&L: {record.realized_pnl:,.2f}
- Net P&L (after costs): {record.net_pnl:,.2f}
- Execution Cost: {record.execution_cost:,.2f}
- Duration: {duration_mins} minutes
- Exit Reason: {record.exit_reason.value}
- Max Drawdown: {record.max_drawdown:,.2f}
- Max Favorable: {record.max_favorable:,.2f}
- Spot at Entry: {record.spot_at_entry:.2f}
- Spot at Exit: {record.spot_at_exit:.2f}
- Spot Movement: {record.spot_at_exit - record.spot_at_entry:+.2f} ({((record.spot_at_exit - record.spot_at_entry) / record.spot_at_entry * 100) if record.spot_at_entry > 0 else 0:+.3f}%)
- Net Premium: {record.net_premium:,.2f}
- Stop Loss: {record.stop_loss_amount:,.2f}
- Profit Target: {record.profit_target_amount:,.2f}

### Legs Summary
{json.dumps(record.legs_summary, indent=2)}

### Recent Performance for this Strategy
{perf_text}

Please provide a thorough critique following the output schema exactly."""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": TRADE_CRITIQUE_SCHEMA,
            },
        },
    )

    result = json.loads(response.content[0].text)

    # Build TradeCritique from response
    param_recs = [
        ParameterRecommendation(**rec)
        for rec in result.get("parameter_recommendations", [])
    ]

    return TradeCritique(
        trade_id=record.id,
        overall_grade=result["overall_grade"],
        pnl_assessment=result["pnl_assessment"],
        entry_signal_analysis=result["entry_signal_analysis"],
        strategy_fitness=result["strategy_fitness"],
        parameter_recommendations=param_recs,
        patterns_observed=result.get("patterns_observed", []),
        risk_management_notes=result.get("risk_management_notes", ""),
        summary=result.get("summary", ""),
    )
