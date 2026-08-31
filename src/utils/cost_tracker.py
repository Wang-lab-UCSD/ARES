"""Cost tracking and estimation for LLM API calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from src.utils.logging import get_logger


@dataclass
class ModelPricing:
    """Pricing information for a model (per 1M tokens)."""

    input_price: float  # USD per 1M input tokens
    output_price: float  # USD per 1M output tokens
    provider: str = ""
    model_id: str = ""
    context_limit: int = 128000  # Default context window size in tokens
    cached_input_price: float = 0.0  # USD per 1M cached input tokens (0 = not supported)


# Pricing table. Every rate below was checked against the provider's own pricing page on
# 2026-08-31; entries that page no longer lists are marked UNVERIFIED where they appear.
#   OpenAI     https://developers.openai.com/api/docs/pricing
#   Anthropic  https://docs.anthropic.com/en/docs/about-claude/pricing
#   Google     https://ai.google.dev/gemini-api/docs/pricing
#   DeepSeek   https://api-docs.deepseek.com/quick_start/pricing
#   MiniMax    https://platform.minimax.io/docs/guides/pricing-paygo
#   Z.AI       https://docs.z.ai/guides/overview/pricing
# A model missing from this table does not break anything: the pipeline runs it normally and
# only the cost estimate degrades -- get_pricing returns None, estimate_cost logs a warning and
# falls back to a deliberately high $10/$40 per 1M guess, which also feeds the per-call and
# session budget checks, so a cheap unlisted model can trip a budget far earlier than its real
# cost warrants. Add it here to get accurate numbers.
# Context limits are the maximum input tokens the model can accept
MODEL_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-5.2": ModelPricing(1.75, 14.0, "openai", "gpt-5.2", context_limit=400000, cached_input_price=0.175),
    "gpt-5.2-chat-latest": ModelPricing(1.75, 14.0, "openai", "gpt-5.2-chat-latest", context_limit=400000, cached_input_price=0.175),
    "gpt-5.2-codex": ModelPricing(1.75, 14.0, "openai", "gpt-5.2-codex", context_limit=400000, cached_input_price=0.175),
    "gpt-5.2-pro": ModelPricing(21.0, 168.0, "openai", "gpt-5.2-pro", context_limit=256000),
    "gpt-5.1": ModelPricing(1.25, 10.0, "openai", "gpt-5.1", context_limit=256000, cached_input_price=0.125),
    "gpt-5.1-chat-latest": ModelPricing(1.25, 10.0, "openai", "gpt-5.1-chat-latest", context_limit=256000, cached_input_price=0.125),
    "gpt-5.1-codex-max": ModelPricing(1.25, 10.0, "openai", "gpt-5.1-codex-max", context_limit=256000, cached_input_price=0.125),
    "gpt-5.1-codex": ModelPricing(1.25, 10.0, "openai", "gpt-5.1-codex", context_limit=256000, cached_input_price=0.125),
    "gpt-5": ModelPricing(1.25, 10.0, "openai", "gpt-5", context_limit=256000, cached_input_price=0.125),
    "gpt-5-chat-latest": ModelPricing(1.25, 10.0, "openai", "gpt-5-chat-latest", context_limit=256000),
    "gpt-5-codex": ModelPricing(1.25, 10.0, "openai", "gpt-5-codex", context_limit=256000),
    "gpt-5-pro": ModelPricing(15.0, 120.0, "openai", "gpt-5-pro", context_limit=256000),
    "gpt-5-mini": ModelPricing(0.25, 2.0, "openai", "gpt-5-mini", context_limit=400000, cached_input_price=0.025),
    "gpt-5.4-mini": ModelPricing(0.75, 4.50, "openai", "gpt-5.4-mini", context_limit=400000, cached_input_price=0.075),
    "gpt-5-nano": ModelPricing(0.05, 0.40, "openai", "gpt-5-nano", context_limit=128000, cached_input_price=0.005),
    # Anthropic. Input/output rates are the published ones; cached_input_price is derived as
    # 0.1x input, the cache-read multiplier, matching every pre-existing row in this block.
    # The 4.6-and-later models take 1M context, unlike the 200K of the 4.0/4.5 generation.
    "claude-fable-5": ModelPricing(10.0, 50.0, "anthropic", "claude-fable-5", context_limit=1000000, cached_input_price=1.00),
    "claude-opus-5": ModelPricing(5.0, 25.0, "anthropic", "claude-opus-5", context_limit=1000000, cached_input_price=0.50),
    "claude-opus-4-8": ModelPricing(5.0, 25.0, "anthropic", "claude-opus-4-8", context_limit=1000000, cached_input_price=0.50),
    "claude-opus-4-7": ModelPricing(5.0, 25.0, "anthropic", "claude-opus-4-7", context_limit=1000000, cached_input_price=0.50),
    "claude-opus-4-6": ModelPricing(5.0, 25.0, "anthropic", "claude-opus-4-6", context_limit=1000000, cached_input_price=0.50),
    "claude-sonnet-5": ModelPricing(2.0, 10.0, "anthropic", "claude-sonnet-5", context_limit=1000000, cached_input_price=0.20),
    "claude-sonnet-4-6": ModelPricing(3.0, 15.0, "anthropic", "claude-sonnet-4-6", context_limit=1000000, cached_input_price=0.30),
    # Was 0.80/4.0 here, which under-billed by a fifth; the published rate is 1.00/5.00.
    "claude-haiku-4-5": ModelPricing(1.00, 5.00, "anthropic", "claude-haiku-4-5", context_limit=200000, cached_input_price=0.10),
    # UNVERIFIED: the 4.5 pair and the two dated 4.0 snapshots are absent from the current
    # pricing page, so these rates are carried over unchecked. The 4.0 snapshots are marked
    # deprecated upstream. Left in place for old runs whose logs name them.
    "claude-opus-4-5": ModelPricing(5.0, 25.0, "anthropic", "claude-opus-4-5", context_limit=200000, cached_input_price=0.50),
    "claude-sonnet-4-5": ModelPricing(3.0, 15.0, "anthropic", "claude-sonnet-4-5", context_limit=200000, cached_input_price=0.30),
    "claude-sonnet-4-20250514": ModelPricing(3.0, 15.0, "anthropic", "claude-sonnet-4", context_limit=200000, cached_input_price=0.30),
    "claude-opus-4-20250514": ModelPricing(15.0, 75.0, "anthropic", "claude-opus-4", context_limit=200000, cached_input_price=1.50),
    # Google — Gemini 2.5 Pro output price is $10/1M for prompts <=200K, $15/1M for >200K;
    # using <=200K tier as default since most pipeline calls are well under that limit
    "gemini-2.5-pro": ModelPricing(1.25, 10.0, "gemini", "gemini-2.5-pro", context_limit=1000000, cached_input_price=0.125),
    "gemini-2.5-flash": ModelPricing(0.30, 2.50, "gemini", "gemini-2.5-flash", context_limit=1000000, cached_input_price=0.03),
    # UNVERIFIED: Gemini 3 Flash Preview is no longer listed on Google's pricing page; this
    # rate is carried over unchecked. Gemini 3.5 Flash below is its listed successor.
    "gemini-3-flash-preview": ModelPricing(0.50, 3.00, "gemini", "gemini-3-flash-preview", context_limit=1000000),
    # Gemini 3.1 Pro Preview — pricing reflects the FLEX service tier (which the
    # provider uses by default; see gemini_provider.py `service_tier="flex"`).
    # Flex: $1.00/M input, $6.00/M output (incl. thinking) for prompts <=200k.
    # Standard tier would be ~2x these values; if you set service_tier: "standard"
    # in config.yaml, update the prices below or the tracker will undercount.
    "gemini-3.1-pro-preview": ModelPricing(1.0, 6.0, "gemini", "gemini-3.1-pro-preview", context_limit=200000, cached_input_price=0.10),
    # Gemini 3.6 / 3.7 Flash — FLEX tier, matching the provider default: $0.375/M input,
    # $1.875/M output (thinking tokens bill at the output rate), $0.0375/M cached input.
    # Standard tier is exactly 2x. These are promotional rates that Google lists as holding
    # through 31 December 2026 and doubling on 1 January 2027, so they need revisiting then.
    "gemini-3.6-flash": ModelPricing(0.375, 1.875, "gemini", "gemini-3.6-flash", context_limit=1048576, cached_input_price=0.0375),
    "gemini-3.7-flash": ModelPricing(0.375, 1.875, "gemini", "gemini-3.7-flash", context_limit=1048576, cached_input_price=0.0375),
    # Gemini 3.5 Flash — FLEX tier ($0.75/$4.50); standard is 2x, as with the other Flash models.
    "gemini-3.5-flash": ModelPricing(0.75, 4.50, "gemini", "gemini-3.5-flash", context_limit=1048576, cached_input_price=0.075),
    # MiniMax
    "MiniMax-M2.7": ModelPricing(0.30, 1.2, "minimax", "MiniMax-M2.7", context_limit=204800, cached_input_price=0.06),
    # GLM-5 (Z.AI) — $1/1M input, $0.20/1M cached input, $3.2/1M output.
    "glm-5": ModelPricing(1.0, 3.2, "openai", "glm-5", context_limit=200000, cached_input_price=0.20),
    # UNVERIFIED: DeepSeek's pricing page now lists only the V4 models, so this V3.2 rate is
    # carried over unchecked.
    "deepseek-chat": ModelPricing(0.28, 0.42, "deepseek", "deepseek-chat", context_limit=128000, cached_input_price=0.028),
    # DeepSeek V4 (model versions DeepSeek-V4-Flash-0731 and DeepSeek-V4-Pro-0813). DeepSeek bills
    # by time of day: peak is 01:00-04:00 and 06:00-10:00 UTC on weekdays, and off-peak is exactly
    # half. The peak rate is recorded here so the pre-call limit gate is never optimistic; a run
    # made entirely off-peak costs half of what the tracker reports.
    "deepseek-v4-flash": ModelPricing(0.44, 1.32, "deepseek", "deepseek-v4-flash", context_limit=1000000, cached_input_price=0.014),
    "deepseek-v4-pro": ModelPricing(1.32, 3.96, "deepseek", "deepseek-v4-pro", context_limit=1000000, cached_input_price=0.044),
}


class CostEstimate(BaseModel):
    """Estimated cost for an API call."""

    input_tokens: int
    estimated_output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    model: str


class CostLimitExceeded(Exception):
    """Raised when estimated cost of a single API call exceeds the per-call limit."""

    def __init__(self, estimate: CostEstimate, per_call_limit: float):
        self.estimate = estimate
        self.per_call_limit = per_call_limit
        super().__init__(
            f"Single API call estimated cost ${estimate.total_cost:.4f} "
            f"(input: {estimate.input_tokens} tokens) exceeds per-call limit ${per_call_limit:.2f}. "
            f"This prevents runaway token usage."
        )


class TokenLimitExceeded(Exception):
    """Raised when input tokens exceed the model's context window limit."""

    def __init__(self, estimate: CostEstimate, context_limit: int):
        self.estimate = estimate
        self.context_limit = context_limit
        super().__init__(
            f"Input tokens ({estimate.input_tokens:,}) exceed model context limit ({context_limit:,}). "
            f"The message is too large for the model to process. "
            f"Consider truncating the input or using a model with a larger context window."
        )


class SessionBudgetExceeded(Exception):
    """Raised when session total would exceed the session budget."""

    def __init__(self, estimate: CostEstimate, session_limit: float, session_total: float):
        self.estimate = estimate
        self.session_limit = session_limit
        self.session_total = session_total
        super().__init__(
            f"Session budget would be exceeded. Current total: ${session_total:.4f}, "
            f"estimated call: ${estimate.total_cost:.4f}, limit: ${session_limit:.2f}"
        )


class CostTracker:
    """Tracks and estimates API costs across a session."""

    def __init__(
        self,
        per_call_limit: float = 10.0,
        session_limit: float = 50.0,
        warn_threshold: float = 0.8,
    ):
        """Initialize cost tracker.

        Args:
            per_call_limit: Maximum allowed cost for a single API call in USD
                           (prevents runaway token usage like 1.9M token calls)
            session_limit: Maximum total cost for the entire session in USD
            warn_threshold: Fraction of session limit at which to warn (0.8 = 80%)
        """
        self.per_call_limit = per_call_limit
        self.session_limit = session_limit
        self.warn_threshold = warn_threshold
        self.session_cost = 0.0
        self.call_history: list[dict[str, Any]] = []
        self.logger = get_logger("cost_tracker")

    def set_zero_pricing(self, model: str) -> None:
        """Set pricing to $0 for a model (e.g. subscription plan).

        Tokens are still tracked but cost is reported as $0.
        """
        existing = MODEL_PRICING.get(model)
        MODEL_PRICING[model] = ModelPricing(
            0.0, 0.0,
            provider=existing.provider if existing else "subscription",
            model_id=model,
            context_limit=existing.context_limit if existing else 128000,
        )
        self.logger.info(f"Subscription mode: {model} pricing set to $0")

    def get_pricing(self, model: str) -> ModelPricing | None:
        """Get pricing for a model, returns None if unknown."""
        return MODEL_PRICING.get(model)

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text for a specific model.

        Uses tiktoken for OpenAI models, approximation for others.
        """
        # Try tiktoken for OpenAI models
        if model.startswith("gpt"):
            try:
                import tiktoken
                # Use cl100k_base as default encoding for GPT-4+ models
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(text))
            except ImportError:
                self.logger.warning("tiktoken not installed, using approximation")
            except Exception as e:
                self.logger.warning(f"tiktoken error: {e}, using approximation")

        # Fallback: approximate 4 characters per token
        return len(text) // 4

    def count_messages_tokens(self, messages: list, model: str) -> int:
        """Count total tokens in a list of messages."""
        total = 0
        for msg in messages:
            # Handle both Message objects and dicts
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content', '')
            else:
                content = str(msg)

            total += self.count_tokens(content, model)
            # Add overhead for role and formatting (approximately 4 tokens per message)
            total += 4

        return total

    def estimate_cost(
        self,
        messages: list,
        model: str,
        expected_output_tokens: int | None = None,
    ) -> CostEstimate:
        """Estimate the cost of an API call.

        Args:
            messages: Input messages
            model: Model identifier
            expected_output_tokens: Expected output tokens (defaults to 2000)

        Returns:
            CostEstimate with breakdown
        """
        pricing = self.get_pricing(model)
        if pricing is None:
            # Unknown model - use conservative estimate
            self.logger.warning(f"Unknown model pricing: {model}, using conservative estimate")
            pricing = ModelPricing(10.0, 40.0, "unknown", model)

        input_tokens = self.count_messages_tokens(messages, model)
        output_tokens = expected_output_tokens or 2000  # Conservative default

        input_cost = (input_tokens / 1_000_000) * pricing.input_price
        output_cost = (output_tokens / 1_000_000) * pricing.output_price

        return CostEstimate(
            input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            model=model,
        )

    def check_and_record(
        self,
        messages: list,
        model: str,
        expected_output_tokens: int | None = None,
    ) -> CostEstimate:
        """Check if call is within budget and record estimate.

        Raises:
            TokenLimitExceeded: If input tokens exceed model's context window
            CostLimitExceeded: If single call cost exceeds per-call limit
            SessionBudgetExceeded: If session total would exceed session limit

        Returns:
            CostEstimate if within budget
        """
        estimate = self.estimate_cost(messages, model, expected_output_tokens)
        pricing = self.get_pricing(model)

        # Check if input tokens exceed model's context window limit
        # This is checked FIRST because it's a hard limit from the API
        context_limit = pricing.context_limit if pricing else 128000  # Default fallback
        # Use 90% of context limit to leave room for output tokens
        safe_input_limit = int(context_limit * 0.9)

        if estimate.input_tokens > safe_input_limit:
            self.logger.error(
                "Input tokens exceed model context limit",
                {
                    "input_tokens": estimate.input_tokens,
                    "context_limit": context_limit,
                    "safe_input_limit": safe_input_limit,
                    "model": model,
                }
            )
            raise TokenLimitExceeded(estimate, context_limit)

        # Check if this single call exceeds per-call limit (prevents runaway token usage)
        if estimate.total_cost > self.per_call_limit:
            self.logger.error(
                "Single API call exceeds per-call limit",
                {
                    "estimated_cost": estimate.total_cost,
                    "per_call_limit": self.per_call_limit,
                    "input_tokens": estimate.input_tokens,
                }
            )
            raise CostLimitExceeded(estimate, self.per_call_limit)

        # Check if session total would exceed session limit
        if self.session_cost + estimate.total_cost > self.session_limit:
            self.logger.error(
                "Session budget would be exceeded",
                {
                    "estimated_cost": estimate.total_cost,
                    "session_total": self.session_cost,
                    "session_limit": self.session_limit,
                    "input_tokens": estimate.input_tokens,
                }
            )
            raise SessionBudgetExceeded(estimate, self.session_limit, self.session_cost)

        self.logger.debug(
            "Cost check passed",
            {
                "estimated_cost": estimate.total_cost,
                "input_tokens": estimate.input_tokens,
                "context_limit": context_limit,
                "remaining_budget": self.session_limit - self.session_cost - estimate.total_cost,
            }
        )

        return estimate

    def record_actual_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        """Record actual token usage after API call.

        Returns:
            Actual cost of the call
        """
        pricing = self.get_pricing(model) or ModelPricing(10.0, 40.0, "unknown", model)

        # Split input cost: cached tokens at cached price, rest at full price
        uncached_input_tokens = input_tokens - cached_input_tokens
        cached_price = pricing.cached_input_price if pricing.cached_input_price > 0 else pricing.input_price
        input_cost = (uncached_input_tokens / 1_000_000) * pricing.input_price + \
                     (cached_input_tokens / 1_000_000) * cached_price
        output_cost = (output_tokens / 1_000_000) * pricing.output_price
        actual_cost = input_cost + output_cost

        self.session_cost += actual_cost
        self.call_history.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "cost": actual_cost,
        })

        self.logger.info(
            "Recorded API usage",
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_input_tokens": cached_input_tokens,
                "cost": actual_cost,
                "session_total": self.session_cost,
            }
        )

        return actual_cost

    def should_warn(self) -> bool:
        """Check if session cost is approaching limit."""
        return self.session_cost >= (self.session_limit * self.warn_threshold)

    def get_summary(self) -> dict[str, Any]:
        """Get cost tracking summary."""
        return {
            "session_cost": self.session_cost,
            "per_call_limit": self.per_call_limit,
            "session_limit": self.session_limit,
            "remaining_budget": self.session_limit - self.session_cost,
            "call_count": len(self.call_history),
            "approaching_limit": self.should_warn(),
        }

    def get_detailed_summary(self) -> dict[str, Any]:
        """Get detailed cost breakdown by model."""
        by_model: dict[str, dict[str, Any]] = {}

        for call in self.call_history:
            model = call["model"]
            if model not in by_model:
                by_model[model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                }
            by_model[model]["calls"] += 1
            by_model[model]["input_tokens"] += call["input_tokens"]
            by_model[model]["output_tokens"] += call["output_tokens"]
            by_model[model]["cost"] += call["cost"]

        return {
            **self.get_summary(),
            "by_model": by_model,
        }
