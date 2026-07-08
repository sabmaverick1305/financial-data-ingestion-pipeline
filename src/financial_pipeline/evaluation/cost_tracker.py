"""Cost Tracker — Dimension 7 (Latency & Cost).

Tracks token usage and calculates per-query cost for every LLM provider
and model we support. All pricing in USD per 1K tokens (as of 2026-07).

Update PRICING when model prices change — it's the only place to edit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── Model pricing (USD per 1 000 tokens) ──────────────────────────────────────
# Source: official pricing pages, updated 2026-07
PRICING: dict[str, dict[str, float]] = {
    # Anthropic Claude (direct API)
    "claude-haiku-4-5-20251001":   {"input": 0.00025, "output": 0.00125},
    "claude-sonnet-4-5-20250929":  {"input": 0.003,   "output": 0.015},
    "claude-sonnet-4-6":           {"input": 0.003,   "output": 0.015},
    "claude-opus-4-5-20251101":    {"input": 0.015,   "output": 0.075},
    # Bedrock Claude (same model, ~same price; AWS may add small markup)
    "anthropic.claude-haiku-4-5-20251001-v1:0":  {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-3-5-haiku-20241022-v1:0":  {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-3-haiku-20240307-v1:0":    {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 0.003,   "output": 0.015},
    # OpenAI
    "gpt-4o-mini":                 {"input": 0.00015, "output": 0.0006},
    "gpt-4o":                      {"input": 0.005,   "output": 0.015},
    "gpt-4.1":                     {"input": 0.002,   "output": 0.008},
    # Embedding (per 1K tokens, output = 0)
    "all-MiniLM-L6-v2":            {"input": 0.0,     "output": 0.0},   # self-hosted, free
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {"input": 0.0, "output": 0.0},
}


def cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for one LLM call."""
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1000.0


@dataclass
class QueryCost:
    model:             str
    prompt_tokens:     int
    completion_tokens: int
    latency_ms:        int
    cost_usd:          float = 0.0

    def __post_init__(self) -> None:
        self.cost_usd = cost_for(self.model, self.prompt_tokens, self.completion_tokens)


@dataclass
class CostSummary:
    queries:               int   = 0
    total_prompt_tokens:   int   = 0
    total_completion_tokens: int = 0
    total_cost_usd:        float = 0.0
    latencies_ms:          list[int] = field(default_factory=list)
    by_model:              dict[str, dict] = field(default_factory=dict)

    def add(self, q: QueryCost) -> None:
        self.queries               += 1
        self.total_prompt_tokens   += q.prompt_tokens
        self.total_completion_tokens += q.completion_tokens
        self.total_cost_usd        += q.cost_usd
        self.latencies_ms.append(q.latency_ms)

        m = self.by_model.setdefault(q.model, {
            "queries": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "cost_usd": 0.0,
        })
        m["queries"]           += 1
        m["prompt_tokens"]     += q.prompt_tokens
        m["completion_tokens"] += q.completion_tokens
        m["cost_usd"]          += q.cost_usd

    @property
    def avg_prompt_tokens(self) -> float:
        return self.total_prompt_tokens / max(self.queries, 1)

    @property
    def avg_completion_tokens(self) -> float:
        return self.total_completion_tokens / max(self.queries, 1)

    @property
    def cost_per_query(self) -> float:
        return self.total_cost_usd / max(self.queries, 1)

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / max(len(self.latencies_ms), 1)

    @property
    def p90_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_ms = sorted(self.latencies_ms)
        idx = max(0, int(len(sorted_ms) * 0.9) - 1)
        return float(sorted_ms[idx])

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_ms = sorted(self.latencies_ms)
        idx = max(0, int(len(sorted_ms) * 0.99) - 1)
        return float(sorted_ms[idx])

    def print_report(self) -> None:
        print(f"\n  Dim 7 — Latency & Cost")
        print(f"    Queries        : {self.queries}")
        print(f"    Avg latency    : {self.avg_latency_ms:.0f} ms")
        print(f"    p90 latency    : {self.p90_latency_ms:.0f} ms")
        print(f"    p99 latency    : {self.p99_latency_ms:.0f} ms")
        print(f"    Avg tokens     : {self.avg_prompt_tokens:.0f}in / {self.avg_completion_tokens:.0f}out")
        print(f"    Total cost     : ${self.total_cost_usd:.4f}")
        print(f"    Cost / query   : ${self.cost_per_query:.4f}")
        if self.by_model:
            print(f"    By model:")
            for model, stats in self.by_model.items():
                print(f"      {model}: {stats['queries']} calls  ${stats['cost_usd']:.4f}")
