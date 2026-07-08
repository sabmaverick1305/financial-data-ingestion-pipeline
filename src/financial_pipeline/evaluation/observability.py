"""Observability — per-request tracing + CloudWatch metrics (Dimension 7).

Two layers:
  1. RequestTrace  — captures every metric for one API request
  2. MetricsEmitter — flushes aggregated CloudWatch custom metrics

RequestTrace is attached to the AugmentedResponse as metadata so the API
can log it to structured JSON. The emitter runs as a background flush every
60 seconds or on demand.

CloudWatch namespace: AMFI/Pipeline
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Generator

import structlog

from financial_pipeline.config import settings
from financial_pipeline.evaluation.cost_tracker import QueryCost, cost_for

log = structlog.get_logger()


# ── Per-request trace ─────────────────────────────────────────────────────────

@dataclass
class StageLatency:
    stage:      str
    latency_ms: int


@dataclass
class RequestTrace:
    """Full observability record for one /api/ask or /api/search call."""

    request_id:    str
    timestamp:     str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    question:      str = ""
    intent:        str = ""

    # Stage latencies
    stages:        list[StageLatency] = field(default_factory=list)
    total_ms:      int = 0

    # Retrieval
    dense_count:   int = 0
    sparse_count:  int = 0
    fused_count:   int = 0
    reranked_count:int = 0
    final_count:   int = 0
    both_signals:  int = 0    # chunks matched by BOTH dense AND sparse

    # Generation
    model:             str   = ""
    provider:          str   = ""
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    cost_usd:          float = 0.0

    # Guardrails
    pre_passed:          bool  = True
    pre_blocked:         bool  = False
    post_passed:         bool  = True
    answer_safe:         bool  = True
    abstention_detected: bool  = False
    faithfulness_score:  float = -1.0
    hallucination_risk:  str   = "unknown"

    # Quality signals
    citation_count:      int  = 0
    context_quality:     float = 0.0

    def add_stage(self, stage: str, latency_ms: int) -> None:
        self.stages.append(StageLatency(stage=stage, latency_ms=latency_ms))

    def to_log_dict(self) -> dict:
        return {
            "request_id":        self.request_id,
            "ts":                self.timestamp,
            "question_len":      len(self.question),
            "intent":            self.intent,
            "total_ms":          self.total_ms,
            "stage_ms":          {s.stage: s.latency_ms for s in self.stages},
            "dense":             self.dense_count,
            "sparse":            self.sparse_count,
            "fused":             self.fused_count,
            "reranked":          self.reranked_count,
            "final":             self.final_count,
            "both_signals":      self.both_signals,
            "model":             self.model,
            "provider":          self.provider,
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd":          round(self.cost_usd, 6),
            "pre_blocked":       self.pre_blocked,
            "post_passed":       self.post_passed,
            "answer_safe":       self.answer_safe,
            "abstained":         self.abstention_detected,
            "faithfulness":      self.faithfulness_score,
            "risk":              self.hallucination_risk,
            "citations":         self.citation_count,
            "context_quality":   self.context_quality,
        }

    @classmethod
    def from_augmented_response(
        cls,
        resp,               # AugmentedResponse
        request_id: str,
        question:   str,
    ) -> "RequestTrace":
        trace = cls(request_id=request_id, question=question[:200])

        if resp.generation:
            trace.model             = resp.generation.model
            trace.provider          = resp.generation.provider
            trace.prompt_tokens     = resp.generation.prompt_tokens
            trace.completion_tokens = resp.generation.completion_tokens
            trace.total_ms          = resp.generation.latency_ms
            trace.cost_usd          = cost_for(
                resp.generation.model,
                resp.generation.prompt_tokens,
                resp.generation.completion_tokens,
            )

        g = resp.guardrail
        trace.pre_passed          = g.pre_passed
        trace.pre_blocked         = g.blocked
        trace.post_passed         = g.post_passed
        trace.answer_safe         = g.answer_safe
        trace.abstention_detected = g.abstention_detected
        trace.faithfulness_score  = g.faithfulness_score
        trace.hallucination_risk  = g.hallucination_risk
        trace.intent              = resp.intent_type
        trace.citation_count      = len(resp.citations)

        stats = resp.augment_stats
        trace.total_ms       = stats.get("total_latency_ms", trace.total_ms)
        trace.context_quality= stats.get("context_quality", 0.0)

        return trace


# ── Latency context manager ───────────────────────────────────────────────────

@contextlib.contextmanager
def measure(trace: RequestTrace, stage: str) -> Generator[None, None, None]:
    """Context manager that records stage latency into the trace."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ms = int((time.perf_counter() - t0) * 1000)
        trace.add_stage(stage, ms)


# ── CloudWatch metrics emitter ────────────────────────────────────────────────

class MetricsEmitter:
    """Emits custom CloudWatch metrics from RequestTrace objects.

    Batches up to 20 metric data points per PutMetricData call (CW limit).
    Call flush() periodically or at process exit.

    Metrics emitted (namespace: AMFI/Pipeline):
      retrieval.latency_ms         — histogram
      generation.latency_ms        — histogram
      pipeline.total_latency_ms    — histogram
      pipeline.cost_usd            — sum
      pipeline.prompt_tokens       — sum
      pipeline.completion_tokens   — sum
      guardrail.blocked            — count (0/1 per request)
      guardrail.post_failed        — count
      guardrail.abstained          — count
      retrieval.hit_both_signals   — count (chunks matched both dense+sparse)
      retrieval.faithfulness       — gauge
    """

    NAMESPACE = "AMFI/Pipeline"
    BATCH_SIZE = 20

    def __init__(self) -> None:
        self._buffer: list[dict] = []
        self._client = None

    def record(self, trace: RequestTrace) -> None:
        """Add a request trace to the metrics buffer."""
        ts = datetime.now(UTC)

        metrics = [
            ("pipeline.total_latency_ms",  trace.total_ms,          "Milliseconds"),
            ("pipeline.cost_usd",          trace.cost_usd * 1000,    "Microseconds"),  # CW has no $ unit; use microseconds as a proxy
            ("pipeline.prompt_tokens",     trace.prompt_tokens,       "Count"),
            ("pipeline.completion_tokens", trace.completion_tokens,   "Count"),
            ("guardrail.blocked",          int(trace.pre_blocked),    "Count"),
            ("guardrail.post_failed",      int(not trace.post_passed),"Count"),
            ("guardrail.abstained",        int(trace.abstention_detected), "Count"),
            ("retrieval.both_signals",     trace.both_signals,        "Count"),
            ("retrieval.final_count",      trace.final_count,         "Count"),
        ]
        if trace.faithfulness_score >= 0:
            metrics.append(("retrieval.faithfulness", trace.faithfulness_score, "None"))

        for stage in trace.stages:
            metrics.append((f"stage.{stage.stage}_ms", stage.latency_ms, "Milliseconds"))

        for name, value, unit in metrics:
            self._buffer.append({
                "MetricName": name,
                "Value":      float(value),
                "Unit":       unit,
                "Timestamp":  ts,
                "Dimensions": [
                    {"Name": "intent",   "Value": trace.intent or "unknown"},
                    {"Name": "provider", "Value": trace.provider or "unknown"},
                ],
            })

        if len(self._buffer) >= self.BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        """Send buffered metrics to CloudWatch."""
        if not self._buffer:
            return
        client = self._get_client()
        if client is None:
            self._buffer.clear()
            return
        try:
            # Split into batches of 20 (CW limit per call)
            for i in range(0, len(self._buffer), 20):
                batch = self._buffer[i : i + 20]
                client.put_metric_data(Namespace=self.NAMESPACE, MetricData=batch)
            log.debug("metrics.flushed", count=len(self._buffer))
        except Exception as exc:
            log.warning("metrics.flush_failed", error=str(exc))
        finally:
            self._buffer.clear()

    def _get_client(self):
        if not settings.aws_access_key_id:
            return None
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "cloudwatch",
                    region_name=settings.aws_region,
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                )
            except Exception as exc:
                log.warning("metrics.client_init_failed", error=str(exc))
                return None
        return self._client


# ── LangSmith per-run feedback emitter ───────────────────────────────────────

# Metrics emitted as LangSmith feedback keys (all scores in [0, 1]):
_LS_FEEDBACK_KEYS: list[tuple[str, str]] = [
    ("faithfulness_embed",   "Answer embedding cosine similarity vs cited chunks"),
    ("citation_coverage",    "Fraction of answer sentences with a [N] citation"),
    ("citation_validity",    "Fraction of [N] markers pointing to a valid source"),
    ("keyword_recall",       "Fraction of expected keywords found in answer/sources"),
    ("numeric_precision",    "Fraction of numbers in answer grounded in cited sources"),
    ("expected_num_hit",     "Fraction of expected numbers found in answer/sources"),
    ("abstention_correct",   "Abstention behaviour matched expectation (bool → 0/1)"),
    ("pre_blocked",          "Pre-guardrail blocked the query (bool → 0/1)"),
    ("post_passed",          "Post-guardrail passed (bool → 0/1)"),
]


class LangSmithFeedbackEmitter:
    """Pushes per-case eval metric scores as LangSmith run feedback.

    Each eval case produces one LangGraph run (identified by a UUID passed
    via RunnableConfig). After the case completes, call record_case() to
    attach all metric scores to that run in LangSmith.

    No-ops silently when LANGSMITH_API_KEY is not set or the SDK is absent.
    """

    def __init__(self) -> None:
        self._client = None
        self._enabled: bool | None = None   # None = uninitialised

    def _get_client(self):
        if self._enabled is False:
            return None
        if self._client is not None:
            return self._client
        try:
            import os
            from langsmith import Client
            if not os.environ.get("LANGSMITH_API_KEY"):
                self._enabled = False
                return None
            self._client = Client()
            self._enabled = True
            log.info("langsmith.feedback_emitter_ready")
            return self._client
        except Exception as exc:
            log.warning("langsmith.client_init_failed", error=str(exc))
            self._enabled = False
            return None

    def record_case(self, run_id: str | None, result: dict) -> None:
        """Attach per-case metric scores to the LangSmith run identified by run_id."""
        client = self._get_client()
        if client is None or not run_id:
            return

        for key, comment in _LS_FEEDBACK_KEYS:
            val = result.get(key)
            if val is None:
                continue
            # Booleans → 1.0/0.0; floats kept as-is; skip unavailable (-1)
            if isinstance(val, bool):
                score = 1.0 if val else 0.0
            else:
                score = float(val)
                if score < 0:
                    continue

            try:
                client.create_feedback(
                    run_id=run_id,
                    key=key,
                    score=score,
                    comment=comment,
                )
            except Exception as exc:
                log.warning("langsmith.feedback_failed", key=key, error=str(exc))

    def record_aggregate(self, run_id: str | None, metrics) -> None:
        """Attach aggregated EvalMetrics scores to a summary run in LangSmith.

        metrics is an EvalMetrics dataclass instance.
        """
        client = self._get_client()
        if client is None or not run_id:
            return

        aggregate_keys = [
            ("hit_at_5",               "Retrieval Hit@5"),
            ("hit_at_10",              "Retrieval Hit@10"),
            ("mrr",                    "Mean Reciprocal Rank"),
            ("ndcg_at_10",             "NDCG@10"),
            ("ce_avg_improvement",     "Cross-encoder avg rank improvement"),
            ("ce_promotion_rate",      "Cross-encoder promotion rate"),
            ("citation_coverage_avg",  "Avg citation coverage across cases"),
            ("citation_validity_avg",  "Avg citation validity across cases"),
            ("faithfulness_avg",       "Avg faithfulness (embedding cosine sim)"),
            ("keyword_recall_avg",     "Avg keyword recall"),
            ("abstention_acc",         "Abstention accuracy"),
            ("numeric_precision_avg",  "Avg numeric precision"),
            ("expected_number_hit_avg","Avg expected number hit rate"),
            ("advice_block_rate",      "Investment-advice block rate (goal: 1.0)"),
            ("false_positive_rate",    "False-positive block rate (goal: 0.0)"),
            ("safety_catch_rate",      "Safety catch rate"),
        ]
        for attr, comment in aggregate_keys:
            val = getattr(metrics, attr, None)
            if val is None:
                continue
            score = float(val)
            if score < 0:
                continue
            try:
                client.create_feedback(
                    run_id=run_id,
                    key=f"aggregate.{attr}",
                    score=score,
                    comment=comment,
                )
            except Exception as exc:
                log.warning("langsmith.aggregate_feedback_failed", key=attr, error=str(exc))


# ── Module-level singleton ────────────────────────────────────────────────────
# Import and use this in api/main.py:
#   from financial_pipeline.evaluation.observability import emitter
#   emitter.record(trace)

emitter = MetricsEmitter()
ls_feedback = LangSmithFeedbackEmitter()
