"""Conditional edge functions for the RAG LangGraph.

Each function receives the full RAGState and returns a string key
that LangGraph uses to select the next node.

Edge map
--------
after_grade          : grade_context → augment | rewrite_query | format_response
after_pre_guardrail  : pre_guardrail → generate | format_response
after_post_guardrail : post_guardrail → repair  | format_response
fan_out              : route → [Send × active_branches]  (parallel fan-out)
"""
from __future__ import annotations

from langgraph.types import Send

from financial_pipeline.graph.state import RAGState

# ── Loop guards ───────────────────────────────────────────────────────────────

MAX_RETRY  = 2   # CRAG retries before abstaining  (0 → 1 → 2 → abstain)
MAX_REPAIR = 1   # repair attempts before abstaining (0 → 1 → abstain)


# ── Edge 1: after grade_context ───────────────────────────────────────────────

def after_grade(state: RAGState) -> str:
    """Route after grade_context.

    sufficient          → augment
    retry + count < 2   → rewrite_query   (CRAG loop)
    retry + count >= 2  → format_response (abstain — retrieval exhausted)
    abstain             → format_response
    """
    grade       = state.get("grade", "abstain")
    retry_count = state.get("retry_count", 0)

    if grade == "sufficient":
        return "augment"

    if grade == "retry" and retry_count < MAX_RETRY:
        return "rewrite_query"

    # grade == "abstain", or retry limit reached
    return "format_response"


# ── Edge 2: after pre_guardrail ───────────────────────────────────────────────

def after_pre_guardrail(state: RAGState) -> str:
    """Route after pre_guardrail.

    blocked=True  → format_response  (investment advice / policy block)
    blocked=False → generate
    """
    return "format_response" if state.get("blocked") else "generate"


# ── Edge 3: after post_guardrail ──────────────────────────────────────────────

def after_post_guardrail(state: RAGState) -> str:
    """Route after post_guardrail.

    pass                                     → format_response
    answer_safe=False OR risk=high           → format_response  (abstain, not repairable)
    citation/number issues + repair_count < 1 → repair
    repair exhausted OR repair failed         → format_response  (abstain)
    """
    post         = state.get("post_result")
    repair_count = state.get("repair_count", 0)

    # No result object means guardrail wasn't reached — pass through
    if not post or post.passed:
        return "format_response"

    # Safety and high hallucination: another LLM call won't fix these
    if not post.answer_safe or post.hallucination_risk == "high":
        return "format_response"

    # Deterministic repair is available for citation and numeric issues
    repairable = (not post.citation_valid) or (not post.number_consistent)
    if repairable and repair_count < MAX_REPAIR:
        return "repair"

    # Repair exhausted or no repairable issue found
    return "format_response"


# ── Edge 4: fan_out (route → parallel retrieval branches) ─────────────────────

def fan_out(state: RAGState) -> list[Send]:
    """Fan-out from route to one or more retrieval branch nodes in parallel.

    LangGraph executes all Send targets concurrently.  Each branch node
    writes only to its own result key (dense_results, sparse_results, …)
    so there are no write conflicts.

    Only branches listed in state["active_branches"] are activated.
    """
    branches = state.get("active_branches", ["dense", "sparse"])
    return [Send(f"retrieve_{branch}", state) for branch in branches]
