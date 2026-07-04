"""Conditional edge functions for the RAG LangGraph.

Each function receives the full RAGState and returns a string key
that LangGraph uses to select the next node.

Edge map
--------
after_grade            : grade_context → augment | rewrite_query | format_response
after_pre_guardrail    : pre_guardrail → generate | format_response
after_post_guardrail   : post_guardrail → repair  | format_response
after_route            : route → retrieve_metadata_first (lookup) | parallel fan-out
after_metadata_first   : retrieve_metadata_first → targeted fan-out | fallback fan-out
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


# ── Edge 4a: internal fan-out helper ─────────────────────────────────────────

def _fan_out(state: RAGState) -> list[Send]:
    """Internal helper: parallel fan-out to all active retrieval branches.

    Each branch node writes only to its own result key so there are no
    write conflicts. LangGraph executes all Send targets concurrently.
    """
    branches = state.get("active_branches", ["dense", "sparse"])
    return [Send(f"retrieve_{branch}", state) for branch in branches]


# ── Edge 4b: after_route (replaces bare fan_out on the route node) ───────────

def after_route(state: RAGState):
    """Route after the route node.

    lookup intent → retrieve_metadata_first (sequential: find doc first)
    all others    → parallel fan-out via _fan_out
    """
    intent = state.get("intent")
    if intent and intent.intent_type == "lookup":
        return "retrieve_metadata_first"
    return _fan_out(state)


# ── Edge 4c: after_metadata_first ────────────────────────────────────────────

def after_metadata_first(state: RAGState) -> list[Send]:
    """Route after retrieve_metadata_first (Phase 1 of lookup sequential path).

    found_document_ids set  → targeted fan-out scoped to those document IDs
    found_document_ids empty → fallback to normal parallel fan-out
    """
    doc_ids  = state.get("found_document_ids", [])
    # Never re-invoke metadata — it already ran in Phase 1
    branches = [b for b in state.get("active_branches", ["dense", "sparse"])
                if b != "metadata"]

    if doc_ids:
        # Phase 2: search inside the found documents only
        return [Send(f"retrieve_{b}", state) for b in branches]

    # Fallback: document not found in metadata → normal parallel search
    return _fan_out(state)
