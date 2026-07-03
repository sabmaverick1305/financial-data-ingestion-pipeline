"""LangGraph-based RAG orchestration layer.

Replaces the sequential RetrievalPipeline + AugmentationPipeline
with an explicit stateful graph:

  analyze_query → route → [retrieval branches] → rrf_fusion
  → rerank → context_optimizer → grade_context
  → augment → pre_guardrail → generate → post_guardrail
  → format_response

Entry point:
    from financial_pipeline.graph import graph
    result = graph.invoke({"query": "...", "retry_count": 0, "repair_count": 0})
"""

from .graph import build_graph
from .nodes import NodeFactory

__all__ = ["build_graph", "NodeFactory"]
