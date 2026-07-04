"""RAG LangGraph — wires all nodes and edges into a compiled StateGraph.

Usage (from api/main.py):
    from financial_pipeline.graph.graph import build_graph
    from financial_pipeline.graph.nodes import NodeFactory

    factory = NodeFactory(repo=repo, retriever=retriever)
    graph   = build_graph(factory)

    result  = graph.invoke({
        "query":        "What is the total number of folios in large cap funds?",
        "retry_count":  0,
        "repair_count": 0,
    })
    response = result["response"]   # serialisable AskResponse dict
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from financial_pipeline.graph.edges import (
    after_grade,
    after_metadata_first,
    after_post_guardrail,
    after_pre_guardrail,
    after_route,
)
from financial_pipeline.graph.nodes import NodeFactory
from financial_pipeline.graph.state import RAGState


def build_graph(factory: NodeFactory):
    """Build and compile the RAG graph with injected dependencies.

    Parameters
    ----------
    factory:
        NodeFactory instance created once at app startup.
        Holds repo, retriever, ranker, generator and all stateless helpers.

    Returns
    -------
    CompiledGraph
        Ready to call via graph.invoke({...}) or graph.stream({...}).
    """
    g = StateGraph(RAGState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    g.add_node("analyze_query",     factory.analyze_query)
    g.add_node("route",             factory.route)
    g.add_node("retrieve_dense",    factory.retrieve_dense)
    g.add_node("retrieve_sparse",   factory.retrieve_sparse)
    g.add_node("retrieve_table",    factory.retrieve_table)
    g.add_node("retrieve_metadata",       factory.retrieve_metadata)
    g.add_node("retrieve_metadata_first", factory.retrieve_metadata_first)
    g.add_node("rrf_fusion",              factory.rrf_fusion)
    g.add_node("rerank",            factory.rerank)
    g.add_node("context_optimizer", factory.context_optimizer)
    g.add_node("grade_context",     factory.grade_context)
    g.add_node("rewrite_query",     factory.rewrite_query)
    g.add_node("augment",           factory.augment)
    g.add_node("pre_guardrail",     factory.pre_guardrail)
    g.add_node("generate",          factory.generate)
    g.add_node("post_guardrail",    factory.post_guardrail)
    g.add_node("repair",            factory.repair)
    g.add_node("format_response",   factory.format_response)

    # ── Entry point ────────────────────────────────────────────────────────
    g.set_entry_point("analyze_query")

    # ── Linear edges ───────────────────────────────────────────────────────
    g.add_edge("analyze_query",     "route")
    g.add_edge("rrf_fusion",        "rerank")
    g.add_edge("rerank",            "context_optimizer")
    g.add_edge("context_optimizer", "grade_context")
    g.add_edge("rewrite_query",     "route")        # CRAG retry loop back
    g.add_edge("augment",           "pre_guardrail")
    g.add_edge("generate",          "post_guardrail")
    g.add_edge("repair",            "post_guardrail")   # bounded repair loop
    g.add_edge("format_response",   END)

    # ── Routing from route node ────────────────────────────────────────────
    # lookup intent  → retrieve_metadata_first (sequential: find doc first)
    # all others     → parallel fan-out via Send
    g.add_conditional_edges("route", after_route)

    # Fan-in: parallel retrieval branches converge at rrf_fusion
    for branch in ("dense", "sparse", "table", "metadata"):
        g.add_edge(f"retrieve_{branch}", "rrf_fusion")

    # ── Sequential lookup path ─────────────────────────────────────────────
    # retrieve_metadata_first → targeted fan-out (doc IDs found)
    #                         → fallback fan-out  (doc IDs empty)
    # Both paths converge at rrf_fusion via the branch edges above.
    g.add_conditional_edges("retrieve_metadata_first", after_metadata_first)

    # ── Conditional edges ──────────────────────────────────────────────────
    g.add_conditional_edges(
        "grade_context",
        after_grade,
        {
            "augment":         "augment",
            "rewrite_query":   "rewrite_query",
            "format_response": "format_response",
        },
    )

    g.add_conditional_edges(
        "pre_guardrail",
        after_pre_guardrail,
        {
            "generate":        "generate",
            "format_response": "format_response",
        },
    )

    g.add_conditional_edges(
        "post_guardrail",
        after_post_guardrail,
        {
            "repair":          "repair",
            "format_response": "format_response",
        },
    )

    return g.compile()
