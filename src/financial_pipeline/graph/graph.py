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
from financial_pipeline.graph.edges_analytical import after_extract, is_range_query
from financial_pipeline.graph.nodes import NodeFactory
from financial_pipeline.graph.nodes_analytical import AnalyticalNodeFactory
from financial_pipeline.graph.state import RAGState


def build_graph(
    factory: NodeFactory,
    analytical: AnalyticalNodeFactory | None = None,
):
    """Build and compile the RAG graph with injected dependencies.

    Parameters
    ----------
    factory:
        NodeFactory instance created once at app startup.
        Holds repo, retriever, ranker, generator and all stateless helpers.
    analytical:
        AnalyticalNodeFactory for year-range aggregation queries.
        If None, a default instance is created using factory's repo.

    Returns
    -------
    CompiledGraph
        Ready to call via graph.invoke({...}) or graph.stream({...}).
    """
    if analytical is None:
        analytical = AnalyticalNodeFactory(repo=factory._repo)
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

    # ── Analytical agent nodes ──────────────────────────────────────────────
    g.add_node("plan_years",     analytical.plan_years)
    g.add_node("retrieve_year",  analytical.retrieve_year)
    g.add_node("extract_metric", analytical.extract_metric)
    g.add_node("synthesize",     analytical.synthesize)

    # ── Entry point ────────────────────────────────────────────────────────
    g.set_entry_point("analyze_query")

    # ── Linear edges ───────────────────────────────────────────────────────
    # analyze_query now branches: range query → analytical, else → parallel
    # (replaced the direct edge to route)
    g.add_edge("rrf_fusion",        "rerank")
    g.add_edge("rerank",            "context_optimizer")
    g.add_edge("context_optimizer", "grade_context")
    g.add_edge("rewrite_query",     "route")        # CRAG retry loop back
    g.add_edge("augment",           "pre_guardrail")
    g.add_edge("generate",          "post_guardrail")
    g.add_edge("repair",            "post_guardrail")   # bounded repair loop
    g.add_edge("format_response",   END)

    # ── Analytical agent edges ─────────────────────────────────────────────
    # Entry branch: range query → plan_years, all others → route
    g.add_conditional_edges(
        "analyze_query",
        is_range_query,
        {"plan_years": "plan_years", "route": "route"},
    )
    g.add_edge("plan_years",    "retrieve_year")
    g.add_edge("retrieve_year", "extract_metric")
    g.add_conditional_edges(
        "extract_metric",
        after_extract,
        {"retrieve_year": "retrieve_year", "synthesize": "synthesize"},
    )
    g.add_edge("synthesize", "post_guardrail")   # reuses existing guardrail path

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
