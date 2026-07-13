"""RAG LangGraph — wires all nodes and edges into a compiled StateGraph.

Usage (from api/main.py):
    from financial_pipeline.graph.graph import build_graph
    from financial_pipeline.graph.nodes import NodeFactory

    factory = NodeFactory(repo=repo, retriever=retriever)
    graph   = build_graph(factory, checkpointer=checkpointer)

    result  = graph.invoke(
        {"query": "What is the total number of folios in large cap funds?",
         "retry_count": 0, "repair_count": 0},
        config={"configurable": {"thread_id": thread_id}},
    )
    response = result["response"]   # serialisable AskResponse dict
"""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from financial_pipeline.graph.edges import (
    after_grade,
    after_metadata_first,
    after_post_guardrail,
    after_pre_guardrail,
    after_route,
)
from financial_pipeline.graph.edges_analytical import (
    after_aggregate_year,
    after_extract,
    after_extract_month,
    after_plan_years,
    is_range_query,
)
from financial_pipeline.graph.nodes import NodeFactory
from financial_pipeline.graph.nodes_analytical import AnalyticalNodeFactory
from financial_pipeline.graph.nodes_fund_performance import FundPerformanceNodeFactory
from financial_pipeline.graph.nodes_reasoning import ReasoningNodeFactory
from financial_pipeline.graph.nodes_sql import SQLNodeFactory
from financial_pipeline.graph.state import RAGState


def build_graph(
    factory: NodeFactory,
    analytical: AnalyticalNodeFactory | None = None,
    sql: SQLNodeFactory | None = None,
    fund_performance: FundPerformanceNodeFactory | None = None,
    reasoning: ReasoningNodeFactory | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
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
    sql:
        SQLNodeFactory wrapping the Vanna text-to-SQL agent.
        If None, the query_sql node is skipped (tabular queries fall through to route).
    fund_performance:
        FundPerformanceNodeFactory answering per-scheme NAV/return/CAGR
        queries against mf_scheme_master/mf_nav_history/mf_scheme_performance
        (a distinct dataset from AMFI's aggregate fund_category/AMC tables
        that query_sql targets). If None, a default instance is created from
        sql's Vanna agent (same agent, already trained on both domains — see
        scripts/train_vanna.py). If sql is also None, this node is skipped
        (fund-performance queries fall through to route, same as query_sql).
    reasoning:
        ReasoningNodeFactory answering causal "why did X change" queries by
        matching real computed metric directions against
        domain/semantic/reasoning_rules.yaml. If None, a default instance is
        created using factory's repo (same DB, no extra dependency to wire).
    checkpointer:
        A LangGraph BaseCheckpointSaver (e.g. storage/checkpointer.py's
        Postgres-backed PostgresSaver). When set, every node's state is
        persisted per thread_id (pass config={"configurable": {"thread_id": ...}}
        to .invoke()/.stream()), enabling replay and crash resumability.
        If None (default), the graph compiles with no persistence at all —
        exactly the pre-checkpointer behavior.

    Returns
    -------
    CompiledGraph
        Ready to call via graph.invoke({...}) or graph.stream({...}).
    """
    if analytical is None:
        analytical = AnalyticalNodeFactory(repo=factory._repo)
    if reasoning is None:
        reasoning = ReasoningNodeFactory(repo=factory._repo)
    if fund_performance is None and sql is not None:
        fund_performance = FundPerformanceNodeFactory(sql._vanna)
    g = StateGraph(RAGState)
    _has_sql = sql is not None
    _has_fund_performance = fund_performance is not None

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
    g.add_node("plan_years",           analytical.plan_years)
    g.add_node("query_db_stats",       analytical.query_db_stats)
    g.add_node("plan_months",          analytical.plan_months)
    g.add_node("retrieve_month",       analytical.retrieve_month)
    g.add_node("extract_month_metric", analytical.extract_month_metric)
    g.add_node("aggregate_year",       analytical.aggregate_year)
    g.add_node("synthesize",           analytical.synthesize)
    # Legacy single-snapshot nodes (kept so graph compiles; not in active path)
    g.add_node("retrieve_year",        analytical.retrieve_year)
    g.add_node("extract_metric",       analytical.extract_metric)

    # ── SQL (Vanna text-to-SQL) node ───────────────────────────────────────
    # If no SQLNodeFactory is provided, tabular intent falls through to route.
    if _has_sql:
        g.add_node("query_sql", sql.query_sql)
    else:
        def _sql_noop(state: RAGState) -> dict:  # noqa: E306
            return {}
        g.add_node("query_sql", _sql_noop)

    # ── Fund performance (per-scheme NAV/return/CAGR) node ─────────────────
    # If no FundPerformanceNodeFactory is available, these queries fall
    # through to route, same fallback pattern as query_sql above.
    if _has_fund_performance:
        g.add_node("fund_performance_sql", fund_performance.query_fund_performance)
    else:
        def _fund_performance_noop(state: RAGState) -> dict:  # noqa: E306
            return {}
        g.add_node("fund_performance_sql", _fund_performance_noop)

    # ── Reasoning (causal "why" queries) node ───────────────────────────────
    g.add_node("reasoning", reasoning.reasoning_node)

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

    # ── Analytical agent edges (Option A — full month aggregation) ────────────
    # Entry branch: tabular → query_sql, range → plan_years, else → route
    g.add_conditional_edges(
        "analyze_query",
        is_range_query,
        {
            "query_sql": "query_sql", "plan_years": "plan_years", "route": "route",
            "format_response": "format_response", "reasoning": "reasoning",
            "fund_performance_sql": "fund_performance_sql",
        },
    )

    # SQL path: feeds the full guardrail + LLM pipeline via pre_guardrail.
    # query_sql sets citations (SQL table as passage [1]), is_analytical=True,
    # and sql_context=True so generate uses the "sql" system prompt.
    if _has_sql:
        g.add_edge("query_sql", "pre_guardrail")
    else:
        g.add_edge("query_sql", "route")  # fallback: treat as normal query

    # Fund performance path: same contract as query_sql — see
    # nodes_fund_performance.py's query_fund_performance for the field-by-field
    # rationale (it sets citations/is_analytical/sql_context identically).
    if _has_fund_performance:
        g.add_edge("fund_performance_sql", "pre_guardrail")
    else:
        g.add_edge("fund_performance_sql", "route")  # fallback: treat as normal query

    # Reasoning path: same pre_guardrail feed as query_sql. reasoning_node
    # sets structured_answer directly (the explanation is already rendered
    # from reasoning_rules.yaml), so generate() short-circuits without an
    # LLM call — same deterministic-answer pattern query_sql uses.
    g.add_edge("reasoning", "pre_guardrail")

    # Outer loop: plan_years → (DB-first or chunk loop) → aggregate_year → synthesize
    # DB-first: plan_years → query_db_stats → synthesize (single SQL, no LLM loop)
    # Chunk loop: plan_years → plan_months → retrieve_month → ... → aggregate_year
    g.add_conditional_edges(
        "plan_years",
        after_plan_years,
        {"query_db_stats": "query_db_stats", "plan_months": "plan_months", "route": "route"},
    )
    g.add_edge("query_db_stats", "synthesize")
    g.add_edge("plan_months", "retrieve_month")

    # Inner loop: retrieve_month → extract_month_metric → [next month | aggregate_year]
    g.add_edge("retrieve_month", "extract_month_metric")
    g.add_conditional_edges(
        "extract_month_metric",
        after_extract_month,
        {"retrieve_month": "retrieve_month", "aggregate_year": "aggregate_year"},
    )

    # After year is aggregated: next year → plan_months | all done → synthesize
    g.add_conditional_edges(
        "aggregate_year",
        after_aggregate_year,
        {"plan_months": "plan_months", "synthesize": "synthesize"},
    )

    g.add_edge("synthesize", "post_guardrail")   # reuses existing guardrail path

    # Legacy snapshot path edges (kept so graph validates; not activated)
    g.add_conditional_edges(
        "extract_metric",
        after_extract,
        {"retrieve_year": "retrieve_year", "synthesize": "synthesize"},
    )

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

    return g.compile(checkpointer=checkpointer)
