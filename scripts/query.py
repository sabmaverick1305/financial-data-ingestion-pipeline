"""Quick interactive query runner — spins up the full graph and runs one or more queries.

Usage:
  .venv/bin/python scripts/query.py "What was the AUM of Large Cap in Dec 2024?"
  .venv/bin/python scripts/query.py --queries queries.txt
  .venv/bin/python scripts/query.py   # interactive REPL
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_graph = None


def _init_graph():
    global _graph
    if _graph is not None:
        return _graph

    from financial_pipeline.config import settings
    from financial_pipeline.graph import build_graph, NodeFactory, AnalyticalNodeFactory
    from financial_pipeline.retrieval.retriever import Retriever
    from financial_pipeline.storage.document_repo import DocumentRepository
    from financial_pipeline.text_to_sql.vanna_agent import build_vanna_agent
    from financial_pipeline.graph.nodes_sql import SQLNodeFactory

    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY",    settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_API_KEY",    settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT",    settings.langchain_project)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true" if settings.langchain_tracing_v2 else "false")

    repo      = DocumentRepository(settings.postgres_url)
    retriever = Retriever(repo, settings.embed_model)
    factory   = NodeFactory(repo=repo, retriever=retriever)
    analytical = AnalyticalNodeFactory(repo=repo)

    sql_factory = None
    if settings.openai_api_key:
        try:
            vn = build_vanna_agent(
                anthropic_api_key=settings.openai_api_key,
                postgres_url=settings.postgres_url,
            )
            sql_factory = SQLNodeFactory(vanna=vn)
            print("SQL agent: ready")
        except Exception as exc:
            print(f"SQL agent: unavailable ({exc.__class__.__name__}: {exc!s:.80})")
            print("  -> SQL/tabular queries will fall through to RAG path")

    _graph = build_graph(factory, analytical=analytical, sql=sql_factory)
    return _graph


def run_query(question: str) -> None:
    graph = _init_graph()
    print(f"\n{'─'*70}")
    print(f"Q: {question}")
    print("─"*70)

    result = graph.invoke({"query": question, "retry_count": 0, "repair_count": 0})

    resp     = result.get("response", {})
    intent   = result.get("intent")
    route    = result.get("route", "?")
    grade    = resp.get("grade", result.get("grade", "?"))
    citations = result.get("citations", [])
    gen_meta  = result.get("generation_meta", {})

    intent_type  = getattr(intent, "intent_type", "?")
    needs_sql    = result.get("sql_context", False)
    is_analytical = result.get("is_analytical", False)

    # Route indicator — SQL and analytical paths bypass the `route` node
    if needs_sql:
        route_tag = "[SQL]"
    elif is_analytical:
        route_tag = "[ANALYTICAL]"
    else:
        route_tag = f"[{route.upper()}]" if route != "?" else "[RAG]"
    print(f"Route: {route_tag}  Intent: {intent_type}  Grade: {grade}")

    # If SQL path — show the SQL
    if needs_sql and gen_meta.get("sql"):
        print(f"\nSQL:\n  {gen_meta['sql']}")
        warnings = gen_meta.get("policy_warnings", [])
        if warnings:
            print(f"  ⚠ {', '.join(warnings)}")

    # Answer
    answer = resp.get("answer") or ""
    print(f"\nAnswer:\n{answer}")

    # Citations summary
    if citations:
        print(f"\nCitations ({len(citations)}):")
        for c in citations[:5]:
            src = getattr(c, "source_type", "?") or "?"
            yr  = getattr(c, "period_year", "") or ""
            mo  = getattr(c, "period_month", "") or ""
            conf = getattr(c, "confidence", 0) or 0
            excerpt = (getattr(c, "excerpt", "") or "")[:80].replace("\n", " ")
            period = f"{yr}/{mo:02d}" if mo else str(yr)
            print(f"  [{getattr(c,'number','-')}] {src} {period} conf={conf:.2f}  \"{excerpt}...\"")

    # Guardrail
    g = resp.get("guardrail", {})
    if g:
        blocked = g.get("blocked", False)
        fsc     = g.get("faithfulness_score")
        fsc_str = "n/a" if fsc is None else f"{fsc:.3f}"
        print(f"\nGuardrail: blocked={blocked}  faithfulness={fsc_str}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="Single query to run")
    parser.add_argument("--queries", help="File with one query per line")
    args = parser.parse_args()

    print("Initialising pipeline…")
    _init_graph()
    print("Ready.\n")

    if args.queries:
        for line in Path(args.queries).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                run_query(line)
    elif args.question:
        run_query(args.question)
    else:
        print("Interactive mode — type a question, blank line to quit.")
        while True:
            try:
                q = input("\nQ> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            run_query(q)


if __name__ == "__main__":
    main()
