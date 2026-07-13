"""FIES — Financial Intelligence Evaluation Suite orchestrator.

Usage:
  python eval/run_eval.py --phase all
  python eval/run_eval.py --phase intent
  python eval/run_eval.py --phase sql retrieval
  python eval/run_eval.py --phase all --out eval_results.json
  python eval/run_eval.py --phase sql --ids Q001 Q002 Q003
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

PHASES = ["intent", "sql", "retrieval", "answer", "guardrail", "data_quality"]


def _upload_to_langsmith(suite_result: dict, phases_to_run: list[str]) -> str | None:
    """Upload the full suite result to LangSmith as a single run.

    Returns a human-readable locator (URL if resolvable, else "project/run_id"),
    or None if LangSmith isn't configured / the upload failed.
    """
    from financial_pipeline.config import settings

    if not settings.langsmith_api_key:
        return None

    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)

    try:
        from langsmith import Client
    except ImportError:
        print("[FIES] langsmith package not installed — skipping upload", file=sys.stderr)
        return None

    project_name = f"{settings.langchain_project}-fies-eval"
    run_id = uuid.uuid4()
    now = datetime.now(timezone.utc)

    try:
        client = Client()
        client.create_run(
            id=run_id,
            project_name=project_name,
            name=f"FIES eval — {'+'.join(phases_to_run)}",
            run_type="chain",
            inputs={"phases": phases_to_run},
            outputs=suite_result,
            start_time=now,
            end_time=now,
        )
    except Exception as exc:
        print(f"[FIES] LangSmith upload failed: {exc}", file=sys.stderr)
        return None

    try:
        run = client.read_run(run_id)
        return client.get_run_url(run=run, project_name=project_name)
    except Exception:
        return f"{project_name}/{run_id}"


def _run_phase(phase: str, ids: list[str] | None, run_ragas: bool = False) -> dict:
    start = time.monotonic()
    if phase == "intent":
        from eval.runners.run_intent_eval import run
        result = run(ids)
    elif phase == "sql":
        from eval.runners.run_sql_eval import run
        result = run(ids)
    elif phase == "retrieval":
        from eval.runners.run_retrieval_eval import run
        result = run(ids)
    elif phase == "answer":
        from eval.runners.run_answer_eval import run
        result = run(ids, run_ragas=run_ragas)
    elif phase == "guardrail":
        from eval.runners.run_guardrail_eval import run
        result = run(ids)
    elif phase == "data_quality":
        from eval.runners.run_dataquality_eval import run
        result = run(ids)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    elapsed = time.monotonic() - start
    result["elapsed_sec"] = round(elapsed, 2)
    return result


def _print_summary(phase: str, result: dict) -> None:
    metrics = result.get("metrics", {})
    n = result.get("n_queries", 0)
    n_err = result.get("n_errors", 0)
    elapsed = result.get("elapsed_sec", 0)
    print(f"\n{'='*60}")
    print(f"  Phase: {phase.upper()}  |  Queries: {n}  |  Errors: {n_err}  |  {elapsed}s")
    print(f"{'='*60}")

    def _fmt(key: str, val) -> str:
        if isinstance(val, float):
            return f"  {key:<40} {val*100:6.1f}%"
        return f"  {key:<40} {val}"

    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, dict):
                    for sk2, sv2 in sub_v.items():
                        if isinstance(sv2, (int, float)):
                            print(_fmt(f"    {sub_k}.{sk2}", sv2))
                elif sub_v is not None:
                    print(_fmt(f"    {sub_k}", sub_v))
        elif v is not None:
            print(_fmt(k, v))


def main() -> None:
    parser = argparse.ArgumentParser(description="FIES evaluation orchestrator")
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=PHASES + ["all"],
        default=["all"],
        help="Which phases to run",
    )
    parser.add_argument("--ids", nargs="*", help="Restrict to specific query IDs")
    parser.add_argument("--out", default=None, help="Also write full results to this JSON file")
    parser.add_argument("--quiet", action="store_true", help="Skip per-phase console output")
    parser.add_argument("--ragas", action="store_true", help="Run RAGAS LLM metrics in the answer phase")
    args = parser.parse_args()

    phases_to_run = PHASES if "all" in args.phase else args.phase

    suite_result = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "phases": {},
    }

    total_start = time.monotonic()
    for phase in phases_to_run:
        print(f"\n[FIES] Running phase: {phase} …")
        try:
            result = _run_phase(phase, args.ids, run_ragas=args.ragas)
            suite_result["phases"][phase] = result
            if not args.quiet:
                _print_summary(phase, result)
        except Exception as exc:
            print(f"[FIES] Phase {phase} failed: {exc}", file=sys.stderr)
            suite_result["phases"][phase] = {"error": str(exc)}

    suite_result["total_elapsed_sec"] = round(time.monotonic() - total_start, 2)

    print(f"\n[FIES] Done in {suite_result['total_elapsed_sec']}s")

    # LangSmith is now the default sink for every run — a local JSON is only
    # written when explicitly requested via --out, or as a fallback if the
    # upload fails / LangSmith isn't configured, so results are never lost.
    locator = _upload_to_langsmith(suite_result, phases_to_run)
    if locator:
        print(f"[FIES] Results uploaded to LangSmith: {locator}")

    if args.out or not locator:
        results_dir = ROOT / "eval" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        if args.out:
            out_path = Path(args.out)
            # Bare filename (no directory component) still lands in eval/results/,
            # so eval_results_*.json never scatters into the repo root.
            if out_path.parent == Path("."):
                out_path = results_dir / out_path
        else:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_path = results_dir / f"eval_results_{ts}.json"

        out_path.write_text(json.dumps(suite_result, indent=2, default=str))
        reason = "" if args.out else " (LangSmith unavailable — fallback)"
        print(f"[FIES] Results written to {out_path}{reason}")


if __name__ == "__main__":
    main()
