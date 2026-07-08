"""Upload FIES eval results to LangSmith as experiments.

Creates one LangSmith dataset ("FIES — AMFI Pipeline Evaluation") and
five phase experiments (intent, sql, retrieval, answer, guardrail).
Each query becomes one run with per-case scores + aggregate phase metrics.

Usage:
    python scripts/upload_to_langsmith.py                       # latest eval_results_*.json
    python scripts/upload_to_langsmith.py eval_results_XYZ.json
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from langsmith import Client

DATASET_NAME = "FIES — AMFI Pipeline Evaluation"

# ── Query text helpers ────────────────────────────────────────────────────────

def load_query_texts() -> dict[str, str]:
    """Return {id: question_text} for all Q* and F* queries."""
    texts: dict[str, str] = {}

    corpus_path = ROOT / "eval/corpus/query_corpus.json"
    if corpus_path.exists():
        data = json.loads(corpus_path.read_text())
        for q in data.get("queries", []):
            texts[q["id"]] = q["query"]

    failure_path = ROOT / "eval/corpus/failure_corpus.json"
    if failure_path.exists():
        data = json.loads(failure_path.read_text())
        for fid, fdata in data.items():
            if fid.startswith("F"):
                texts[fid] = fdata["query"]

    return texts


# ── Dataset management ────────────────────────────────────────────────────────

def ensure_dataset(client: Client, query_texts: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Return (dataset_id, {query_id: example_id}).

    Creates the dataset + examples on first run; reuses on subsequent runs.
    """
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"  Reusing dataset: {dataset.id}")
        example_id_map = {
            ex.inputs["id"]: str(ex.id)
            for ex in client.list_examples(dataset_id=dataset.id)
        }
        # Add any new queries missing from the dataset
        missing = {qid: text for qid, text in query_texts.items() if qid not in example_id_map}
        if missing:
            print(f"  Adding {len(missing)} new examples to dataset …")
            inputs  = [{"id": qid, "question": text} for qid, text in sorted(missing.items())]
            outputs = [{} for _ in inputs]
            client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)
            # Refresh map with newly added examples
            example_id_map = {
                ex.inputs["id"]: str(ex.id)
                for ex in client.list_examples(dataset_id=dataset.id)
            }
        return str(dataset.id), example_id_map
    except Exception:
        pass

    dataset = client.create_dataset(
        DATASET_NAME,
        description=(
            "AMFI mutual fund pipeline — 5-phase evaluation suite (FIES). "
            "Covers intent routing, SQL quality, retrieval, answer quality, and guardrails."
        ),
    )
    print(f"  Created dataset: {dataset.id}")

    inputs = [{"id": qid, "question": text} for qid, text in query_texts.items()]
    outputs = [{} for _ in inputs]
    client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)

    example_id_map = {
        ex.inputs["id"]: str(ex.id)
        for ex in client.list_examples(dataset_id=dataset.id)
    }
    return str(dataset.id), example_id_map


# ── Phase uploaders ───────────────────────────────────────────────────────────

_RESULTS_KEY: dict[str, str] = {
    "intent": "predictions",
    "sql": "results",
    "retrieval": "results",
    "answer": "results",
    "guardrail": "results",
}


def _outputs_for(phase: str, result: dict) -> dict:
    if phase == "intent":
        return {
            "intent_type": result.get("intent_type"),
            "year": result.get("year"),
            "year_from": result.get("year_from"),
            "year_to": result.get("year_to"),
            "month": result.get("month"),
            "needs_analytical": result.get("needs_analytical"),
        }
    if phase == "sql":
        return {
            "sql": (result.get("sql") or "")[:800],
            "answer": (result.get("answer") or "")[:600],
            "hard_blocked": result.get("hard_blocked", False),
            "timed_out": result.get("timed_out", False),
        }
    if phase == "retrieval":
        citations = result.get("citations", [])
        return {
            "n_citations": len(citations),
            "years": sorted({c["period_year"] for c in citations if c.get("period_year")}),
            "months": sorted({c["period_month"] for c in citations if c.get("period_month")}),
        }
    if phase == "answer":
        return {"answer": (result.get("answer") or "")[:600]}
    if phase == "guardrail":
        return {
            "hard_blocked": result.get("hard_blocked", False),
            "scope_rejected": result.get("scope_rejected", False),
            "blocked_at_layer": result.get("blocked_at_layer"),
            "answer": (result.get("answer") or "")[:300],
        }
    return {}


def _scores_for(phase: str, result: dict) -> dict[str, float]:
    scores: dict[str, float] = {}
    if phase == "sql":
        scores["executed"] = 0.0 if result.get("hard_blocked") or result.get("timed_out") else 1.0
        scores["hard_blocked"] = 1.0 if result.get("hard_blocked") else 0.0
    elif phase == "retrieval":
        scores["has_citations"] = 1.0 if result.get("citations") else 0.0
    elif phase == "guardrail":
        scores["hard_blocked"] = 1.0 if result.get("hard_blocked") else 0.0
        scores["scope_rejected"] = 1.0 if result.get("scope_rejected") else 0.0
        blocked = result.get("hard_blocked") or result.get("scope_rejected")
        scores["any_blocked"] = 1.0 if blocked else 0.0
    return scores


def _aggregate_scores(metrics: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}

    def _walk(d: dict, pfx: str) -> None:
        for k, v in d.items():
            key = f"{pfx}{k}" if pfx else k
            if isinstance(v, (int, float)) and v is not None and v >= 0:
                out[key] = float(v)
            elif isinstance(v, dict):
                _walk(v, f"{key}.")

    _walk(metrics, prefix)
    return out


def upload_phase(
    client: Client,
    phase: str,
    phase_data: dict,
    dataset_id: str,
    example_id_map: dict[str, str],
    run_at: str,
) -> None:
    experiment_name = f"FIES/{phase.upper()} — {run_at[:10]}"
    metrics = phase_data.get("metrics", {})

    project = client.create_project(
        experiment_name,
        reference_dataset_id=dataset_id,
        metadata={"fies_phase": phase, "run_at": run_at},
        upsert=True,
    )
    print(f"  Experiment: '{experiment_name}'  ({project.id})")

    results = phase_data.get(_RESULTS_KEY.get(phase, "results"), [])
    if not results and phase == "intent":
        results = phase_data.get("predictions", [])

    now = datetime.now(timezone.utc)
    uploaded = 0
    for result in results:
        qid = result.get("id")
        if not qid:
            continue
        example_id = example_id_map.get(qid)
        run_id = uuid.uuid4()

        client.create_run(
            id=run_id,
            name=f"{phase}.{qid}",
            inputs={"id": qid},
            outputs=_outputs_for(phase, result),
            run_type="chain",
            project_name=experiment_name,
            reference_example_id=example_id,
            start_time=now,
            end_time=now,
        )

        for key, score in _scores_for(phase, result).items():
            client.create_feedback(run_id=run_id, key=key, score=score)

        uploaded += 1

    # Aggregate metrics as feedback on a summary run
    agg_run_id = uuid.uuid4()
    client.create_run(
        id=agg_run_id,
        name=f"{phase}.aggregate",
        inputs={"phase": phase},
        outputs={"metrics": metrics},
        run_type="chain",
        project_name=experiment_name,
        start_time=now,
        end_time=now,
    )
    for key, score in _aggregate_scores(metrics).items():
        client.create_feedback(run_id=agg_run_id, key=key, score=score)

    print(f"    {uploaded} runs + aggregate metrics uploaded")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) >= 2:
        results_path = Path(sys.argv[1])
    else:
        candidates = sorted(ROOT.glob("eval_results_*.json"), reverse=True)
        if not candidates:
            print("ERROR: No eval_results_*.json found in project root", file=sys.stderr)
            sys.exit(1)
        results_path = candidates[0]

    print(f"Loading {results_path.name}")
    data = json.loads(results_path.read_text())
    run_at = data.get("run_at", "unknown")
    print(f"  run_at: {run_at}")

    client = Client()
    print(f"  LangSmith API: {client.api_url}")

    print("\n[1/2] Ensuring dataset …")
    query_texts = load_query_texts()
    print(f"      {len(query_texts)} query texts loaded")
    dataset_id, example_id_map = ensure_dataset(client, query_texts)

    print("\n[2/2] Uploading phase experiments …")
    for phase, phase_data in data.get("phases", {}).items():
        if "error" in phase_data:
            print(f"  Skipping {phase} (error in results)")
            continue
        print(f"\n  Phase: {phase}")
        upload_phase(client, phase, phase_data, dataset_id, example_id_map, run_at)

    print(f"\n✓ Done.")
    print(f"  Dataset: https://smith.langchain.com/datasets/{dataset_id}")


if __name__ == "__main__":
    main()
