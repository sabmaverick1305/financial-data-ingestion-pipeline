"""RAGAS-based answer quality metrics for Phase 4.

Wraps ragas 0.4.x evaluate() using Claude Haiku as the judge LLM and the
project's existing sentence-transformer model for embeddings.

Supported metrics:
  - faithfulness        (LLM): are all claims in the answer supported by the contexts?
  - answer_relevancy    (LLM + embed): is the answer relevant to the question?
  - context_precision   (LLM): are the retrieved contexts ranked well?
  - context_recall      (LLM): does context cover everything in the ground truth?
"""
from __future__ import annotations

import warnings
from typing import Any

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")


def _build_llm():
    """Return a ragas-compatible LLM wrapper backed by Claude Haiku."""
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper
    from financial_pipeline.config import settings

    api_key = settings.openai_api_key  # sk-ant-... stored here per convention
    model = settings.anthropic_model   # claude-haiku-4-5-20251001

    lc_llm = ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=0,
        max_tokens=1024,
    )
    return LangchainLLMWrapper(lc_llm)


def _build_embeddings():
    """Return a ragas-compatible embeddings wrapper using the project's embed model."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from financial_pipeline.config import settings

    lc_embed = HuggingFaceEmbeddings(model_name=settings.embed_model)
    return LangchainEmbeddingsWrapper(lc_embed)


def _build_dataset(
    questions: list[str],
    answers: list[str],
    contexts_list: list[list[str]],
    ground_truths: list[str] | None = None,
):
    """Build a ragas Dataset from raw lists."""
    from datasets import Dataset

    data: dict[str, list] = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    return Dataset.from_dict(data)


def run_ragas(
    questions: list[str],
    answers: list[str],
    contexts_list: list[list[str]],
    ground_truths: list[str] | None = None,
    metric_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run RAGAS evaluation and return per-metric mean scores.

    Parameters
    ----------
    questions:
        The user queries.
    answers:
        The pipeline-generated answers for each query.
    contexts_list:
        For each query, the list of context strings that were retrieved
        (i.e. citation excerpts passed to the LLM).
    ground_truths:
        Optional reference answers (required for context_recall).
    metric_names:
        Subset of ["faithfulness", "answer_relevancy", "context_precision",
        "context_recall"]. Defaults to faithfulness + answer_relevancy
        (the two that don't need ground_truth).

    Returns
    -------
    dict with keys: scores (per-metric means), per_row (list of per-query scores),
    n_queries, errors.
    """
    if not questions:
        return {"scores": {}, "per_row": [], "n_queries": 0, "errors": ["No questions provided"]}

    # Pick metrics
    if metric_names is None:
        metric_names = ["faithfulness", "answer_relevancy"]
        if ground_truths:
            metric_names += ["context_precision", "context_recall"]

    from ragas.metrics import (
        faithfulness as _faithfulness,
        answer_relevancy as _answer_relevancy,
        context_precision as _context_precision,
        context_recall as _context_recall,
    )
    from ragas import evaluate

    metric_map = {
        "faithfulness": _faithfulness,
        "answer_relevancy": _answer_relevancy,
        "context_precision": _context_precision,
        "context_recall": _context_recall,
    }
    metrics = [metric_map[m] for m in metric_names if m in metric_map]

    dataset = _build_dataset(questions, answers, contexts_list, ground_truths)

    llm = _build_llm()
    embeddings = _build_embeddings()

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            show_progress=True,
        )
    except Exception as exc:
        return {"scores": {}, "per_row": [], "n_queries": len(questions), "errors": [str(exc)]}

    scores: dict[str, float] = {}
    for m in metric_names:
        try:
            scores[m] = float(result[m])
        except (KeyError, TypeError):
            scores[m] = None  # type: ignore[assignment]

    per_row: list[dict] = []
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        for _, row in df.iterrows():
            entry = {"question": row.get("question")}
            for m in metric_names:
                entry[m] = row.get(m)
            per_row.append(entry)

    return {
        "scores": scores,
        "per_row": per_row,
        "n_queries": len(questions),
        "errors": [],
    }
