from __future__ import annotations

from types import SimpleNamespace

from financial_pipeline.augmentation.citations import Citation
from financial_pipeline.graph.nodes import NodeFactory


class _FailingGenerator:
    def generate(self, *_args, **_kwargs):  # pragma: no cover - should never run
        raise AssertionError("extractive document answers should not call the LLM")


def test_factual_document_answers_are_extractively_grounded() -> None:
    factory = NodeFactory.__new__(NodeFactory)
    factory._generator = _FailingGenerator()

    citations = [
        Citation(
            number=1,
            source_type="chunk",
            file_name="report.pdf",
            period_year=2024,
            period_month=6,
            category="monthly",
            chunk_index=7,
            table_index=None,
            excerpt="Large cap funds reported net inflows of ₹120 crore in June 2024.",
            confidence=0.92,
            rank_method="fallback",
        )
    ]

    result = factory.generate(
        {
            "query": "What were the large cap fund inflows?",
            "intent": SimpleNamespace(intent_type="factual"),
            "citations": citations,
            "structured_answer": None,
            "sql_context": False,
        }
    )

    assert result["structured_answer"] == result["answer"]
    assert result["generation_meta"]["provider"] == "rules"
    assert "[1]" in result["answer"]
    assert "net inflows of ₹120 crore" in result["answer"]


def test_factual_document_answers_fail_soft_without_citations() -> None:
    factory = NodeFactory.__new__(NodeFactory)
    factory._generator = _FailingGenerator()

    result = factory.generate(
        {
            "query": "What were the large cap fund inflows?",
            "intent": SimpleNamespace(intent_type="factual"),
            "citations": [],
            "structured_answer": None,
            "sql_context": False,
        }
    )

    assert "couldn’t find enough supporting passages" in result["answer"]
    assert result["generation_meta"]["provider"] == "rules"
