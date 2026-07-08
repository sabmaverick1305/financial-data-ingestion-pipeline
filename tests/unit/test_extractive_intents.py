from __future__ import annotations

from types import SimpleNamespace

from financial_pipeline.augmentation.citations import Citation
from financial_pipeline.graph.nodes import NodeFactory


class _FailingGenerator:
    def generate(self, *_args, **_kwargs):  # pragma: no cover - should never run
        raise AssertionError("extractive intents should not call the LLM")


def test_trend_intent_uses_extractives() -> None:
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
            excerpt="Net inflows grew steadily from ₹80 crore in April to ₹120 crore in June 2024.",
            confidence=0.88,
            rank_method="fallback",
        )
    ]

    result = factory.generate(
        {
            "query": "Show the trend in net inflows",
            "intent": SimpleNamespace(intent_type="trend"),
            "citations": citations,
            "structured_answer": None,
            "sql_context": False,
        }
    )

    assert result["generation_meta"]["provider"] == "rules"
    assert "[1]" in result["answer"]
    assert "Net inflows grew steadily" in result["answer"]

