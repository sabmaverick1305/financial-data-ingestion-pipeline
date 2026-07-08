from __future__ import annotations

from financial_pipeline.graph.nodes_analytical import AnalyticalNodeFactory


class _FailingGenerator:
    def generate(self, *_args, **_kwargs):  # pragma: no cover - should never run
        raise AssertionError("synthesize should not call the LLM")


def test_synthesize_is_deterministic() -> None:
    factory = AnalyticalNodeFactory(repo=object(), generator=_FailingGenerator())

    result = factory.synthesize(
        {
            "extraction_metric": "funds_mobilized",
            "target_scheme": "large cap",
            "year_results": {
                2023: {"value": "120.00", "months_found": 12, "months_total": 12, "is_stock": False},
                2024: {"value": "180.00", "months_found": 10, "months_total": 12, "is_stock": False},
            },
        }
    )

    assert result["is_analytical"] is True
    assert "Annual totals ranged from ₹120.00 crore in 2023 to ₹180.00 crore in 2024." in result["answer"]
    assert result["generation_meta"]["provider"] == "rules"

