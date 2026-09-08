from financial_pipeline.intelligence.production_capabilities import ProductionCapabilityPack
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction, ActionStatus
from financial_pipeline.retrieval.rag import RAGResponse

class EmptyRepo:
    pass

def _action():
    return ResearchAction(
        ActionType.RETRIEVE_EVIDENCE,
        evidence_types=("prospectus", "fact_sheet", "strategy"),
        parameters={
            "candidate_names": ["Alpha Fund Direct Growth"],
            "document_search_names": ["alpha fund"],
            "query": "prospectus fact sheet strategy for alpha fund",
        },
    )

def test_zero_identity_coverage_does_not_repeat_document_discovery():
    class Rag:
        def __init__(self):
            self.ask_documentary_calls = 0

        def documentary_coverage(self, **kwargs):
            return {
                "alpha fund": {
                    "coverage_ratio": 0.0,
                    "missing_document_types": [
                        "prospectus", "fact_sheet", "strategy"
                    ],
                    "by_type": {
                        "prospectus": {"document_ids": []},
                        "fact_sheet": {"document_ids": []},
                        "strategy": {"document_ids": []},
                    },
                }
            }

        def ask_documentary(self, *args, **kwargs):
            self.ask_documentary_calls += 1
            raise AssertionError("document discovery should not be repeated")

    rag = Rag()
    result = ProductionCapabilityPack(
        repository=EmptyRepo(),
        rag_pipeline=rag,
    ).documentary(_action())

    assert result.status is ActionStatus.PARTIAL
    assert rag.ask_documentary_calls == 0
    assert result.result["documentary_coverage_ratio"] == 0.0
    assert len(result.result["ingestion_backlog"]) == 3

def test_resolved_document_ids_continue_through_full_rag():
    class Rag:
        def __init__(self):
            self.ids = None

        def documentary_coverage(self, **kwargs):
            return {
                "alpha fund": {
                    "coverage_ratio": 1.0,
                    "missing_document_types": [],
                    "by_type": {
                        "prospectus": {"document_ids": ["doc-1"]},
                        "fact_sheet": {"document_ids": ["doc-2"]},
                        "strategy": {"document_ids": ["doc-3"]},
                    },
                }
            }

        def ask_documentary_ids(self, query, *, document_ids):
            self.ids = list(document_ids)
            return RAGResponse(
                query=query,
                answer="verified documentary evidence",
                sources=[
                    {
                        "document_id": "doc-1",
                        "document_type": "fund_prospectus",
                        "text": "Alpha Fund prospectus",
                    }
                ],
                retrieval_count=1,
            )

    rag = Rag()
    result = ProductionCapabilityPack(
        repository=EmptyRepo(),
        rag_pipeline=rag,
    ).documentary(_action())

    assert set(rag.ids) == {"doc-1", "doc-2", "doc-3"}
    assert result.result["documentary_coverage_ratio"] == 1.0
