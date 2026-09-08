from financial_pipeline.config import settings
from financial_pipeline.intelligence.capability_registry import CapabilityRegistry
from financial_pipeline.intelligence.category_ontology import CategoryOntology
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction
from financial_pipeline.retrieval.rag import RAGPipeline


class DummyRetriever:
    pass


def test_category_taxonomy_rejects_polluted_scheme_labels() -> None:
    ontology = CategoryOntology()
    assert ontology.canonicalize("Equity Scheme - Mid Cap Fund") == "Mid Cap"
    assert ontology.canonicalize("Hybrid Scheme - Aggressive Hybrid Fund") == "Aggressive Hybrid"
    assert ontology.canonicalize("1098 Days") is None
    assert ontology.canonicalize("Growth") is None
    assert ontology.canonicalize("Direct") is None


def test_rag_uses_provider_active_model_by_default() -> None:
    rag = RAGPipeline(DummyRetriever())
    assert rag._model == settings.active_llm_model


def test_model_not_found_is_non_retryable_configuration_failure() -> None:
    registry = CapabilityRegistry()
    def handler(action):
        raise RuntimeError("Error code: 404 - model: gpt-4o-mini not_found_error")
    registry.register(ActionType.RETRIEVE_EVIDENCE, handler)
    observation = registry.execute(ResearchAction(ActionType.RETRIEVE_EVIDENCE))
    assert observation.failure_domain == "configuration"
    assert observation.retryable is False
    assert observation.replannable is False
