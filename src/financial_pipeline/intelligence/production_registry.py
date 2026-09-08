"""Composition root for production FIES reasoning capabilities."""
from __future__ import annotations
from financial_pipeline.intelligence.capability_contracts import CapabilityContract
from financial_pipeline.intelligence.capability_metadata import CapabilityMetadata, CapabilityMode
from financial_pipeline.intelligence.capability_registry import CapabilityRegistry
from financial_pipeline.intelligence.production_capabilities import ProductionCapabilityPack
from financial_pipeline.intelligence.research_plan import ActionType

def build_production_registry(pack: ProductionCapabilityPack) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(ActionType.DISCOVER_CATEGORIES, pack.discover_categories, contract=CapabilityContract(), metadata=CapabilityMetadata(source_authority="mfapi/Postgres", freshness_seconds=86400, estimated_latency_ms=20))
    registry.register(ActionType.DISCOVER_FUNDS, pack.discover_funds, contract=CapabilityContract(), metadata=CapabilityMetadata(required_inputs=(), source_authority="mfapi/Postgres", freshness_seconds=86400, estimated_latency_ms=50))
    registry.register(ActionType.FETCH_PERFORMANCE, pack.performance, contract=CapabilityContract(supported_metrics=("return_1y","return_3y_cagr","return_5y_cagr","return_10y_cagr")), metadata=CapabilityMetadata(required_inputs=("scheme_code",), source_authority="mfapi NAV/Postgres", freshness_seconds=86400, estimated_latency_ms=30))
    registry.register(ActionType.COMPUTE_RISK, pack.risk, contract=CapabilityContract(supported_metrics=("volatility","sharpe_ratio","max_drawdown")), metadata=CapabilityMetadata(required_inputs=("scheme_code","nav_history"), source_authority="mfapi NAV/Postgres", freshness_seconds=86400, estimated_latency_ms=80))
    registry.register(ActionType.COMPARE_PEERS, pack.peer_compare, contract=CapabilityContract(supported_metrics=("percentile_rank","peer_outperformance")), metadata=CapabilityMetadata(required_inputs=("category",), source_authority="mfapi/Postgres", freshness_seconds=86400, estimated_latency_ms=80))
    registry.register(ActionType.FETCH_AUM, pack.aum, contract=CapabilityContract(supported_metrics=("aum","aum_trend","aum_growth_rate")), metadata=CapabilityMetadata(required_inputs=("category",), source_authority="AMFI", freshness_seconds=2678400, estimated_latency_ms=50, allow_partial=True))
    registry.register(ActionType.FETCH_FLOWS, pack.flows, contract=CapabilityContract(supported_metrics=("net_inflow","flow_trend","redemption_rate","relative_flows")), metadata=CapabilityMetadata(required_inputs=("category",), source_authority="AMFI", freshness_seconds=2678400, estimated_latency_ms=50, allow_partial=True))
    registry.register(ActionType.RETRIEVE_EVIDENCE, pack.documentary, contract=CapabilityContract(supported_evidence_types=("fund_prospectus","fund_fact_sheet","fund_strategy_document","regulatory_filing","annual_report","portfolio_disclosure"), evidence_aliases={"disclosures":"portfolio_disclosure"}), metadata=CapabilityMetadata(source_authority="AMFI/SEBI documents", mode=CapabilityMode.RETRIEVAL, estimated_latency_ms=1000, allow_partial=True))
    registry.register(ActionType.CHECK_CONTRADICTIONS, pack.contradictions, contract=CapabilityContract(), metadata=CapabilityMetadata(source_authority="FIES deterministic rules", estimated_latency_ms=5))
    return registry
