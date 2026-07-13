"""Registry of AMC websites to scrape for Scheme Information Documents (SIDs).

SEBI has no single bulk SID feed — each AMC publishes its own SIDs on its
own website, in whatever page structure/CMS it happens to run. This is a
scaffold covering a handful of confirmed-working sources; extend
AMC_SID_SOURCES as more AMCs are added. Two other major AMCs (HDFC, ICICI
Prudential) were tried and are NOT included here — both return 403/404 to a
plain GET, most likely bot detection or content that's only reachable via a
JS-rendered client-side fetch, neither of which PageScraper's static-HTML/
Next.js-RSC extraction can currently reach.

amc_entity_id matches domain/semantic/taxonomy.yaml's amcs: entity_id, so a synced
SID's AMC can be cross-referenced against the same ontology used elsewhere
(mf_scheme_master.amc_name, the AMFI PDF pipeline's AMC recognition).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AMCSidSource:
    amc_entity_id: str  # matches domain/semantic/taxonomy.yaml's amcs: entity_id
    amc_name: str
    page_url: str
    note: str = ""


AMC_SID_SOURCES: list[AMCSidSource] = [
    AMCSidSource(
        amc_entity_id="sbi_mf",
        amc_name="SBI Mutual Fund",
        page_url="https://www.sbimf.com/offer-document-sid-kim",
        note="Static HTML — 16 PDF links found in testing (mix of SIDs and scheme-differentiation docs).",
    ),
    AMCSidSource(
        amc_entity_id="axis_mf",
        amc_name="Axis Mutual Fund",
        page_url="https://www.axismf.com/downloads",
        note=(
            "Next.js (RSC) page. Only 1 PDF link found in testing — the real SID list is "
            "likely behind a client-side fund-category filter not present in the initial "
            "RSC payload. Kept in the registry as a known-weak source to improve later."
        ),
    ),
    AMCSidSource(
        amc_entity_id="nippon_india_mf",
        amc_name="Nippon India Mutual Fund",
        page_url="https://mf.nipponindiaim.com/investor-service/downloads/scheme-information-document",
        note="Static HTML — 670 PDF links found in testing, the richest source by far.",
    ),
]
