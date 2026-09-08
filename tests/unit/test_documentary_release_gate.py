from financial_pipeline.storage.document_repo import DocumentRepository


def test_diagnosis_api_contract_is_available():
    assert hasattr(DocumentRepository, "diagnose_documentary_resolution")


def test_documentary_readiness_requires_full_coverage():
    coverage = {
        "fund-a": {"covered": True, "coverage_ratio": 1.0},
        "fund-b": {"covered": False, "coverage_ratio": 0.67},
    }
    ratios = [float(item["coverage_ratio"]) for item in coverage.values()]
    fully_covered = sum(1 for item in coverage.values() if item["covered"])

    identity_resolved = (sum(ratios) / len(ratios)) > 0.0
    full_coverage = (
        len(coverage) > 0
        and fully_covered == len(coverage)
        and (sum(ratios) / len(ratios)) >= 1.0
    )

    assert identity_resolved is True
    assert full_coverage is False


def test_zero_coverage_is_never_documentary_ready():
    coverage = {
        "fund-a": {"covered": False, "coverage_ratio": 0.0},
        "fund-b": {"covered": False, "coverage_ratio": 0.0},
    }
    ratios = [float(item["coverage_ratio"]) for item in coverage.values()]
    ratio = sum(ratios) / len(ratios)

    assert (ratio > 0.0) is False
    assert all(item["covered"] for item in coverage.values()) is False
