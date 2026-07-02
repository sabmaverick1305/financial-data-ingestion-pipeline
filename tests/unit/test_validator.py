import pandas as pd
import pytest

from financial_pipeline.processing.validator import DataValidator, ValidationError


@pytest.fixture
def validator() -> DataValidator:
    return DataValidator(source="test")


def test_require_columns_passes(validator: DataValidator) -> None:
    df = pd.DataFrame({"price": [1.0], "volume": [100]})
    validator.require_columns(df, ["price", "volume"])


def test_require_columns_raises(validator: DataValidator) -> None:
    df = pd.DataFrame({"price": [1.0]})
    with pytest.raises(ValidationError, match="Missing required columns"):
        validator.require_columns(df, ["price", "volume"])


def test_check_nulls_raises(validator: DataValidator) -> None:
    df = pd.DataFrame({"price": [1.0, None]})
    with pytest.raises(ValidationError, match="nulls"):
        validator.check_nulls(df, ["price"])
