import pandas as pd
import pytest

from financial_pipeline.processing.transformer import DataTransformer


@pytest.fixture
def transformer() -> DataTransformer:
    return DataTransformer(source="test")


def test_normalize_columns(transformer: DataTransformer) -> None:
    df = pd.DataFrame({"Price USD": [1.0], "Volume ": [100]})
    result = transformer.normalize_columns(df)
    assert list(result.columns) == ["price_usd", "volume"]


def test_cast_numeric(transformer: DataTransformer) -> None:
    df = pd.DataFrame({"price": ["1.5", "2.0", "bad"]})
    result = transformer.cast_numeric(df, ["price"])
    assert result["price"].isna().sum() == 1


def test_drop_duplicates(transformer: DataTransformer) -> None:
    df = pd.DataFrame({"id": [1, 1, 2], "val": ["a", "a", "b"]})
    result = transformer.drop_duplicates(df, subset=["id"])
    assert len(result) == 2
