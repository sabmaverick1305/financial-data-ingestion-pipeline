from __future__ import annotations

import pytest

from financial_pipeline.graph.nodes_sql import SQLNodeFactory


def test_query_sql_raises_clear_error_when_vanna_is_unavailable() -> None:
    factory = SQLNodeFactory(vanna=object())

    with pytest.raises(RuntimeError, match="optional 'vanna' dependency"):
        factory.query_sql({"query": "show me a table"})
