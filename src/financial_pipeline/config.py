from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "sqlite:///./pipeline.db"
    postgres_url: str = ""  # postgresql+psycopg2://user:pass@host:5432/dbname

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "console"

    # HTTP
    request_timeout: int = 30
    max_retries: int = 3

    # Pipeline
    batch_size: int = 1000

    # AWS / S3
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "ap-south-1"
    s3_bucket: str = ""
    s3_prefix: str = "bronze/amfi/nav"

    # HuggingFace
    hf_token: str = ""

    # Retrieval — embedding model
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384

    # LLM — supports OpenAI (sk-...) and Anthropic/Claude (sk-ant-...)
    # Provider is auto-detected from the key prefix.
    openai_api_key: str = ""  # OpenAI key  OR  Anthropic key (sk-ant-...)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"  # OpenAI default
    anthropic_model: str = "claude-haiku-4-5-20251001"  # Claude Haiku 4.5 — fast + cost-efficient

    # Dedicated key for the cost-routing tier (small JSON / classification tasks
    # forced onto OpenAI regardless of the main provider above). Separate field
    # because openai_api_key above is frequently actually an Anthropic key
    # (sk-ant-...) — see llm_provider auto-detection.
    openai_mini_api_key: str = ""  # set OPENAI_MINI_API_KEY in .env

    @property
    def llm_provider(self) -> str:
        """Auto-detect provider from key prefix."""
        if self.openai_api_key.startswith("sk-ant-"):
            return "anthropic"
        return "openai"

    @property
    def active_llm_model(self) -> str:
        """Return the correct model for the detected provider."""
        if self.llm_provider == "anthropic":
            return self.anthropic_model
        return self.openai_model

    # Retrieval API server
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_top_k: int = 8  # chunks returned per query by default
    api_cors_origins: str = "*"

    # LangSmith tracing
    langsmith_api_key: str = ""   # set LANGSMITH_API_KEY in .env
    langchain_project: str = "amfi-pipeline"
    langchain_tracing_v2: bool = False  # set LANGCHAIN_TRACING_V2=true in .env to enable

    # AWS Bedrock LLM backend (alternative to direct Anthropic API)
    # Requires: model enabled in Bedrock console + IAM permission bedrock:InvokeModel
    # Note: Claude is NOT available in ap-south-1 — use us-east-1 or us-west-2
    use_bedrock: bool = False
    bedrock_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-haiku-4-5-20251001-v1:0"

    # Ontology-driven retrieval query expansion (domain/semantic/thesaurus.yaml +
    # financial_relationships.yaml, via SemanticEngine). Additive RRF arm —
    # set ENABLE_ONTOLOGY_RETRIEVAL=false to fall back to pre-expansion behavior exactly.
    enable_ontology_retrieval: bool = True

    # Ontology-aware reranking bonus on top of the cross-encoder score
    # (see retrieval/ontology_reranker.py). Additive re-sort — set
    # ENABLE_ONTOLOGY_RERANKING=false to fall back to pure cross-encoder order.
    enable_ontology_reranking: bool = True

    # Causal "why did X change" reasoning engine (domain/semantic/reasoning_rules.yaml
    # matched against real computed metric directions — see graph/nodes_reasoning.py).
    # Set ENABLE_REASONING_ENGINE=false to route causal queries through the
    # standard tabular/RAG path instead (pre-this-feature behavior).
    enable_reasoning_engine: bool = True

    # Mutual fund NAV ingestion (mf_ingestion/) — separate dataset from the
    # AMFI PDF pipeline above, sourced from api.mfapi.in + an S3 scheme list.
    # Snapshots live at {prefix}/{YYYY-MM-DD}/mf_scheme_master.json — the sync
    # job resolves the latest dated snapshot automatically (see s3_source.py).
    mf_scheme_master_s3_prefix: str = "bronze/amfi/scheme_master"
    mfapi_default_start_date: str = "2000-01-01"
    mfapi_request_delay_seconds: float = 0.15
    mfapi_max_workers: int = 10

    # Postgres-backed LangGraph checkpointer (storage/checkpointer.py) —
    # persists every node's state per thread_id, enabling replay/resumability.
    # Set ENABLE_CHECKPOINTING=false to compile the graph with no checkpointer
    # at all (pre-this-feature behavior — fully stateless, no Postgres
    # dependency for graph execution).
    enable_checkpointing: bool = True


settings = Settings()
