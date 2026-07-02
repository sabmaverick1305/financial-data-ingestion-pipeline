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
    s3_prefix: str = "amfi/nav"

    # HuggingFace
    hf_token: str = ""

    # Retrieval — embedding model
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384

    # LLM — supports OpenAI (sk-...) and Anthropic/Claude (sk-ant-...)
    # Provider is auto-detected from the key prefix.
    openai_api_key: str = ""      # OpenAI key  OR  Anthropic key (sk-ant-...)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"          # OpenAI default
    anthropic_model: str = "claude-haiku-4-5-20251001"  # Claude Haiku 4.5 — fast + cost-efficient

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
    api_top_k: int = 8        # chunks returned per query by default
    api_cors_origins: str = "*"


settings = Settings()
