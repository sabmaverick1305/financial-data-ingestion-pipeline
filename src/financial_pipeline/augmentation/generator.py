"""Answer Generator — Stage 4 of the augmentation layer.

Provider-agnostic LLM call with:
  - Auto-detection of OpenAI vs. Anthropic from key prefix
  - Retry with exponential backoff on transient errors
  - Temperature tuned per intent (lower for factual, higher for summaries)
  - Token usage tracking
  - Streaming support (for CLI / WebSocket use)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from financial_pipeline.config import settings

log = structlog.get_logger()

# Temperature per intent — factual queries need low temp, summaries more freedom
INTENT_TEMPERATURE: dict[str, float] = {
    "factual": 0.05,
    "regulatory": 0.10,
    "tabular": 0.05,
    "trend": 0.15,
    "comparison": 0.15,
    "definition": 0.20,
    "lookup": 0.05,
    "default": 0.10,
}


@dataclass
class GenerationResult:
    answer: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class AnswerGenerator:
    """Provider-agnostic LLM answer generation.

    Reads provider from ``settings.llm_provider`` (auto-detected from key
    prefix). Supports OpenAI, Anthropic, and any OpenAI-compatible endpoint.
    """

    MAX_RETRIES = 2

    def __init__(self, model: str | None = None, provider: str | None = None) -> None:
        """provider: force "openai" or "anthropic" for this instance, bypassing
        settings.llm_provider auto-detection. None (default) preserves the
        existing global-settings behavior exactly."""
        self._provider_override = provider
        if model:
            self._model = model
        elif provider == "openai":
            self._model = settings.openai_model
        elif provider == "anthropic":
            self._model = settings.anthropic_model
        else:
            self._model = settings.active_llm_model
        self._client = None

    @property
    def _provider(self) -> str:
        return self._provider_override or settings.llm_provider

    def generate(
        self,
        messages: list[dict],
        intent_type: str = "default",
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        temperature = INTENT_TEMPERATURE.get(intent_type, 0.10)

        if settings.use_bedrock and self._provider_override is None:
            model    = model or settings.bedrock_model_id
            provider = "bedrock"
            answer, usage = self._call_anthropic(messages, model, temperature, max_tokens)
        else:
            model    = model or self._model
            provider = self._provider
            for attempt in range(self.MAX_RETRIES + 1):
                try:
                    if provider == "anthropic":
                        answer, usage = self._call_anthropic(messages, model, temperature, max_tokens)
                    else:
                        answer, usage = self._call_openai(messages, model, temperature, max_tokens)
                    break
                except Exception as exc:
                    if attempt == self.MAX_RETRIES:
                        raise
                    wait = 2**attempt
                    log.warning("generator.retry", attempt=attempt + 1, wait=wait, error=str(exc))
                    time.sleep(wait)

        latency = int((time.perf_counter() - t0) * 1000)
        log.info(
            "generator.done",
            provider=provider,
            model=model,
            latency_ms=latency,
            prompt_tokens=usage[0],
            completion_tokens=usage[1],
        )

        return GenerationResult(
            answer=answer,
            model=model,
            provider=provider,
            prompt_tokens=usage[0],
            completion_tokens=usage[1],
            latency_ms=latency,
        )

    def is_configured(self) -> bool:
        return bool(settings.openai_api_key) or settings.use_bedrock

    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            if settings.use_bedrock and self._provider_override is None:
                from anthropic import AnthropicBedrock
                self._client = AnthropicBedrock(
                    aws_region=settings.bedrock_region,
                    # Pass explicit credentials only when set; otherwise falls back
                    # to boto3 credential chain (IAM role, env vars, ~/.aws/credentials)
                    aws_access_key=settings.aws_access_key_id or None,
                    aws_secret_key=settings.aws_secret_access_key or None,
                )
                log.info("generator.bedrock_client_ready",
                         region=settings.bedrock_region,
                         model=settings.bedrock_model_id)
            elif self._provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic(api_key=settings.openai_api_key)
            else:
                # The routed "openai" tier uses its own dedicated key, since
                # openai_api_key is frequently actually an Anthropic key here.
                api_key = settings.openai_api_key
                if self._provider_override == "openai":
                    api_key = settings.openai_mini_api_key or settings.openai_api_key
                if not api_key:
                    raise ValueError(
                        "No LLM API key configured. Set OPENAI_API_KEY "
                        "(or OPENAI_MINI_API_KEY for the routed tier) or USE_BEDROCK=true in .env"
                    )
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=settings.openai_base_url,
                )
        return self._client

    def _call_anthropic(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, tuple[int, int]]:
        client = self._get_client()
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_ms = [m for m in messages if m["role"] != "system"]
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=user_ms,
        )
        answer = resp.content[0].text
        usage = (resp.usage.input_tokens, resp.usage.output_tokens)
        return answer, usage

    def _call_openai(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, tuple[int, int]]:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = resp.choices[0].message.content or ""
        usage = (resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return answer, usage
