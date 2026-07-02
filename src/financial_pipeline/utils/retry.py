import functools
from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import retry, stop_after_attempt, wait_fixed

F = TypeVar("F", bound=Callable[..., Any])


def with_retry(attempts: int = 3, wait_seconds: float = 1.0) -> Callable[[F], F]:
    """Decorator that retries a function on exception."""
    return retry(stop=stop_after_attempt(attempts), wait=wait_fixed(wait_seconds))  # type: ignore[return-value]
