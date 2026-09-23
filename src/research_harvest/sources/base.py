"""Shared plumbing for the REST clients: politeness, retries, and one error type."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any, Protocol

import requests

from research_harvest.models import Article

logger = logging.getLogger(__name__)


class SourceError(RuntimeError):
    """A source could not answer. Carries a message meant for the operator."""


class Source(Protocol):
    """What every source must offer."""

    name: str

    def search(self, query: str, *, limit: int, year_from: int | None, year_to: int | None) -> Iterable[Article]:
        ...


class RateLimiter:
    """
    Keeps at least `min_interval` seconds between calls.

    Both APIs are free, published with a request ceiling, and run by people who
    would rather not be hammered. Pacing to the ceiling is also faster than
    sleeping a fixed amount between calls.
    """

    def __init__(self, requests_per_second: float) -> None:
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._last: float = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


class HttpClient:
    """A small requests wrapper with pacing, retries and a real User-Agent."""

    def __init__(
        self,
        *,
        requests_per_second: float = 3.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        user_agent: str = "research-harvest/1.0 (+https://github.com/vipul21435/Web-Scraping-Tools-for-Research-Paper)",
        session: requests.Session | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = timeout
        self.max_retries = max_retries
        self.limiter = RateLimiter(requests_per_second)

    def get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self.limiter.wait()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("Request to %s failed (attempt %d): %s", url, attempt, exc)
            else:
                # 429 and 5xx are worth retrying; a 400 will fail the same way forever.
                if response.status_code < 400:
                    return response
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = SourceError(f"{url} returned HTTP {response.status_code}")
                    logger.warning("HTTP %s from %s (attempt %d)", response.status_code, url, attempt)
                else:
                    raise SourceError(f"{url} returned HTTP {response.status_code}")

            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 30))

        raise SourceError(f"Giving up on {url} after {self.max_retries} attempts: {last_error}")
