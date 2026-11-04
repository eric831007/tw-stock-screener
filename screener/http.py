"""A polite HTTP client: one request every few seconds, with retries.

The TWSE website blocks IPs that send requests too quickly, so every call
waits at least MIN_INTERVAL seconds after the previous one.
"""
from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)

MIN_INTERVAL = 3.0  # seconds between requests
RETRIES = 3
TIMEOUT = 30

_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (tw-stock-screener; personal research script)",
    "Accept": "application/json",
})
_last_call = 0.0


def get_json(url: str, params: dict | None = None):
    global _last_call
    error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            resp = _session.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            text = resp.text.strip()
            if not text:
                return None  # some endpoints answer an empty body on holidays
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            error = exc
            log.warning("第 %d 次請求失敗 %s：%s", attempt, url, exc)
            time.sleep(MIN_INTERVAL * attempt * 2)
    raise RuntimeError(f"連續 {RETRIES} 次請求失敗：{url}（{error}）")
