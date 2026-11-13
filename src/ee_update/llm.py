"""Minimal Gemini client for the real-LLM replay table (docs/06_replay_table_spec.md).

* The key is read from the environment and never written anywhere: GEMINI_API_KEY, or a
  comma-separated GEMINI_API_KEYS to spread calls over several projects' quotas (each key
  gets its own rate limiter; calls go to whichever key is free first).
* Every response is cached on disk, keyed by hash(model, prompt, decoding settings), so a
  rebuild never pays for the same call twice and the table can be resumed.
* Token counts are the service's own `usageMetadata` (prompt, output and, if a model
  thinks, thought tokens); nothing is estimated.
* Retries with exponential backoff on 429 and 5xx; any other error is raised.
* There is no offline fallback that invents answers: without a key, a call that is not
  in the cache raises.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


@dataclass(frozen=True)
class Reply:
    text: str
    prompt_tokens: int
    output_tokens: int        # completion + thought tokens: everything billed as output
    model: str
    response_hash: str
    cached: bool

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


class KeyPool:
    """API keys, each limited to `rpm` request starts per minute; hands out the key that is
    free soonest and sleeps until then.  Shared by all threads."""

    def __init__(self, keys: list, rpm: float):
        self.keys = keys
        self.interval = 60.0 / rpm
        self._next = [0.0] * len(keys)
        self._lock = threading.Lock()

    def take(self) -> str:
        with self._lock:
            now = time.monotonic()
            i = min(range(len(self.keys)), key=lambda j: self._next[j])
            start = max(now, self._next[i])
            self._next[i] = start + self.interval
        if start > now:
            time.sleep(start - now)
        return self.keys[i]


class Gemini:
    def __init__(self, model: str, cache_dir: str | Path, rpm: float = 600.0,
                 max_output_tokens: int = 256, temperature: float = 0.0):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        keys = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()]
        if not keys and os.environ.get("GEMINI_API_KEY"):
            keys = [os.environ["GEMINI_API_KEY"]]
        self.pool = KeyPool(keys, rpm) if keys else None
        self.gen = {"temperature": temperature, "maxOutputTokens": max_output_tokens}
        self._http = httpx.Client(timeout=90.0)
        self.live_calls = 0
        self.cached_calls = 0

    def _path(self, prompt: str) -> Path:
        blob = json.dumps({"model": self.model, "prompt": prompt, "gen": self.gen}, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(blob.encode()).hexdigest()[:32] + ".json")

    def ask(self, prompt: str, retries: int = 8) -> Reply:
        p = self._path(prompt)
        if p.exists():
            d = json.loads(p.read_text())
            self.cached_calls += 1
            return Reply(d["text"], d["prompt_tokens"], d["output_tokens"], d["model"],
                         d["response_hash"], True)
        if self.pool is None:
            raise RuntimeError("GEMINI_API_KEY(S) not set and this call is not cached")
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": self.gen}
        err = ""
        for attempt in range(retries):
            key = self.pool.take()
            try:
                r = self._http.post(API.format(model=self.model), headers={"x-goog-api-key": key}, json=body)
            except httpx.HTTPError as e:
                err = repr(e)
            else:
                if r.status_code == 200:
                    d = r.json()
                    cand = (d.get("candidates") or [{}])[0]
                    parts = (cand.get("content") or {}).get("parts") or []
                    text = "".join(x.get("text", "") for x in parts if not x.get("thought"))
                    u = d.get("usageMetadata") or {}
                    out = dict(text=text, prompt_tokens=int(u.get("promptTokenCount", 0)),
                               output_tokens=int(u.get("candidatesTokenCount", 0)) + int(u.get("thoughtsTokenCount", 0)),
                               model=d.get("modelVersion", self.model),
                               response_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
                               finish_reason=cand.get("finishReason", ""))
                    tmp = p.with_suffix(".tmp")
                    tmp.write_text(json.dumps(out))
                    tmp.replace(p)
                    self.live_calls += 1
                    return Reply(out["text"], out["prompt_tokens"], out["output_tokens"], out["model"],
                                 out["response_hash"], False)
                err = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(err)
            wait = min(60.0, 2 ** attempt) * (0.5 + random.random())
            print(f"[llm] retry {attempt + 1}/{retries} in {wait:.1f}s: {err[:160]}", file=sys.stderr, flush=True)
            time.sleep(wait)
        raise RuntimeError(f"gave up after {retries} attempts: {err}")
