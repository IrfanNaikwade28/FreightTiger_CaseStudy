"""Optional LLM rewrites of justified-anomaly explanations (Groq only).

The LLM never decides flags, evidence validity, note ids, or numbers.
It is invoked only for route-weeks that the deterministic pipeline has
already marked as detector-flagged AND evidence-justified, and its
output may only replace the free-text ``reason`` cell of those rows.
Unexplained anomalies always keep the deterministic "unexplained"
explanation. Any failure (HTTP, JSON, empty or ungrounded text) falls
back to the deterministic explanation without affecting verdicts.

Provider: **Groq** only, via its OpenAI-compatible chat-completions
endpoint (stdlib ``urllib.request``; no third-party SDK). Configuration
via environment (values already set in the process environment always
win):

* ``FREIGHTTIGER_LLM_API_KEY``  - Groq API key (required for online
  mode; never hardcoded; e.g. from https://console.groq.com/keys).
* ``FREIGHTTIGER_LLM_BASE_URL`` - endpoint base (default
  ``https://api.groq.com/openai/v1``).
* ``FREIGHTTIGER_LLM_MODEL``    - Groq model id (default
  ``qwen/qwen3.8-27b``; see https://console.groq.com/docs/models).

Python does not load ``.env`` files automatically, so
``build_explainer`` first calls ``load_dotenv``: a stdlib-only loader
that reads ``.env`` from the project root (resolved from this file's
location, not the current working directory) and fills only *missing*
environment variables. Secrets are never printed or logged.

Generation is deterministic where supported (temperature 0). Calls are
capped by ``max_calls`` (default ``MAX_LLM_CALLS``). Actual usage
(prompt/completion tokens from Groq's ``usage`` metadata) is
instrumented per instance; estimated cost uses the published-price
table below and is reported as ``None`` for unknown models rather than
fabricated.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from cost_analysis import RouteWeekMetrics
from note_retriever import EvidenceVerdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MAX_CALLS = 20
MAX_REASON_CHARS = 600
TEMPERATURE = 0.0
MAX_TOKENS = 200
REQUEST_TIMEOUT_S = 30.0

# Published list prices, USD per 1M tokens (input, output), as
# documented on console.groq.com/docs/models for production models.
# Verify against Groq's current pricing before relying on estimates;
# unknown models report cost=None.
PRICING_USD_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "qwen/qwen3.8-27b": (0.80, 4.00),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}

SYSTEM_PROMPT = (
    "You rewrite shipping-cost anomaly explanations for an audit report. "
    "Use ONLY the verified evidence provided; never invent causes, dates, "
    "numbers, or notes. Cite the note id in your text. One or two "
    "sentences, plain English, no line breaks."
)


@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration for one explainer instance."""

    enabled: bool
    api_key: str | None
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    timeout_s: float = REQUEST_TIMEOUT_S
    max_calls: int = DEFAULT_MAX_CALLS


@dataclass
class LLMErrorRecord:
    """One sanitized failure (never contains the API key)."""

    call_index: int
    kind: str  # "http" | "network" | "parse" | "sanitize" | "unknown"
    http_status: int | None = None
    message: str = ""
    ray_id: str | None = None  # Cloudflare Ray ID when present (non-secret)


@dataclass
class LLMUsage:
    """Measured usage for one run (no fabricated numbers)."""

    mode: str = "offline"
    model: str | None = None
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    fallbacks: int = 0
    usage_metadata_missing: int = 0
    estimated_cost_usd: float | None = None
    error_records: list[LLMErrorRecord] = field(default_factory=list)


Transport = Callable[..., dict]


def _is_env_name(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


def load_dotenv(
    path: Path | str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into ``environ``.

    Resolves ``path`` relative to this module's project root when
    omitted, so the file is found regardless of the current working
    directory. Keys already present in ``environ`` are never
    overwritten (shell/environment values win). Blank lines, ``#``
    comments, an optional ``export`` prefix, and single/double-quoted
    values are supported. A missing or unreadable file is a no-op.
    Returns only the key names (and values) that were applied — callers
    must not print or log those values when they may be secrets.
    """
    target = os.environ if environ is None else environ
    env_path = DOTENV_PATH if path is None else Path(path)
    applied: dict[str, str] = {}
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return applied
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not _is_env_name(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key in target:
            continue
        target[key] = value
        applied[key] = value
    return applied


def build_explainer(
    llm_enabled: bool,
    max_calls: int = DEFAULT_MAX_CALLS,
    dotenv_path: Path | str | None = None,
) -> "LLMExplainer":
    """Create an explainer from CLI intent, environment, and ``.env``.

    ``.env`` in the project root is loaded first (missing keys only;
    never overwrites an already-set environment variable). Pass
    ``dotenv_path`` to read a different file (or a missing path to skip
    loading). Online mode requires both ``llm_enabled`` and
    ``FREIGHTTIGER_LLM_API_KEY``; otherwise the returned explainer is
    offline (zero calls, deterministic output).
    """
    load_dotenv(dotenv_path)
    api_key = os.environ.get("FREIGHTTIGER_LLM_API_KEY")
    model = os.environ.get("FREIGHTTIGER_LLM_MODEL", DEFAULT_MODEL)
    base_url = os.environ.get("FREIGHTTIGER_LLM_BASE_URL", DEFAULT_BASE_URL)
    config = LLMConfig(
        enabled=bool(llm_enabled and api_key),
        api_key=api_key,
        model=model,
        base_url=base_url.rstrip("/"),
        max_calls=max(0, max_calls),
    )
    return LLMExplainer(config)


class LLMExplainer:
    """Bounded, instrumented, evidence-grounded explanation rewriter."""

    def __init__(self, config: LLMConfig, transport: Transport | None = None):
        self.config = config
        self._transport = transport or _http_transport
        self.usage = LLMUsage(
            mode="online" if config.enabled else "offline",
            model=config.model if config.enabled else None,
        )

    def explain_justified(
        self,
        metric: RouteWeekMetrics,
        anomaly_flagged: bool,
        verdict: EvidenceVerdict,
    ) -> str | None:
        """Return a grounded plain-English reason, or ``None`` to fall back.

        Calls the model only when online mode is active, the row is a
        detector-flagged justified anomaly with a matched note, and the
        call budget is not exhausted. Any failure (HTTP, network, bad
        JSON, ungrounded text) returns ``None`` so the caller keeps the
        deterministic explanation; verdicts and numbers are never
        affected. Failures increment both ``errors`` and ``fallbacks``
        (deterministic reason is used) and append a sanitized
        ``LLMErrorRecord``.
        """
        if not self.config.enabled:
            return None
        if not anomaly_flagged or not verdict.justified:
            return None
        if not verdict.matched_note_id:
            return None
        if self.usage.calls >= self.config.max_calls:
            return None
        self.usage.calls += 1
        call_index = self.usage.calls
        user_prompt = _build_prompt(metric, verdict)
        try:
            data = self._transport(
                base_url=self.config.base_url,
                api_key=self.config.api_key or "",
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout_s=self.config.timeout_s,
                system=SYSTEM_PROMPT,
                user=user_prompt,
            )
            content, prompt_tokens, completion_tokens = _parse_response(data)
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            body = body[:MAX_ERROR_BODY_CHARS]
            ray_id = None
            try:
                ray_id = exc.headers.get("cf-ray") if exc.headers else None
            except Exception:
                ray_id = None
            self._record_failure(
                call_index,
                kind="http",
                http_status=int(exc.code),
                message=body or str(exc.reason or exc),
                ray_id=ray_id,
            )
            return None
        except urllib.error.URLError as exc:
            self._record_failure(
                call_index,
                kind="network",
                http_status=None,
                message=str(exc.reason if hasattr(exc, "reason") else exc),
            )
            return None
        except Exception as exc:
            self._record_failure(
                call_index,
                kind="parse",
                http_status=None,
                message=f"{type(exc).__name__}: {exc}",
            )
            return None
        if prompt_tokens is None or completion_tokens is None:
            # Response arrived but usage metadata was absent; do not
            # invent token counts.
            self.usage.usage_metadata_missing += 1
        else:
            self.usage.prompt_tokens += prompt_tokens
            self.usage.completion_tokens += completion_tokens
        text = _sanitize(content, verdict.matched_note_id)
        if text is None:
            self._record_failure(
                call_index,
                kind="sanitize",
                http_status=None,
                message="response empty, overlong, or missing matched note id",
            )
            return None
        self._update_cost()
        return text

    def _record_failure(
        self,
        call_index: int,
        *,
        kind: str,
        http_status: int | None,
        message: str,
        ray_id: str | None = None,
    ) -> None:
        """Count a failure and keep a redacted diagnostic record."""
        self.usage.errors += 1
        self.usage.fallbacks += 1  # deterministic reason is used
        self.usage.error_records.append(
            LLMErrorRecord(
                call_index=call_index,
                kind=kind,
                http_status=http_status,
                message=_safe_error_text(message, self.config.api_key),
                ray_id=ray_id,
            )
        )

    def _update_cost(self) -> None:
        rates = PRICING_USD_PER_1M_TOKENS.get(self.config.model)
        if rates is None:
            self.usage.estimated_cost_usd = None
            return
        input_rate, output_rate = rates
        self.usage.estimated_cost_usd = (
            self.usage.prompt_tokens / 1_000_000 * input_rate
            + self.usage.completion_tokens / 1_000_000 * output_rate
        )


def _build_prompt(metric: RouteWeekMetrics, verdict: EvidenceVerdict) -> str:
    own = (
        f"{metric.pct_vs_own_history:+.1f}%"
        if metric.pct_vs_own_history is not None
        else "unavailable"
    )
    peer = (
        f"{metric.pct_vs_similar_routes:+.1f}%"
        if metric.pct_vs_similar_routes is not None
        else "unavailable"
    )
    return (
        f"Route: {metric.route}\n"
        f"Week of: {metric.week_of.isoformat()}\n"
        f"Cost vs own history: {own}\n"
        f"Cost vs similar routes: {peer}\n"
        f"Verified evidence (deterministic pipeline): {verdict.explanation}\n"
        "Rewrite this evidence as a short plain-English explanation, "
        f"citing note {verdict.matched_note_id}."
    )


def _parse_response(data: dict) -> tuple[str, int | None, int | None]:
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("model content is not a string")
    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return (
        content,
        int(prompt_tokens) if prompt_tokens is not None else None,
        int(completion_tokens) if completion_tokens is not None else None,
    )


def _sanitize(content: str, note_id: str) -> str | None:
    """Return single-line text citing ``note_id``, else ``None``."""
    text = " ".join(content.split())
    if not text or len(text) > MAX_REASON_CHARS:
        return None
    if note_id not in text:
        return None
    return text


_SECRET_PATTERNS = (
    re.compile(r"gsk_[A-Za-z0-9]+"),
    re.compile(r"(?i)bearer\s+\S+"),
    re.compile(r"(?i)api[_-]?key['\"]?\s*[:=]\s*['\"]?[^'\"\s,}]+"),
)


def _safe_error_text(message: str, api_key: str | None = None, limit: int = 240) -> str:
    """Redact secrets from an error message and truncate it."""
    text = message if isinstance(message, str) else str(message)
    if api_key:
        text = text.replace(api_key, "[redacted]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[redacted]", text)
    text = " ".join(text.split())
    return text[:limit]


# Explicit User-Agent: urllib's default "Python-urllib/3.x" is blocked by
# Cloudflare Browser Integrity Check (HTTP 403, error code 1010) in front
# of api.groq.com. A stable client identity is required; this is not an
# auth bypass and does not affect TLS or routing.
USER_AGENT = "FreightTiger-CostMonitor/1.0 (+stdlib urllib)"
MAX_ERROR_BODY_CHARS = 400


def _http_transport(
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    system: str,
    user: str,
) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))
