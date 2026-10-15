"""Tests for the optional LLM explanation layer (no network; stub transport).

The LLM may only rewrite the ``reason`` text of detector-flagged,
evidence-justified rows. Flags, numbers, and matched note ids must stay
identical to the offline pipeline, and all failures must fall back to
the deterministic explanation.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_detection import AnomalyResult  # noqa: E402
from cost_analysis import RouteWeekMetrics  # noqa: E402
from llm_explainer import (  # noqa: E402
    LLMConfig,
    LLMExplainer,
    MAX_REASON_CHARS,
    build_explainer,
)
from note_retriever import EvidenceVerdict  # noqa: E402
from report_generator import build_rows  # noqa: E402

WEEK = dt.date(2025, 1, 20)


def _metric(route: str = "Ahmedabad-Mumbai") -> RouteWeekMetrics:
    origin, destination = route.split("-")[0], route.split("-")[1]
    return RouteWeekMetrics(
        origin=origin,
        destination=destination,
        route_type="Medium",
        week_of=WEEK,
        cost_per_tonne_km=3.29,
        pct_vs_own_history=29.5,
        pct_vs_similar_routes=22.5,
        prior_weeks_used=8,
        peer_routes_used=2,
        n_shipments=4,
    )


def _flagged(metric: RouteWeekMetrics) -> AnomalyResult:
    return AnomalyResult(
        metrics=metric,
        flagged=True,
        triggers=("own_history", "similar_routes"),
        own_history_pct=29.5,
        similar_routes_pct=22.5,
    )


def _justified_verdict(route: str = "Ahmedabad-Mumbai") -> EvidenceVerdict:
    return EvidenceVerdict(
        route=route,
        week_of=WEEK,
        justified=True,
        matched_note_id="N002",
        explanation=(
            "Matches note N002 dated 2025-01-20 for Ahmedabad-Mumbai, "
            "effective 2025-01-20 to 2025-01-26: A regional festival week "
            "saw a temporary surcharge. The cost rise is explained."
        ),
        applicable_note_ids=("N002",),
        supporting_note_ids=("N002",),
        rejected_note_ids=(),
    )


def _unexplained_verdict() -> EvidenceVerdict:
    return EvidenceVerdict(
        route="Mumbai-Pune",
        week_of=WEEK,
        justified=False,
        matched_note_id=None,
        explanation="No context note found; cost rise is unexplained.",
        applicable_note_ids=(),
        supporting_note_ids=(),
        rejected_note_ids=(),
    )


def _ok_transport(content: str = "Note N002 attributes the rise to a festival surcharge.",
                  usage: dict | None = None) -> tuple[callable, list]:
    calls: list[dict] = []

    def transport(**kwargs):
        calls.append(kwargs)
        return {
            "model": kwargs["model"],
            "choices": [{"message": {"content": content}}],
            "usage": usage
            if usage is not None
            else {"prompt_tokens": 120, "completion_tokens": 35},
        }

    return transport, calls


def _config(**overrides) -> LLMConfig:
    base = dict(enabled=True, api_key="test-key", model="qwen/qwen3.8-27b", max_calls=20)
    base.update(overrides)
    return LLMConfig(**base)


class TestSupportedNoteExplanation(unittest.TestCase):
    def test_llm_reason_applies_only_to_flagged_justified_row(self) -> None:
        metric = _metric()
        anomaly = _flagged(metric)
        verdict = _justified_verdict()
        transport, calls = _ok_transport()
        explainer = LLMExplainer(_config(), transport=transport)
        text = explainer.explain_justified(metric, True, verdict)
        self.assertIsNotNone(text)
        self.assertIn("N002", text)
        self.assertEqual(len(calls), 1)
        # minimum data only: no shipment dataset in the prompt
        self.assertNotIn("SHP", calls[0]["user"])
        self.assertIn("N002", calls[0]["user"])
        self.assertEqual(calls[0]["temperature"], 0.0)
        rows = build_rows(
            [metric], [anomaly], [verdict],
            reason_overrides={(metric.route, WEEK): text},
        )
        row = rows[0]
        self.assertEqual(row.reason, text)
        self.assertEqual(row.flagged, "No (justified)")
        self.assertEqual(row.matched_note_id, "N002")
        self.assertEqual(row.cost_per_tonne_km, "3.29")
        self.assertEqual(row.vs_own_history, "+29.5% vs this route's past average")

    def test_usage_and_cost_from_provider_metadata(self) -> None:
        transport, _ = _ok_transport(
            usage={"prompt_tokens": 1000, "completion_tokens": 400}
        )
        explainer = LLMExplainer(_config(), transport=transport)
        explainer.explain_justified(_metric(), True, _justified_verdict())
        self.assertEqual(explainer.usage.calls, 1)
        self.assertEqual(explainer.usage.prompt_tokens, 1000)
        self.assertEqual(explainer.usage.completion_tokens, 400)
        # qwen/qwen3.8-27b published rates: $0.80 in / $4.00 out per 1M
        self.assertAlmostEqual(
            explainer.usage.estimated_cost_usd, 1000 / 1e6 * 0.80 + 400 / 1e6 * 4.00
        )

    def test_missing_usage_metadata_reported_not_fabricated(self) -> None:
        transport, _ = _ok_transport(usage=None)
        # transport returns dict without "usage" key path: override
        def bare(**kwargs):
            return {"choices": [{"message": {"content": "Note N002: festival."}}]}
        explainer = LLMExplainer(_config(), transport=bare)
        text = explainer.explain_justified(_metric(), True, _justified_verdict())
        self.assertIsNotNone(text)
        self.assertEqual(explainer.usage.usage_metadata_missing, 1)
        self.assertEqual(explainer.usage.prompt_tokens, 0)
        self.assertEqual(explainer.usage.completion_tokens, 0)


class TestMissingEvidence(unittest.TestCase):
    def test_unexplained_rows_never_call_the_llm(self) -> None:
        metric = _metric("Mumbai-Pune")
        anomaly = _flagged(metric)
        verdict = _unexplained_verdict()
        _, calls = _ok_transport()
        explainer = LLMExplainer(_config(), transport=lambda **kw: calls.append(kw) or {})
        text = explainer.explain_justified(metric, True, verdict)
        self.assertIsNone(text)
        self.assertEqual(len(calls), 0)
        self.assertEqual(explainer.usage.calls, 0)
        rows = build_rows([metric], [anomaly], [verdict])
        self.assertEqual(rows[0].reason, verdict.explanation)
        self.assertIn("unexplained", rows[0].reason)
        self.assertEqual(rows[0].matched_note_id, "")

    def test_non_flagged_rows_never_call_the_llm(self) -> None:
        metric = _metric()
        anomaly = AnomalyResult(
            metrics=metric, flagged=False, triggers=(),
            own_history_pct=29.5, similar_routes_pct=22.5,
        )
        _, calls = _ok_transport()
        explainer = LLMExplainer(_config(), transport=lambda **kw: calls.append(kw) or {})
        text = explainer.explain_justified(metric, False, _justified_verdict())
        self.assertIsNone(text)
        self.assertEqual(len(calls), 0)


class TestFailuresFallBack(unittest.TestCase):
    def _fallback_text(self, transport) -> str | None:
        explainer = LLMExplainer(_config(), transport=transport)
        return explainer.explain_justified(_metric(), True, _justified_verdict())

    def test_empty_response_falls_back(self) -> None:
        transport, _ = _ok_transport(content="   ")
        explainer = LLMExplainer(_config(), transport=transport)
        self.assertIsNone(explainer.explain_justified(_metric(), True, _justified_verdict()))
        self.assertEqual(explainer.usage.fallbacks, 1)

    def test_response_missing_note_id_falls_back(self) -> None:
        transport, _ = _ok_transport(content="Transporters raised rates for unclear reasons.")
        explainer = LLMExplainer(_config(), transport=transport)
        self.assertIsNone(explainer.explain_justified(_metric(), True, _justified_verdict()))
        self.assertEqual(explainer.usage.fallbacks, 1)

    def test_overlong_response_falls_back(self) -> None:
        transport, _ = _ok_transport(content="N002 " + "x" * (MAX_REASON_CHARS + 10))
        explainer = LLMExplainer(_config(), transport=transport)
        self.assertIsNone(explainer.explain_justified(_metric(), True, _justified_verdict()))

    def test_transport_exception_falls_back_and_counts_error(self) -> None:
        def boom(**kwargs):
            raise RuntimeError("network down")
        explainer = LLMExplainer(_config(), transport=boom)
        self.assertIsNone(explainer.explain_justified(_metric(), True, _justified_verdict()))
        self.assertEqual(explainer.usage.errors, 1)
        self.assertEqual(explainer.usage.fallbacks, 1)
        self.assertEqual(explainer.usage.calls, 1)
        # deterministic reason still applies for the row
        rows = build_rows([_metric()], [_flagged(_metric())], [_justified_verdict()])
        self.assertIn("N002", rows[0].reason)

    def test_malformed_json_structure_falls_back(self) -> None:
        def malformed(**kwargs):
            return {"unexpected": "shape"}
        explainer = LLMExplainer(_config(), transport=malformed)
        self.assertIsNone(explainer.explain_justified(_metric(), True, _justified_verdict()))
        self.assertEqual(explainer.usage.errors, 1)

    def test_failure_leaves_all_non_reason_fields_intact(self) -> None:
        metric = _metric()
        anomaly = _flagged(metric)
        verdict = _justified_verdict()
        offline = build_rows([metric], [anomaly], [verdict])[0]
        explainer = LLMExplainer(_config(), transport=lambda **kw: {})
        text = explainer.explain_justified(metric, True, verdict)
        self.assertIsNone(text)
        online = build_rows([metric], [anomaly], [verdict])[0]
        for field in (
            "route", "week_of", "cost_per_tonne_km", "vs_own_history",
            "vs_similar_routes", "flagged", "matched_note_id",
        ):
            self.assertEqual(getattr(offline, field), getattr(online, field))


def _http_error(status: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.groq.com/openai/v1/chat/completions",
        code=status,
        msg="error",
        hdrs=None,
        fp=io.BytesIO(body.encode("utf-8")),
    )


class TestHttpErrorDiagnostics(unittest.TestCase):
    """Mocked HTTP failures: status recorded, secrets redacted, fallback used."""

    def _run(self, status: int, body: str, api_key: str = "gsk_secretkey123"):
        def transport(**kwargs):
            raise _http_error(status, body)

        explainer = LLMExplainer(_config(api_key=api_key), transport=transport)
        text = explainer.explain_justified(_metric(), True, _justified_verdict())
        return explainer, text

    def test_auth_error_401(self) -> None:
        explainer, text = self._run(
            401, '{"error":{"message":"Invalid API Key","type":"invalid_request_error"}}'
        )
        self.assertIsNone(text)
        self.assertEqual(explainer.usage.errors, 1)
        self.assertEqual(explainer.usage.fallbacks, 1)
        self.assertEqual(explainer.usage.prompt_tokens, 0)
        self.assertEqual(explainer.usage.completion_tokens, 0)
        self.assertIsNone(explainer.usage.estimated_cost_usd)
        record = explainer.usage.error_records[0]
        self.assertEqual(record.http_status, 401)
        self.assertEqual(record.kind, "http")
        self.assertIn("Invalid API Key", record.message)

    def test_unsupported_model_error_404(self) -> None:
        explainer, text = self._run(
            404, '{"error":{"message":"model_not_found","type":"invalid_request_error"}}'
        )
        self.assertIsNone(text)
        self.assertEqual(explainer.usage.error_records[0].http_status, 404)
        self.assertIn("model_not_found", explainer.usage.error_records[0].message)

    def test_rate_limit_error_429(self) -> None:
        explainer, text = self._run(
            429, '{"error":{"message":"Rate limit reached","type":"rate_limit_exceeded"}}'
        )
        self.assertIsNone(text)
        self.assertEqual(explainer.usage.error_records[0].http_status, 429)
        self.assertIn("Rate limit", explainer.usage.error_records[0].message)

    def test_error_message_never_contains_api_key(self) -> None:
        secret = "gsk_supersecretvalue999"
        explainer, _ = self._run(
            401,
            f'{{"error":"Invalid API Key Bearer {secret}"}}',
            api_key=secret,
        )
        record = explainer.usage.error_records[0]
        self.assertNotIn(secret, record.message)
        self.assertNotIn(secret, json.dumps(explainer.usage.error_records[0].__dict__))
        self.assertIn("[redacted]", record.message)

    def test_network_error_is_recorded(self) -> None:
        def transport(**kwargs):
            raise urllib.error.URLError("connection refused")

        explainer = LLMExplainer(_config(), transport=transport)
        text = explainer.explain_justified(_metric(), True, _justified_verdict())
        self.assertIsNone(text)
        record = explainer.usage.error_records[0]
        self.assertEqual(record.kind, "network")
        self.assertIsNone(record.http_status)
        self.assertIn("connection refused", record.message)

    def test_failed_calls_use_deterministic_reason_not_llm(self) -> None:
        metric = _metric()
        anomaly = _flagged(metric)
        verdict = _justified_verdict()
        offline_row = build_rows([metric], [anomaly], [verdict])[0]
        explainer, text = self._run(500, "internal error")
        self.assertIsNone(text)
        online_row = build_rows([metric], [anomaly], [verdict])[0]
        self.assertEqual(online_row.reason, offline_row.reason)
        self.assertEqual(online_row.flagged, offline_row.flagged)
        self.assertEqual(online_row.matched_note_id, offline_row.matched_note_id)
        self.assertEqual(online_row.cost_per_tonne_km, offline_row.cost_per_tonne_km)

    def test_all_failed_calls_do_not_fabricate_tokens_or_cost(self) -> None:
        explainer, _ = self._run(401, "invalid")
        self.assertEqual(explainer.usage.calls, 1)
        self.assertEqual(explainer.usage.prompt_tokens, 0)
        self.assertEqual(explainer.usage.completion_tokens, 0)
        self.assertEqual(explainer.usage.usage_metadata_missing, 0)
        self.assertIsNone(explainer.usage.estimated_cost_usd)

    def test_success_records_measured_tokens(self) -> None:
        transport, _ = _ok_transport(
            content="Note N002 explains the temporary festival surcharge.",
            usage={"prompt_tokens": 150, "completion_tokens": 40},
        )
        explainer = LLMExplainer(_config(), transport=transport)
        text = explainer.explain_justified(_metric(), True, _justified_verdict())
        self.assertIsNotNone(text)
        self.assertEqual(explainer.usage.errors, 0)
        self.assertEqual(explainer.usage.fallbacks, 0)
        self.assertEqual(explainer.usage.prompt_tokens, 150)
        self.assertEqual(explainer.usage.completion_tokens, 40)
        self.assertIsNotNone(explainer.usage.estimated_cost_usd)

    def test_safe_error_text_redacts_bearer_and_gsk(self) -> None:
        from llm_explainer import _safe_error_text

        raw = "Authorization: Bearer gsk_abc123DEF; api_key=gsk_abc123DEF"
        cleaned = _safe_error_text(raw, api_key=None)
        self.assertNotIn("gsk_abc123DEF", cleaned)
        self.assertIn("[redacted]", cleaned)

    def test_cloudflare_1010_body_and_ray_id_recorded(self) -> None:
        import email.message

        headers = email.message.Message()
        headers["cf-ray"] = "abc123-LHR"
        err = urllib.error.HTTPError(
            url="https://api.groq.com/openai/v1/chat/completions",
            code=403,
            msg="Forbidden",
            hdrs=headers,
            fp=io.BytesIO(b"error code: 1010"),
        )

        def transport(**kwargs):
            raise err

        explainer = LLMExplainer(_config(), transport=transport)
        text = explainer.explain_justified(_metric(), True, _justified_verdict())
        self.assertIsNone(text)
        record = explainer.usage.error_records[0]
        self.assertEqual(record.http_status, 403)
        self.assertIn("1010", record.message)
        self.assertEqual(record.ray_id, "abc123-LHR")
        self.assertEqual(explainer.usage.fallbacks, 1)

    def test_http_transport_sets_explicit_user_agent(self) -> None:
        from llm_explainer import USER_AGENT, _http_transport

        captured = {}

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"N002 ok"}}],"usage":{}}'

        def fake_urlopen(request, timeout=None):
            captured["user_agent"] = request.get_header("User-agent")
            captured["url"] = request.full_url
            return _FakeResponse()

        import urllib.request as ur

        original = ur.urlopen
        ur.urlopen = fake_urlopen
        try:
            _http_transport(
                base_url="https://api.groq.com/openai/v1",
                api_key="k",
                model="m",
                temperature=0.0,
                max_tokens=10,
                timeout_s=1.0,
                system="s",
                user="u",
            )
        finally:
            ur.urlopen = original
        self.assertEqual(captured["user_agent"], USER_AGENT)
        self.assertNotIn("Python-urllib", USER_AGENT)
        self.assertEqual(
            captured["url"], "https://api.groq.com/openai/v1/chat/completions"
        )
        self.assertEqual(captured["url"].count("/chat/completions"), 1)


class TestOfflineMode(unittest.TestCase):
    def test_offline_explainer_makes_zero_calls(self) -> None:
        explainer = build_explainer(llm_enabled=False)
        self.assertFalse(explainer.config.enabled)
        self.assertEqual(explainer.usage.mode, "offline")
        transport, calls = _ok_transport()
        explainer2 = LLMExplainer(
            LLMConfig(enabled=False, api_key=None), transport=transport
        )
        self.assertIsNone(explainer2.explain_justified(_metric(), True, _justified_verdict()))
        self.assertEqual(len(calls), 0)
        self.assertEqual(explainer2.usage.calls, 0)

    def test_online_requires_key(self) -> None:
        import os

        old_key = os.environ.pop("FREIGHTTIGER_LLM_API_KEY", None)
        try:
            # Missing dotenv path so the project .env cannot supply a
            # key for this negative case.
            explainer = build_explainer(
                llm_enabled=True, dotenv_path="/nonexistent/freighttiger/.env"
            )
            self.assertFalse(explainer.config.enabled)
            self.assertEqual(explainer.usage.mode, "offline")
        finally:
            if old_key is not None:
                os.environ["FREIGHTTIGER_LLM_API_KEY"] = old_key

    def test_defaults_point_at_groq(self) -> None:
        from llm_explainer import DEFAULT_BASE_URL, DEFAULT_MODEL

        self.assertEqual(DEFAULT_BASE_URL, "https://api.groq.com/openai/v1")
        self.assertEqual(DEFAULT_MODEL, "qwen/qwen3.8-27b")
        self.assertNotIn("openai.com", DEFAULT_BASE_URL)
        self.assertNotIn("gpt-4o-mini", DEFAULT_MODEL)


class TestBoundedCalls(unittest.TestCase):
    def test_call_count_capped_at_max_calls(self) -> None:
        metrics = [_metric("Ahmedabad-Mumbai"), _metric("Chennai-Bangalore")]
        anomalies = [_flagged(m) for m in metrics]
        verdicts = [_justified_verdict(m.route) for m in metrics]
        transport, calls = _ok_transport()
        explainer = LLMExplainer(_config(max_calls=1), transport=transport)
        overrides = {}
        for m, a, v in zip(metrics, anomalies, verdicts):
            text = explainer.explain_justified(m, a.flagged, v)
            if text is not None:
                overrides[(m.route, WEEK)] = text
        self.assertEqual(len(calls), 1)
        self.assertEqual(explainer.usage.calls, 1)
        self.assertEqual(len(overrides), 1)

    def test_negative_max_calls_makes_no_calls(self) -> None:
        transport, calls = _ok_transport()
        explainer = LLMExplainer(_config(max_calls=0), transport=transport)
        self.assertIsNone(explainer.explain_justified(_metric(), True, _justified_verdict()))
        self.assertEqual(len(calls), 0)


class TestNonReasonFieldsIdentical(unittest.TestCase):
    def test_llm_and_offline_rows_identical_except_reason(self) -> None:
        metric = _metric()
        anomaly = _flagged(metric)
        verdict = _justified_verdict()
        llm_text = "Note N002 explains the spike as a temporary festival surcharge."
        offline = build_rows([metric], [anomaly], [verdict])[0]
        online = build_rows(
            [metric], [anomaly], [verdict],
            reason_overrides={(metric.route, WEEK): llm_text},
        )[0]
        self.assertEqual(online.reason, llm_text)
        self.assertNotEqual(offline.reason, llm_text)
        for field in (
            "route", "week_of", "cost_per_tonne_km", "vs_own_history",
            "vs_similar_routes", "flagged", "matched_note_id",
        ):
            self.assertEqual(getattr(offline, field), getattr(online, field))


class TestDotEnvLoading(unittest.TestCase):
    def _write_env(self, directory, content: str) -> Path:
        path = Path(directory) / ".env"
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_keys_from_file(self) -> None:
        import tempfile

        from llm_explainer import load_dotenv

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_env(
                tmp,
                "FREIGHTTIGER_LLM_API_KEY=secret-from-file\n"
                "FREIGHTTIGER_LLM_MODEL=file-model\n",
            )
            environ: dict[str, str] = {}
            applied = load_dotenv(path=path, environ=environ)
            self.assertEqual(environ["FREIGHTTIGER_LLM_API_KEY"], "secret-from-file")
            self.assertEqual(environ["FREIGHTTIGER_LLM_MODEL"], "file-model")
            self.assertIn("FREIGHTTIGER_LLM_API_KEY", applied)

    def test_missing_file_is_noop(self) -> None:
        from llm_explainer import load_dotenv

        environ: dict[str, str] = {}
        applied = load_dotenv(path="/nonexistent/freighttiger/.env", environ=environ)
        self.assertEqual(applied, {})
        self.assertEqual(environ, {})

    def test_does_not_overwrite_existing_environment(self) -> None:
        import tempfile

        from llm_explainer import load_dotenv

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_env(
                tmp, "FREIGHTTIGER_LLM_API_KEY=from-file\nFREIGHTTIGER_LLM_MODEL=m\n"
            )
            environ = {"FREIGHTTIGER_LLM_API_KEY": "already-set"}
            load_dotenv(path=path, environ=environ)
            self.assertEqual(environ["FREIGHTTIGER_LLM_API_KEY"], "already-set")
            self.assertEqual(environ["FREIGHTTIGER_LLM_MODEL"], "m")

    def test_ignores_comments_blanks_export_and_quotes(self) -> None:
        import tempfile

        from llm_explainer import load_dotenv

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_env(
                tmp,
                "# comment line\n"
                "\n"
                "export FREIGHTTIGER_LLM_MODEL='quoted-model'\n"
                'FREIGHTTIGER_LLM_BASE_URL="https://api.groq.com/openai/v1"\n'
                "NOT_AN_ASSIGNMENT\n"
                "1BADKEY=x\n",
            )
            environ: dict[str, str] = {}
            load_dotenv(path=path, environ=environ)
            self.assertEqual(environ["FREIGHTTIGER_LLM_MODEL"], "quoted-model")
            self.assertEqual(
                environ["FREIGHTTIGER_LLM_BASE_URL"], "https://api.groq.com/openai/v1"
            )
            self.assertNotIn("NOT_AN_ASSIGNMENT", environ)
            self.assertNotIn("1BADKEY", environ)

    def test_build_explainer_reads_key_from_dotenv_file(self) -> None:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_env(tmp, "FREIGHTTIGER_LLM_API_KEY=gsk_test_from_dotenv\n")
            old_key = os.environ.pop("FREIGHTTIGER_LLM_API_KEY", None)
            try:
                explainer = build_explainer(llm_enabled=True, dotenv_path=path)
                self.assertTrue(explainer.config.enabled)
                self.assertEqual(explainer.usage.mode, "online")
                # never expose the key on the usage summary object
                self.assertFalse(hasattr(explainer.usage, "api_key"))
            finally:
                if old_key is not None:
                    os.environ["FREIGHTTIGER_LLM_API_KEY"] = old_key
                else:
                    os.environ.pop("FREIGHTTIGER_LLM_API_KEY", None)

    def test_build_explainer_stays_offline_without_any_key(self) -> None:
        import os

        old_key = os.environ.pop("FREIGHTTIGER_LLM_API_KEY", None)
        try:
            explainer = build_explainer(
                llm_enabled=True, dotenv_path="/nonexistent/freighttiger/.env"
            )
            self.assertFalse(explainer.config.enabled)
            self.assertEqual(explainer.usage.mode, "offline")
            self.assertIsNone(explainer.usage.model)
        finally:
            if old_key is not None:
                os.environ["FREIGHTTIGER_LLM_API_KEY"] = old_key

    def test_project_dotenv_path_is_anchored_to_repo_root(self) -> None:
        from llm_explainer import DOTENV_PATH, PROJECT_ROOT

        self.assertEqual(DOTENV_PATH, PROJECT_ROOT / ".env")
        self.assertTrue(DOTENV_PATH.is_absolute())
        self.assertEqual(PROJECT_ROOT, ROOT)


if __name__ == "__main__":
    unittest.main()
