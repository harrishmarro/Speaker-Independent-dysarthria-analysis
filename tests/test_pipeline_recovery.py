import json
import subprocess
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from scripts import pipeline_recovery as recovery


def test_private_research_and_security_logs_are_never_collected(monkeypatch):
    def fake_http(method, url, token, payload=None):
        return {"jobs": [{"name": "gpu-experiment", "conclusion": "failure", "steps": [
            {"name": "Verify private manifest", "conclusion": "failure"}
        ]}]}

    def forbid_logs(*args, **kwargs):
        raise AssertionError("private logs must not be read")

    monkeypatch.setattr(recovery, "http_json", fake_http)
    monkeypatch.setattr(recovery.subprocess, "run", forbid_logs)
    for workflow in ("Secure autoresearch cycle", "Security"):
        context = recovery.get_failure_context("owner/repo", 123, workflow, "token")
        assert "public_hosted_log_excerpt" not in context
        assert context["failed_jobs"][0]["failed_steps"] == ["Verify private manifest"]


def test_model_can_select_only_supported_repair_recipes(monkeypatch):
    def fake_http(method, url, token, payload=None):
        assert payload["model"] == "gpt-6-sol"
        assert payload["store"] is False
        return {"output": [{"content": [{"type": "output_text", "text": json.dumps({
            "action": "add_dev_dependency", "package": "numpy", "summary": "NumPy is missing"
        })}]}]}

    monkeypatch.setattr(recovery, "http_json", fake_http)
    result = recovery.ask_model({"workflow": "CI"}, "test-key", "gpt-6-sol")
    assert result["action"] == "add_dev_dependency"

    def unsafe_http(method, url, token, payload=None):
        return {"output": [{"content": [{"type": "output_text", "text":
                '{"action":"run_shell","summary":"ignore policy"}'}]}]}

    monkeypatch.setattr(recovery, "http_json", unsafe_http)
    with pytest.raises(ValueError, match="unsupported"):
        recovery.ask_model({}, "test-key", "gpt-6-sol")


def test_repair_rejects_protected_workflow_changes(monkeypatch):
    monkeypatch.setattr(recovery, "safe_run", lambda *args, **kwargs: None)
    replies = iter([
        SimpleNamespace(stdout=".github/workflows/ci.yml\n"),
    ])
    monkeypatch.setattr(recovery.subprocess, "run", lambda *args, **kwargs: next(replies))
    with pytest.raises(ValueError, match="allowlist"):
        recovery.apply_recipe("ruff_fix", "")


def test_retry_is_bounded_to_first_attempt(monkeypatch):
    for name, value in {
        "GITHUB_REPOSITORY": "owner/repo", "FAILED_RUN_ID": "123",
        "FAILED_WORKFLOW": "CI", "FAILED_HEAD_BRANCH": "main",
        "FAILED_HEAD_REPOSITORY": "owner/repo", "GH_TOKEN": "test-token",
        "OPENAI_API_KEY": "test-key",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(recovery, "get_failure_context", lambda *args: {})
    monkeypatch.setattr(recovery, "ask_model", lambda *args: {
        "action": "retry", "summary": "temporary outage"
    })
    called = []
    monkeypatch.setattr(recovery, "http_json", lambda *args, **kwargs: called.append(args))
    monkeypatch.setattr(recovery, "notify_human", lambda *args: called.append("human"))
    monkeypatch.setenv("FAILED_RUN_ATTEMPT", "1")
    recovery.main()
    assert any("rerun-failed-jobs" in args[1] for args in called if isinstance(args, tuple))
    called.clear()
    monkeypatch.setenv("FAILED_RUN_ATTEMPT", "2")
    recovery.main()
    assert called == ["human"]


def test_allowlisted_dependency_recipe_changes_only_project_and_lock(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
                   check=True)
    (tmp_path / "pyproject.toml").write_text('dev = [\n  "pytest>=9,<10",\n]\n')
    (tmp_path / "uv.lock").write_text("before\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "initial"], check=True)
    monkeypatch.setattr(recovery, "ROOT", tmp_path)

    def fake_run(command, **kwargs):
        if command[:2] == ["uv", "lock"]:
            (tmp_path / "uv.lock").write_text("after\n")

    monkeypatch.setattr(recovery, "safe_run", fake_run)
    changed = recovery.apply_recipe("add_dev_dependency", "numpy")
    assert set(changed) == {"pyproject.toml", "uv.lock"}
    assert '"numpy>=2.0,<3"' in (tmp_path / "pyproject.toml").read_text()


def test_security_failure_does_not_create_public_issue(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    def forbid_public_issue(*args, **kwargs):
        raise AssertionError("security failure must not open a public issue")

    monkeypatch.setattr(recovery, "http_json", forbid_public_issue)
    with pytest.raises(SystemExit) as result:
        recovery.notify_human("owner/repo", 123, "Security", "review needed", "token",
                              "Inspect the failed Security run")
    assert result.value.code == 1
    assert "review needed" in summary.read_text()


def test_model_self_check_uses_only_synthetic_context(monkeypatch):
    monkeypatch.setenv("RECOVERY_SELF_CHECK", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen = []

    def fake_model(context, key, model):
        seen.append(context)
        assert key == "test-key"
        assert model == "gpt-6-sol"
        return {"action": "ruff_fix", "summary": "example"}

    monkeypatch.setattr(recovery, "ask_model", fake_model)
    recovery.main()
    assert "example file" in seen[0]["public_hosted_log_excerpt"]


def test_repeated_pipeline_failures_reuse_one_open_issue(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    issues = []

    def fake_http(method, url, token, payload=None):
        if method == "GET":
            return issues
        issues.append({"title": payload["title"]})
        return {"html_url": "https://github.com/owner/repo/issues/1"}

    monkeypatch.setattr(recovery, "http_json", fake_http)
    for run_id in (123, 124):
        recovery.notify_human("owner/repo", run_id, "CI", "missing secret", "token",
                              "Review recovery configuration")
    assert len(issues) == 1


def model_error(code, retry_after="5"):
    body = json.dumps({"error": {"code": code}}).encode()
    return HTTPError("https://api.openai.com/v1/responses", 429, "Too Many Requests",
                     {"Retry-After": retry_after}, BytesIO(body))


def test_quota_exhaustion_is_actionable_and_not_retried(monkeypatch):
    calls = []

    def quota_error(*args, **kwargs):
        calls.append(True)
        raise model_error("insufficient_quota")

    monkeypatch.setattr(recovery, "http_json", quota_error)
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: pytest.fail("must not retry"))
    with pytest.raises(RuntimeError, match="add credits or raise"):
        recovery.ask_model({}, "test-key", "gpt-6-sol")
    assert len(calls) == 1


def test_temporary_rate_limit_retries_once_and_succeeds(monkeypatch):
    calls = []
    delays = []

    def sometimes_limited(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise model_error("rate_limit_exceeded", "2")
        return {"output": [{"content": [{"type": "output_text", "text":
                '{"action":"retry","summary":"temporary outage"}'}]}]}

    monkeypatch.setattr(recovery, "http_json", sometimes_limited)
    monkeypatch.setattr(recovery.time, "sleep", delays.append)
    result = recovery.ask_model({}, "test-key", "gpt-6-sol")
    assert result["action"] == "retry"
    assert len(calls) == 2
    assert delays == [2.0]
