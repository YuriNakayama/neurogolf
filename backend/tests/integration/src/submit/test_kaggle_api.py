"""kaggle_api のユニットテスト（CLI は subprocess をモック）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from submit import kaggle_api
from submit.packager import SUBMISSION_NAME


def test_extract_submission_id() -> None:
    out = "Your submission was accepted. Submission ID: 98765"
    assert kaggle_api._extract_submission_id(out) == "98765"


def test_extract_submission_id_not_found() -> None:
    assert kaggle_api._extract_submission_id("no id here") is None


class _FakeProc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_submit_success(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(
        cmd: list[str], capture_output: bool, text: bool, check: bool
    ) -> _FakeProc:
        captured["cmd"] = cmd
        return _FakeProc(0, stdout="Submission ID: 12345\n")

    monkeypatch.setattr("submit.kaggle_api.subprocess.run", fake_run)
    out = kaggle_api.submit(Path(SUBMISSION_NAME), "msg")
    assert "12345" in out
    cmd = captured["cmd"]
    assert cmd[0] == "kaggle"
    assert "competitions" in cmd
    assert "submit" in cmd
    assert kaggle_api.COMPETITION in cmd


def test_submit_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        cmd: list[str], capture_output: bool, text: bool, check: bool
    ) -> _FakeProc:
        return _FakeProc(1, stderr="auth error")

    monkeypatch.setattr("submit.kaggle_api.subprocess.run", fake_run)
    with pytest.raises(kaggle_api.KaggleCLIError):
        kaggle_api.submit(Path(SUBMISSION_NAME), "msg")


def test_submit_rejects_wrong_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    """basename が submission.zip でなければ kaggle を呼ぶ前に弾く。"""

    def fail_run(*_a: object, **_k: object) -> _FakeProc:  # pragma: no cover
        raise AssertionError("kaggle CLI must not be invoked for a bad filename")

    monkeypatch.setattr("submit.kaggle_api.subprocess.run", fail_run)
    with pytest.raises(kaggle_api.SubmissionNameError):
        kaggle_api.submit(Path("candidate_submission.zip"), "msg")


def test_submit_name_error_is_kaggle_cli_error() -> None:
    """SubmissionNameError は KaggleCLIError のサブクラス（既存 except で捕捉可能）。"""

    assert issubclass(kaggle_api.SubmissionNameError, kaggle_api.KaggleCLIError)


def test_confirm_submission_found(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"description": "case0 v2 trial", "status": "pending"}]
    monkeypatch.setattr(kaggle_api, "list_submissions", lambda: rows)
    monkeypatch.setattr("submit.kaggle_api.time.sleep", lambda _s: None)
    found = kaggle_api.confirm_submission("case0 v2", timeout_s=1, interval_s=0)
    assert found is not None
    assert found["status"] == "pending"


def test_confirm_submission_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kaggle_api, "list_submissions", lambda: [])
    monkeypatch.setattr("submit.kaggle_api.time.sleep", lambda _s: None)
    monkeypatch.setattr(
        "submit.kaggle_api.time.time",
        _CounterClock(start=0.0, step=10.0).now,
    )
    result = kaggle_api.confirm_submission("missing", timeout_s=5, interval_s=0)
    assert result is None


def test_confirm_submission_tolerates_cli_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def flaky() -> list[dict[str, str]]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise kaggle_api.KaggleCLIError("temporary")
        return [{"description": "hello world"}]

    monkeypatch.setattr(kaggle_api, "list_submissions", flaky)
    monkeypatch.setattr("submit.kaggle_api.time.sleep", lambda _s: None)
    found = kaggle_api.confirm_submission("hello", timeout_s=10, interval_s=0)
    assert found is not None


class _CounterClock:
    def __init__(self, start: float, step: float) -> None:
        self.value = start
        self.step = step

    def now(self) -> float:
        current = self.value
        self.value += self.step
        return current
