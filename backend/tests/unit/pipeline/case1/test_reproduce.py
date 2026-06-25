"""Tests for case1 reproduce (bundle verify) and the submit loop."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pipeline.case1 import reproduce, submit_loop


def _make_zip(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def test_verify_bundle_accepts_matching_digest(tmp_path: Path, monkeypatch) -> None:
    content = b"matching-bundle-bytes"
    zp = _make_zip(tmp_path / "submission.zip", content)
    monkeypatch.setattr(
        reproduce, "EXPECTED_SHA256", hashlib.sha256(content).hexdigest()
    )
    monkeypatch.setattr(reproduce, "EXPECTED_BYTES", len(content))
    bundle = reproduce.verify_bundle(zp)
    assert bundle.size_bytes == len(content)
    assert bundle.zip_path == zp


def test_verify_bundle_rejects_wrong_digest(tmp_path: Path) -> None:
    zp = _make_zip(tmp_path / "submission.zip", b"unexpected")
    with pytest.raises(reproduce.ReproduceError, match="SHA256"):
        reproduce.verify_bundle(zp)


def test_verify_bundle_missing_file(tmp_path: Path) -> None:
    with pytest.raises(reproduce.ReproduceError):
        reproduce.verify_bundle(tmp_path / "nope.zip")


def test_resolve_target_uses_local_zip_without_fetch(
    tmp_path: Path, monkeypatch
) -> None:
    content = b"local"
    zp = _make_zip(tmp_path / "submission.zip", content)
    monkeypatch.setattr(
        reproduce, "EXPECTED_SHA256", hashlib.sha256(content).hexdigest()
    )
    monkeypatch.setattr(reproduce, "EXPECTED_BYTES", len(content))

    def _no_fetch(_out: Path) -> Path:  # pragma: no cover - must not be called
        raise AssertionError("fetch should not run when local_zip is given")

    monkeypatch.setattr(reproduce, "fetch_target", _no_fetch)
    bundle = reproduce.resolve_target(tmp_path, local_zip=zp)
    assert bundle.zip_path == zp


def test_reached_tolerance() -> None:
    assert submit_loop.reached(7166.06, 7166.06)
    assert submit_loop.reached(7170.0, 7166.06)
    assert not submit_loop.reached(7000.0, 7166.06)
    assert not submit_loop.reached(None, 7166.06)


def test_run_until_target_stops_on_success(monkeypatch) -> None:
    attempts: list[str] = []

    def fake_submit_once(zip_path, message, **kw) -> submit_loop.SubmitOutcome:
        attempts.append(message)
        return submit_loop.SubmitOutcome(
            submitted=True, status="complete", public_score=7166.06, raw=""
        )

    monkeypatch.setattr(submit_loop, "submit_once", fake_submit_once)
    out = submit_loop.run_until_target(
        "x.zip", "m", 7166.06, max_attempts=5, poll_seconds=1, poll_interval=1
    )
    assert out.public_score == 7166.06
    assert len(attempts) == 1  # stopped immediately on success


def test_run_until_target_exhausts_attempts(monkeypatch) -> None:
    attempts: list[str] = []

    def fake_submit_once(zip_path, message, **kw) -> submit_loop.SubmitOutcome:
        attempts.append(message)
        return submit_loop.SubmitOutcome(
            submitted=True, status="complete", public_score=10.0, raw=""
        )

    monkeypatch.setattr(submit_loop, "submit_once", fake_submit_once)
    out = submit_loop.run_until_target(
        "x.zip", "m", 7166.06, max_attempts=3, poll_seconds=1, poll_interval=1
    )
    assert not submit_loop.reached(out.public_score, 7166.06)
    assert len(attempts) == 3
