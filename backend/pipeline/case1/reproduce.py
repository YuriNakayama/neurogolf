"""case1 baseline — 既知の高スコア ``submission.zip`` を再現（取得 + 検証）する。

NeuroGolf の公開ノートブックには 400 タスク完成済みの ONNX バンドルを出力する
ものがある。本モジュールは Kaggle カーネル出力からその ``submission.zip`` を
取得し、**バイト数と SHA256 を固定値と照合**して「正しいベースラインを掴んでいる」
ことを保証する。新しいモデルは作らない — 既知良好バンドルをそのまま提出すること
で、ある程度の Public Score を担保するのが狙い。

固定対象（Public Score 7159.44）::

    Kaggle kernel: boristown/agi-neural-golf-visualization-baseline
    submission.zip: 542649 bytes, sha256 33a16642…9baa1e, 400 task ONNX
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

# 再現対象バンドル（boristown/agi-neural-golf-visualization-baseline, LB 7159.44）。
# kaggle kernels output で取得した submission.zip を固定。差し替え時はこの 3 値を更新。
TARGET_KERNEL = "boristown/agi-neural-golf-visualization-baseline"
EXPECTED_SHA256 = "33a16642e139d04ad61d6edcccf1a72b26013e2aeee2c9070a7f1f095e9baa1e"
EXPECTED_BYTES = 542649

SUBMISSION_NAME = "submission.zip"
_CHUNK = 1 << 20


class ReproduceError(RuntimeError):
    """ベースラインバンドルの取得・検証に失敗したときに投げる。"""


@dataclass(frozen=True)
class Bundle:
    """検証済みの提出バンドル。"""

    zip_path: Path
    size_bytes: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(zip_path: Path) -> Bundle:
    """``zip_path`` が固定 SHA256 / バイト数と一致するか検証する。

    一致すれば ``Bundle`` を返し、欠損・不一致なら ``ReproduceError`` を投げる。
    """

    if not zip_path.is_file():
        raise ReproduceError(f"バンドルが見つかりません: {zip_path}")

    # SHA256 が一意の身元保証。digest 一致ならバイト数も必ず一致するので、digest を
    # 先に照合する（バイト数は人間可読な補助チェックとして後段で確認）。
    digest = _sha256(zip_path)
    if digest != EXPECTED_SHA256:
        raise ReproduceError(
            "SHA256 が一致しません:\n"
            f"  期待 {EXPECTED_SHA256}\n"
            f"  実際 {digest}\n"
            f"  {zip_path}"
        )

    size = zip_path.stat().st_size
    if size != EXPECTED_BYTES:
        raise ReproduceError(
            f"バイト数が一致しません: 期待 {EXPECTED_BYTES} != 実際 {size} ({zip_path})"
        )

    return Bundle(zip_path=zip_path, size_bytes=size, sha256=digest)


def fetch_target(out_dir: Path) -> Path:
    """Kaggle カーネル出力から ``submission.zip`` を ``out_dir`` に取得して返す。

    ``kaggle kernels output <TARGET_KERNEL>`` を実行する。認証は環境変数
    （``KAGGLE_USERNAME`` / ``KAGGLE_KEY``）または ``~/.kaggle/kaggle.json``。
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "kernels", "output", TARGET_KERNEL, "-p", str(out_dir)]
    try:
        proc = subprocess.run(  # noqa: S603 — trusted CLI
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ReproduceError(
            "`kaggle` コマンドが見つかりません。`uv run` 経由で実行してください。"
        ) from exc
    if proc.returncode != 0:
        raise ReproduceError(f"kaggle kernels output 失敗:\n{proc.stderr.strip()}")

    zip_path = out_dir / SUBMISSION_NAME
    if not zip_path.is_file():
        raise ReproduceError(f"取得物に {SUBMISSION_NAME} が含まれません: {out_dir}")
    return zip_path


def resolve_target(work_dir: Path, *, local_zip: Path | None = None) -> Bundle:
    """検証済みベースラインバンドルを返す。

    ``local_zip`` が与えられればそれを検証して使う（fetch しない）。なければ
    ``fetch_target`` で取得してから検証する。どちらも ``verify_bundle`` を通す。
    """

    if local_zip is not None:
        return verify_bundle(local_zip)
    fetched = fetch_target(work_dir)
    return verify_bundle(fetched)
