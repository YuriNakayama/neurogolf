"""候補バンドルの解決（dir または submission.zip → taskNNN.onnx を含む dir）。

Kaggle ノートブック ``biohack44/neurogolf-2026-blend-max`` の Cell 6 ``_resolve``
に対応。提出バンドルは「``taskNNN.onnx`` を直接含むディレクトリ」または
「``submission.zip``」のどちらかで与えられる。zip の場合は展開し、実際に
``task*.onnx`` を含むディレクトリを返す。
"""

from __future__ import annotations

import glob
import os
import zipfile


class BundleError(RuntimeError):
    """バンドルの解決に失敗したとき送出。"""


def resolve_bundle(
    dir_: str | None,
    zip_: str | None,
    tag: str,
    work_dir: str,
) -> str:
    """バンドルを解決し ``task*.onnx`` を含むディレクトリのパスを返す。

    - ``dir_`` が与えられればそれをそのまま使う。
    - そうでなければ ``zip_`` を ``work_dir/_src_<tag>`` に展開し、``task*.onnx``
      を含むディレクトリを再帰的に探して返す。

    どちらも無効なら :class:`BundleError` を送出する。
    """
    if dir_:
        n = len(glob.glob(os.path.join(dir_, "task*.onnx")))
        print(f"[{tag}] using dir {dir_}  ({n} onnx)")
        return dir_
    if not zip_:
        raise BundleError(f"[{tag}] dir も zip も指定されていません")
    dst = os.path.join(work_dir, f"_src_{tag}")
    os.makedirs(dst, exist_ok=True)
    with zipfile.ZipFile(zip_) as z:
        z.extractall(dst)
    # find the dir that actually holds task*.onnx
    for root, _, files in os.walk(dst):
        if any(f.startswith("task") and f.endswith(".onnx") for f in files):
            n = len(glob.glob(os.path.join(root, "task*.onnx")))
            print(f"[{tag}] unzipped {zip_} -> {root}  ({n} onnx)")
            return root
    raise BundleError(f"[{tag}] no task*.onnx found in {zip_}")
