#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""APA.py

実行コマンド（Windows / リポジトリルート `C:/Users/takumi/develop` から）
----------------------------------------------------------

    C:/Users/takumi/develop/miniconda3/python.exe APA/APA.py --log-level INFO --console-log-level INFO

例（最初の1枚だけ確認する）
--------------------------

    C:/Users/takumi/develop/miniconda3/python.exe APA/APA.py --limit 1 --log-level INFO --console-log-level INFO

概要
----
- `./APA/apa_input` 配下の画像をファイル名に依存せず上から順に処理します。
- テンプレは `./APA/apa_template/A` と `./APA/apa_template/B` を使用します。
- 出力は **9_demo 相当の画像のみ** を `./APA/apa_output/run_.../` に保存します。
- ログは `./APA/apa_log/run_.../run.log` と `summary.csv` に保存します。

注意
----
- 本ファイルは薄いランチャーで、重い処理は `APA_back.py` のみを import します。
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from APA_back import build_arg_parser, run_apa_pipeline

    args = build_arg_parser().parse_args(argv)
    out_dir, log_dir = run_apa_pipeline(args)
    print(f"[DONE] output: {out_dir}")
    print(f"[DONE] log   : {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
