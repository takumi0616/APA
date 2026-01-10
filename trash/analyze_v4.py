"""Analyze results produced by paper_pipeline_v4.py.

This script reads an output_pipeline run's summary.csv and produces:

- Overall KPI stats (ok_expected_behavior / ok_warp)
- Stage counts (done / form_unknown / homography_unstable / ...)
- Per-form metrics (A/B accuracy, C reject success)
- Failure breakdowns (homography_unstable reasons, C false positives, B misclassifications)

Designed to be dependency-free (no pandas).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _as_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    if s == "":
        return None
    return None


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _safe_json_loads(v: Any) -> Optional[Any]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def read_summary_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


@dataclass
class Metrics:
    total: int
    ok_expected: int
    ok_warp: int
    stage_counts: Dict[str, int]
    form_counts: Dict[str, int]


def compute_metrics(rows: List[Dict[str, str]]) -> Metrics:
    stage_counter: Counter[str] = Counter()
    form_counter: Counter[str] = Counter()
    ok_expected = 0
    ok_warp = 0
    for r in rows:
        stage = (r.get("pipeline_stop_stage") or "").strip() or "(empty)"
        stage_counter[stage] += 1
        form = (r.get("source_form_folder_name(A_or_B_or_C)") or "").strip() or "(empty)"
        form_counter[form] += 1

        if _as_bool(r.get("pipeline_final_ok(expected_behavior)")):
            ok_expected += 1
        if _as_bool(r.get("pipeline_final_ok(warp_done)")):
            ok_warp += 1

    return Metrics(
        total=len(rows),
        ok_expected=ok_expected,
        ok_warp=ok_warp,
        stage_counts=dict(stage_counter),
        form_counts=dict(form_counter),
    )


def _pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else 100.0 * float(n) / float(d)


def compute_per_form_stats(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    # NOTE: C cases are expected to be rejected (form_unknown)
    by_form: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_form[(r.get("source_form_folder_name(A_or_B_or_C)") or "").strip()].append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for form, items in by_form.items():
        total = len(items)
        out[form] = {"cases": total}
        if form in {"A", "B"}:
            form_ok = sum(1 for r in items if _as_bool(r.get("is_predicted_form_correct")) is True)
            tpl_ok = sum(1 for r in items if _as_bool(r.get("is_predicted_best_template_correct")) is True)
            out[form].update(
                {
                    "form_accuracy": form_ok,
                    "form_accuracy_pct": _pct(form_ok, total),
                    "template_accuracy": tpl_ok,
                    "template_accuracy_pct": _pct(tpl_ok, total),
                }
            )
        elif form == "C":
            reject = sum(1 for r in items if (r.get("pipeline_stop_stage") or "").strip() == "form_unknown")
            fp_as_a = sum(1 for r in items if (r.get("predicted_decided_form(A_or_B_or_empty)") or "").strip() == "A")
            fp_as_b = sum(1 for r in items if (r.get("predicted_decided_form(A_or_B_or_empty)") or "").strip() == "B")
            out[form].update(
                {
                    "reject_success": reject,
                    "reject_success_pct": _pct(reject, total),
                    "false_positive_as_A": fp_as_a,
                    "false_positive_as_B": fp_as_b,
                }
            )
    return out


def compute_failure_breakdowns(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Collect breakdowns useful for debugging."""
    out: Dict[str, Any] = {}

    # 1) homography_unstable cases
    homo_rows = [r for r in rows if (r.get("pipeline_stop_stage") or "").strip() == "homography_unstable"]
    reason_counter: Counter[str] = Counter()
    for r in homo_rows:
        reason = (r.get("homography_inversion_reject_reason") or "").strip() or "(empty)"
        reason_counter[reason] += 1

    out["homography_unstable"] = {
        "count": len(homo_rows),
        "reasons": dict(reason_counter),
        "examples": [
            {
                "case_id": r.get("case_id"),
                "src_form": r.get("source_form_folder_name(A_or_B_or_C)"),
                "src_image": r.get("source_image_filename"),
                "gt_form": r.get("ground_truth_source_form(A_or_B)"),
                "pred_form": r.get("predicted_decided_form(A_or_B_or_empty)"),
                "best_template": r.get("predicted_best_template_filename"),
                "xfeat_inliers": _as_int(r.get("xfeat_best_inliers")),
                "xfeat_inlier_ratio": _as_float(r.get("xfeat_best_inlier_ratio")),
                "reject_reason": (r.get("homography_inversion_reject_reason") or "").strip(),
            }
            for r in homo_rows[:20]
        ],
    }

    # 2) B misclassification patterns
    b_rows = [r for r in rows if (r.get("source_form_folder_name(A_or_B_or_C)") or "").strip() == "B"]
    b_mis = [r for r in b_rows if _as_bool(r.get("is_predicted_form_correct")) is False]
    b_pred_counter: Counter[str] = Counter(
        (r.get("predicted_decided_form(A_or_B_or_empty)") or "").strip() or "(empty)" for r in b_mis
    )
    b_stage_counter: Counter[str] = Counter((r.get("pipeline_stop_stage") or "").strip() or "(empty)" for r in b_mis)
    out["B_misclassified"] = {
        "count": len(b_mis),
        "predicted_form_breakdown": dict(b_pred_counter),
        "stage_breakdown": dict(b_stage_counter),
        "examples": [
            {
                "case_id": r.get("case_id"),
                "src_image": r.get("source_image_filename"),
                "pred_form": r.get("predicted_decided_form(A_or_B_or_empty)"),
                "pred_rot": r.get("predicted_decided_rotation_angle_deg"),
                "best_template": r.get("predicted_best_template_filename"),
                "stage": r.get("pipeline_stop_stage"),
                "xfeat_inliers": _as_int(r.get("xfeat_best_inliers")),
                "xfeat_inlier_ratio": _as_float(r.get("xfeat_best_inlier_ratio")),
                "homo_reject_reason": r.get("homography_inversion_reject_reason"),
            }
            for r in b_mis[:20]
        ],
    }

    # 3) C false positives (expected to be rejected)
    c_rows = [r for r in rows if (r.get("source_form_folder_name(A_or_B_or_C)") or "").strip() == "C"]
    c_fp = [r for r in c_rows if (r.get("pipeline_stop_stage") or "").strip() != "form_unknown"]
    c_fp_pred_counter: Counter[str] = Counter(
        (r.get("predicted_decided_form(A_or_B_or_empty)") or "").strip() or "(empty)" for r in c_fp
    )
    out["C_false_positive"] = {
        "count": len(c_fp),
        "predicted_form_breakdown": dict(c_fp_pred_counter),
        "examples": [
            {
                "case_id": r.get("case_id"),
                "src_image": r.get("source_image_filename"),
                "stage": r.get("pipeline_stop_stage"),
                "pred_form": r.get("predicted_decided_form(A_or_B_or_empty)"),
                "pred_rot": r.get("predicted_decided_rotation_angle_deg"),
                "best_template": r.get("predicted_best_template_filename"),
                "form_score": _as_float(r.get("form_decision_score")),
                "xfeat_inliers": _as_int(r.get("xfeat_best_inliers")),
                "xfeat_inlier_ratio": _as_float(r.get("xfeat_best_inlier_ratio")),
            }
            for r in c_fp[:50]
        ],
    }

    return out


def render_markdown(
    summary_path: Path,
    metrics: Metrics,
    per_form: Dict[str, Dict[str, Any]],
    breakdowns: Dict[str, Any],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"# paper_pipeline_v4 summary 分析レポート")
    lines.append("")
    lines.append(f"- 生成日時: {now}")
    lines.append(f"- 入力: `{summary_path.as_posix()}`")
    lines.append("")

    lines.append("## 全体 KPI")
    lines.append("")
    lines.append(f"- total_cases: **{metrics.total}**")
    lines.append(f"- ok_expected_behavior: **{metrics.ok_expected}** ({_pct(metrics.ok_expected, metrics.total):.1f}%)")
    lines.append(f"- ok_warp: **{metrics.ok_warp}** ({_pct(metrics.ok_warp, metrics.total):.1f}%)")
    lines.append("")

    lines.append("## stage counts")
    lines.append("")
    lines.append("| stage | count |")
    lines.append("|---|---:|")
    for stage, cnt in sorted(metrics.stage_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {stage} | {cnt} |")
    lines.append("")

    lines.append("## form 別")
    lines.append("")
    lines.append("| form | cases | form_acc | template_acc | reject_success(C) | false_pos_A(C) | false_pos_B(C) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for form in ["A", "B", "C"]:
        d = per_form.get(form, {"cases": 0})
        if form in {"A", "B"}:
            lines.append(
                "| {form} | {cases} | {fa} ({fa_pct:.1f}%) | {ta} ({ta_pct:.1f}%) |  |  |  |".format(
                    form=form,
                    cases=d.get("cases", 0),
                    fa=d.get("form_accuracy", 0),
                    fa_pct=d.get("form_accuracy_pct", 0.0),
                    ta=d.get("template_accuracy", 0),
                    ta_pct=d.get("template_accuracy_pct", 0.0),
                )
            )
        else:
            lines.append(
                "| C | {cases} |  |  | {rej} ({rej_pct:.1f}%) | {fpA} | {fpB} |".format(
                    cases=d.get("cases", 0),
                    rej=d.get("reject_success", 0),
                    rej_pct=d.get("reject_success_pct", 0.0),
                    fpA=d.get("false_positive_as_A", 0),
                    fpB=d.get("false_positive_as_B", 0),
                )
            )
    lines.append("")

    # Breakdown: homography unstable
    homo = breakdowns.get("homography_unstable", {})
    lines.append("## 失敗内訳: homography_unstable")
    lines.append("")
    lines.append(f"- count: **{homo.get('count', 0)}**")
    lines.append("")
    lines.append("### reject reason breakdown")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("|---|---:|")
    for reason, cnt in sorted((homo.get("reasons") or {}).items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {reason} | {cnt} |")
    lines.append("")

    # Breakdown: B misclassifications
    bmis = breakdowns.get("B_misclassified", {})
    lines.append("## 失敗内訳: B 誤判定（is_predicted_form_correct=FALSE）")
    lines.append("")
    lines.append(f"- count: **{bmis.get('count', 0)}**")
    lines.append("")
    lines.append("### predicted form breakdown")
    lines.append("")
    lines.append("| predicted_form | count |")
    lines.append("|---|---:|")
    for k, v in sorted((bmis.get("predicted_form_breakdown") or {}).items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("### stage breakdown")
    lines.append("")
    lines.append("| stage | count |")
    lines.append("|---|---:|")
    for k, v in sorted((bmis.get("stage_breakdown") or {}).items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Breakdown: C false positives
    cfp = breakdowns.get("C_false_positive", {})
    lines.append("## 失敗内訳: C の false positive（form_unknown にならなかった）")
    lines.append("")
    lines.append(f"- count: **{cfp.get('count', 0)}**")
    lines.append("")
    lines.append("### predicted form breakdown")
    lines.append("")
    lines.append("| predicted_form | count |")
    lines.append("|---|---:|")
    for k, v in sorted((cfp.get("predicted_form_breakdown") or {}).items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## 参考（次のアクション案）")
    lines.append("")
    lines.append("- B の失敗は `homography_unstable` と `form_unknown/ambiguous` が混在しているため、")
    lines.append("  - form判定の閾値（unknown化）と、XFeat のマッチ条件（inliers/inlier_ratio）を分けて見直す")
    lines.append("  - `homography_inversion_reject_reason` の内訳に応じて、RANSAC/候補テンプレ数/解像度設定を調整")
    lines.append("- C の false positive は A 側のマーカー検出/幾何制約が通っているケースなので、")
    lines.append("  - `marker_area_page_ratio_mean` が下限ギリギリのものを弾くなど、条件を微調整する余地あり")
    lines.append("")

    return "\n".join(lines) + "\n"


def write_failures_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    """Write only non-expected cases for easy inspection."""
    failures = [r for r in rows if _as_bool(r.get("pipeline_final_ok(expected_behavior)")) is not True]
    if not failures:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text("", encoding="utf-8")
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(failures[0].keys()))
        writer.writeheader()
        writer.writerows(failures)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze paper_pipeline_v4 summary.csv")
    p.add_argument(
        "--input",
        required=True,
        help="Path to summary.csv (e.g., APA/output_pipeline/run_xxx/summary.csv)",
    )
    p.add_argument(
        "--output-md",
        default="",
        help="Optional path to write markdown report.",
    )
    p.add_argument(
        "--output-failures-csv",
        default="",
        help="Optional path to write filtered failures CSV (expected_behavior!=TRUE).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    # Windows console sometimes defaults to a legacy code page.
    # Ensure we can print Japanese text without crashing.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = build_arg_parser().parse_args(argv)

    summary_path = Path(args.input)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    rows = read_summary_csv(summary_path)
    metrics = compute_metrics(rows)
    per_form = compute_per_form_stats(rows)
    breakdowns = compute_failure_breakdowns(rows)

    md = render_markdown(summary_path, metrics, per_form, breakdowns)

    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
        print(f"[OK] wrote markdown: {out_md}")
    else:
        # Print to stdout only when not writing to file
        print(md)

    if args.output_failures_csv:
        write_failures_csv(rows, Path(args.output_failures_csv))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

