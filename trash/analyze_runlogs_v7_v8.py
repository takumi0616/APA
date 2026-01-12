import re
from collections import Counter
from pathlib import Path


CASE_RE = re.compile(
    r"\[CASE\]\s+id=(?P<case_id>[^ ]+)\s+"
    r"ok=(?P<ok>TRUE|FALSE)\s+ok_warp=(?P<ok_warp>TRUE|FALSE)\s+"
    r"stage=(?P<stage>[^ ]+)\s+"
    r"unknown_reason=(?P<unknown_reason>[^ ]*)\s+"
    r"gt_form=(?P<gt_form>[^ ]*)\s+pred_form=(?P<pred_form>[^ ]*)"
)


def parse_log(path: Path):
    by_form_stage = Counter()
    by_form_unknown_reason = Counter()
    by_form_unknown_reason_stage = Counter()
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = CASE_RE.search(line)
            if not m:
                continue
            total += 1
            case_id = m.group("case_id")
            src_form = case_id.split("_", 1)[0] if "_" in case_id else "?"
            stage = m.group("stage")
            unknown_reason = (m.group("unknown_reason") or "").strip()

            by_form_stage[(src_form, stage)] += 1
            if stage == "form_unknown":
                by_form_unknown_reason[(src_form, unknown_reason or "(empty)")] += 1
                by_form_unknown_reason_stage[(src_form, stage, unknown_reason or "(empty)")] += 1

    return {
        "path": str(path),
        "total_cases": total,
        "by_form_stage": by_form_stage,
        "by_form_unknown_reason": by_form_unknown_reason,
    }


def print_report(label: str, result: dict):
    print(f"=== {label}: {result['path']}")
    print(f"total_cases(parsed): {result['total_cases']}")
    print("\n[by_form_stage]")
    for form in ["A", "B", "C"]:
        rows = [(stage, c) for (f, stage), c in result["by_form_stage"].items() if f == form]
        for stage, c in sorted(rows, key=lambda x: (-x[1], x[0])):
            print(f"  {form} {stage:20} {c}")
    print("\n[form_unknown reasons]")
    for form in ["A", "B", "C"]:
        rows = [(reason, c) for (f, reason), c in result["by_form_unknown_reason"].items() if f == form]
        for reason, c in sorted(rows, key=lambda x: (-x[1], x[0])):
            print(f"  {form} {reason:20} {c}")


def main():
    root = Path(__file__).resolve().parents[1]  # .../APA
    v7 = root / "output_pipeline" / "run_20260112_110811" / "run.log"
    v8 = root / "output_pipeline" / "run_20260112_131554" / "run.log"

    r7 = parse_log(v7)
    r8 = parse_log(v8)

    print_report("v7", r7)
    print("\n" + "-" * 70 + "\n")
    print_report("v8", r8)


if __name__ == "__main__":
    main()

