"""Print F1 and BLEU-1 from a completed MemCoRe run."""

import argparse
import json
from pathlib import Path

LABELS = {
    "overall": "Overall",
    "category_1": "C1 Multi-hop",
    "category_2": "C2 Temporal",
    "category_3": "C3 Open-domain",
    "category_4": "C4 Single-hop",
    "category_5": "C5 Adversarial",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Run directory or final_results.json")
    args = parser.parse_args()
    path = args.result / "final_results.json" if args.result.is_dir() else args.result
    result = json.loads(path.read_text(encoding="utf-8"))
    print(f"Model: {result['model']}")
    print(f"Questions: {result['total_questions']}/{result['expected_questions']}")
    print(f"Complete: {result['complete']} | Metrics: {result['metrics_mode']}")
    print(f"{'Category':<20} {'N':>6} {'F1':>8} {'BLEU-1':>8}")
    for key, label in LABELS.items():
        metrics = result["aggregate_metrics"].get(key)
        if metrics is not None:
            print(f"{label:<20} {metrics['f1']['count']:>6} "
                  f"{100 * metrics['f1']['mean']:>8.2f} "
                  f"{100 * metrics['bleu1']['mean']:>8.2f}")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
