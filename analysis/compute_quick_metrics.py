import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def _to_set(values) -> Set[str]:
    if not isinstance(values, list):
        return set()
    return {str(v) for v in values}


def _rate(numerator: int, denominator: int) -> float:
    return (numerator / denominator) if denominator else 0.0


def compute_metrics(rows: List[Dict], version: str) -> Dict:
    total = len(rows)
    hit_count = 0
    correct_count = 0
    hallucination_count = 0

    retrieved_key = f"{version}_retrieved_chunk_ids"
    correct_key = f"{version}_correct"
    halluc_key = f"hallucination_{version}"

    for row in rows:
        gt = _to_set(row.get("ground_truth_chunk_ids", []))
        pred = _to_set(row.get(retrieved_key, []))
        if gt and pred and (gt & pred):
            hit_count += 1

        judge = row.get("judge", {})
        if bool(judge.get(correct_key, False)):
            correct_count += 1
        if bool(judge.get(halluc_key, False)):
            hallucination_count += 1

    return {
        "samples": total,
        "hit_rate": round(_rate(hit_count, total), 4),
        "accuracy": round(_rate(correct_count, total), 4),
        "hallucination_rate": round(_rate(hallucination_count, total), 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute quick V1 vs V2 metrics from v1_v2_comparison.json"
    )
    parser.add_argument(
        "--input",
        default="reports/v1_v2_comparison.json",
        help="Path to comparison JSON file",
    )
    parser.add_argument(
        "--output",
        default="reports/quick_metrics.json",
        help="Path to save metrics JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input JSON must be a list of benchmark rows")

    report = {
        "v1": compute_metrics(rows, "v1"),
        "v2": compute_metrics(rows, "v2"),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
