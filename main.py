import asyncio
import json
import os
import time
from typing import Dict, List, Tuple
from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge

# ---------------------------------------------------------------------------
# Ngưỡng Release Gate (có thể override bằng biến môi trường)
# ---------------------------------------------------------------------------
RELEASE_GATE = {
    # Sàn chất lượng tuyệt đối của V2
    "min_avg_score":       float(os.getenv("GATE_MIN_SCORE",      "2.5")),
    "min_hit_rate":        float(os.getenv("GATE_MIN_HIT_RATE",   "0.45")),
    "min_mrr":             float(os.getenv("GATE_MIN_MRR",        "0.40")),
    "min_faithfulness":    float(os.getenv("GATE_MIN_FAITH",      "0.10")),
    # Giới hạn hiệu năng
    "max_avg_latency_sec": float(os.getenv("GATE_MAX_LATENCY",    "3.0")),
    # Giới hạn hồi quy so với V1 (V2 không được thấp hơn V1 quá mức này)
    "max_score_regression":   float(os.getenv("GATE_MAX_SCORE_REG",   "-0.2")),
    "max_hitrate_regression": float(os.getenv("GATE_MAX_HR_REG",      "-0.10")),
    "max_mrr_regression":     float(os.getenv("GATE_MAX_MRR_REG",     "-0.10")),
}

# Chi phí ước tính (USD / 1 000 token) — dùng để báo cáo cost
COST_PER_1K_TOKENS = float(os.getenv("COST_PER_1K_TOKENS", "0.002"))


class ExpertEvaluator:
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()

    async def score(self, case: Dict, resp: Dict) -> Dict:
        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = resp.get("metadata", {}).get("sources", [])
        top_k = case.get("top_k", 3)

        hit_rate = self.retrieval_evaluator.calculate_hit_rate(
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids,
            top_k=top_k,
        )
        mrr = self.retrieval_evaluator.calculate_mrr(
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids,
        )

        return {"retrieval": {"hit_rate": hit_rate, "mrr": mrr, "top_k": top_k}}


def _build_summary(agent_version: str, results: List[Dict]) -> Dict:
    """Tổng hợp metrics từ danh sách kết quả (bỏ qua các case lỗi)."""
    valid = [r for r in results if "error" not in r]
    total_valid = len(valid)

    if total_valid == 0:
        return {
            "metadata": {
                "version": agent_version,
                "total": len(results),
                "valid": 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "metrics": {},
        }

    total_tokens = sum(r.get("tokens_used", 0) for r in valid)
    estimated_cost = (total_tokens / 1000) * COST_PER_1K_TOKENS

    return {
        "metadata": {
            "version": agent_version,
            "total": len(results),
            "valid": total_valid,
            "passed": sum(1 for r in valid if r.get("status") == "pass"),
            "failed": sum(1 for r in valid if r.get("status") == "fail"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score":      sum(r["judge"]["final_score"] for r in valid) / total_valid,
            "hit_rate":       sum(r["ragas"]["retrieval"]["hit_rate"] for r in valid) / total_valid,
            "mrr":            sum(r["ragas"]["retrieval"]["mrr"] for r in valid) / total_valid,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in valid) / total_valid,
            "faithfulness":   sum(r["judge"]["faithfulness"] for r in valid) / total_valid,
            "relevancy":      sum(r["judge"]["relevancy"] for r in valid) / total_valid,
            "avg_latency":    sum(r["latency"] for r in valid) / total_valid,
            "total_tokens":   total_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
        },
    }


def _build_v1_v2_compare(v1_results: List[Dict], v2_results: List[Dict]) -> List[Dict]:
    """Ghép kết quả V1/V2 theo question để phục vụ phân tích và báo cáo."""
    v1_by_question = {
        r.get("question"): r for r in v1_results
        if "error" not in r and r.get("question")
    }
    v2_by_question = {
        r.get("question"): r for r in v2_results
        if "error" not in r and r.get("question")
    }

    merged: List[Dict] = []
    for question in sorted(set(v1_by_question) & set(v2_by_question)):
        r1 = v1_by_question[question]
        r2 = v2_by_question[question]

        merged.append(
            {
                "question": question,
                "expected_answer": r2.get("expected_answer", r1.get("expected_answer", "")),
                "ground_truth_chunk_ids": r2.get(
                    "expected_retrieval_ids", r1.get("expected_retrieval_ids", [])
                ),
                "v1_answer": r1.get("agent_response", ""),
                "v1_retrieved_chunk_ids": r1.get("retrieved_ids", []),
                "v2_answer": r2.get("agent_response", ""),
                "v2_retrieved_chunk_ids": r2.get("retrieved_ids", []),
                "judge": {
                    "v1_score": r1.get("judge", {}).get("final_score"),
                    "v2_score": r2.get("judge", {}).get("final_score"),
                    "v1_correct": r1.get("status") == "pass",
                    "v2_correct": r2.get("status") == "pass",
                    "hallucination_v1": (
                        r1.get("judge", {}).get("faithfulness", 1.0) < 0.5
                    ),
                    "hallucination_v2": (
                        r2.get("judge", {}).get("faithfulness", 1.0) < 0.5
                    ),
                    "winner": "v2"
                    if (r2.get("judge", {}).get("final_score", 0)
                        >= r1.get("judge", {}).get("final_score", 0))
                    else "v1",
                },
                "latency_sec": {"v1": r1.get("latency"), "v2": r2.get("latency")},
            }
        )
    return merged


def apply_release_gate(
    v1_summary: Dict, v2_summary: Dict
) -> Tuple[bool, Dict, Dict]:
    """
    Kiểm tra đa tiêu chí (Chất lượng / Hiệu năng / Hồi quy) để quyết định Release.

    Trả về:
        approved  : True nếu tất cả checks đều pass
        checks    : dict chi tiết từng tiêu chí (True/False)
        deltas    : dict chênh lệch các metric V2 - V1
    """
    m1 = v1_summary["metrics"]
    m2 = v2_summary["metrics"]

    score_delta   = m2["avg_score"]  - m1["avg_score"]
    hr_delta      = m2["hit_rate"]   - m1["hit_rate"]
    mrr_delta     = m2["mrr"]        - m1["mrr"]
    latency_delta = m2["avg_latency"] - m1["avg_latency"]

    checks = {
        # --- Sàn chất lượng tuyệt đối ---
        f"score_floor  (V2={m2['avg_score']:.2f} ≥ {RELEASE_GATE['min_avg_score']})":
            m2["avg_score"] >= RELEASE_GATE["min_avg_score"],

        f"hit_rate_floor  (V2={m2['hit_rate']:.2%} ≥ {RELEASE_GATE['min_hit_rate']:.0%})":
            m2["hit_rate"] >= RELEASE_GATE["min_hit_rate"],

        f"mrr_floor  (V2={m2['mrr']:.2%} ≥ {RELEASE_GATE['min_mrr']:.0%})":
            m2["mrr"] >= RELEASE_GATE["min_mrr"],

        f"faithfulness_floor  (V2={m2['faithfulness']:.2%} ≥ {RELEASE_GATE['min_faithfulness']:.0%})":
            m2["faithfulness"] >= RELEASE_GATE["min_faithfulness"],

        # --- Giới hạn hiệu năng ---
        f"latency_budget  (V2={m2['avg_latency']:.2f}s ≤ {RELEASE_GATE['max_avg_latency_sec']}s)":
            m2["avg_latency"] <= RELEASE_GATE["max_avg_latency_sec"],

        # --- Không hồi quy so với V1 ---
        f"no_score_regression  (Δ={score_delta:+.2f} ≥ {RELEASE_GATE['max_score_regression']})":
            score_delta >= RELEASE_GATE["max_score_regression"],

        f"no_hitrate_regression  (Δ={hr_delta:+.2%} ≥ {RELEASE_GATE['max_hitrate_regression']:.0%})":
            hr_delta >= RELEASE_GATE["max_hitrate_regression"],

        f"no_mrr_regression  (Δ={mrr_delta:+.2%} ≥ {RELEASE_GATE['max_mrr_regression']:.0%})":
            mrr_delta >= RELEASE_GATE["max_mrr_regression"],
    }

    deltas = {
        "avg_score":    round(score_delta, 4),
        "hit_rate":     round(hr_delta, 4),
        "mrr":          round(mrr_delta, 4),
        "avg_latency":  round(latency_delta, 4),
    }

    return all(checks.values()), checks, deltas


async def run_benchmark_with_results(
    agent: MainAgent, label: str
) -> Tuple[List[Dict], Dict]:
    """Chạy toàn bộ benchmark cho một phiên bản agent cụ thể."""
    print(f"\n🚀 Benchmark [{label}] ...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng.")
        return None, None

    runner = BenchmarkRunner(agent, ExpertEvaluator(), LLMJudge())
    results = await runner.run_all(dataset)
    summary = _build_summary(label, results)
    return results, summary


async def main():
    # --- Chạy V1 (baseline) ---
    v1_results, v1_summary = await run_benchmark_with_results(
        agent=MainAgent(version="v1"),
        label="Agent_V1_Base",
    )

    # --- Chạy V2 (phiên bản cải tiến: keyword map mở rộng, latency thấp hơn) ---
    v2_results, v2_summary = await run_benchmark_with_results(
        agent=MainAgent(version="v2"),
        label="Agent_V2_Optimized",
    )

    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    # --- Regression Gate ---
    approved, checks, deltas = apply_release_gate(v1_summary, v2_summary)

    print("\n" + "=" * 60)
    print("📊  REGRESSION REPORT: V1  →  V2")
    print("=" * 60)
    header = f"{'Metric':<20} {'V1':>8} {'V2':>8} {'Δ':>8}"
    print(header)
    print("-" * 46)
    for key in ("avg_score", "hit_rate", "mrr", "faithfulness", "avg_latency"):
        v1v = v1_summary["metrics"].get(key, 0)
        v2v = v2_summary["metrics"].get(key, 0)
        d   = deltas.get(key, v2v - v1v)
        print(f"  {key:<18} {v1v:>8.3f} {v2v:>8.3f} {d:>+8.3f}")

    m2 = v2_summary["metrics"]
    print(f"\n  💰 Cost V2: ${m2['estimated_cost_usd']:.4f} "
          f"({m2['total_tokens']:,} tokens × ${COST_PER_1K_TOKENS}/1K)")

    print("\n📋  GATE CHECKS:")
    for label, passed in checks.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {label}")

    print("\n" + "=" * 60)
    if approved:
        decision = "APPROVE"
        print("✅  QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        decision = "BLOCK"
        failed_count = sum(1 for v in checks.values() if not v)
        print(f"❌  QUYẾT ĐỊNH: TỪ CHỐI (BLOCK) — {failed_count} tiêu chí không đạt")
    print("=" * 60)

    # --- Lưu báo cáo ---
    os.makedirs("reports", exist_ok=True)

    # summary.json lưu V2 + thông tin regression
    v2_summary["regression"] = {
        "baseline_version": v1_summary["metadata"]["version"],
        "candidate_version": v2_summary["metadata"]["version"],
        "deltas": deltas,
        "gate_checks": {k: bool(v) for k, v in checks.items()},
        "decision": decision,
    }
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)

    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    compare_payload = _build_v1_v2_compare(v1_results, v2_results)
    with open("reports/v1_v2_comparison.json", "w", encoding="utf-8") as f:
        json.dump(compare_payload, f, ensure_ascii=False, indent=2)

    print(
        "\n📁 Đã lưu: reports/summary.json  |  reports/benchmark_results.json"
        "  |  reports/v1_v2_comparison.json"
    )


if __name__ == "__main__":
    asyncio.run(main())
