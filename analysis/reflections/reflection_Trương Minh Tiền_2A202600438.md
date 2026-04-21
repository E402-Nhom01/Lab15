# Reflection — Trương Minh Tiền (2A202600438)

**Vai trò trong nhóm:** Integration Lead / Tech Owner — điều phối integration, xây Release Gate, phân tích failure, duy trì phần còn lại của repo (không thuộc data/agent/runner/judge).
**Module đảm nhận:** `main.py` (Release Gate + regression report), `analysis/failure_analysis.md`, `analysis/compute_quick_metrics.py`, `check_lab.py`, `README.md`, `requirements.txt`, `GRADING_RUBRIC.md`; review & tuning chéo toàn repo.

---

## 1. Engineering Contribution (15đ)

### 1.1 Module / file cụ thể
| File | Nội dung |
|------|----------|
| `main.py` | Orchestrator của toàn pipeline: chạy V1 → V2 → Release Gate → lưu 3 reports. Định nghĩa `RELEASE_GATE` 8 tiêu chí (4 sàn chất lượng + 1 latency + 3 regression). Hàm `apply_release_gate`, `_build_summary`, `_build_v1_v2_compare`. |
| `analysis/failure_analysis.md` | Báo cáo 5-Whys cho 3 case tệ nhất, clustering 8 fail thành 3 nhóm, Action Plan theo priority P0/P1/P2. |
| `analysis/compute_quick_metrics.py` | CLI tool tính nhanh `hit_rate / accuracy / hallucination_rate` cho V1 và V2 từ `v1_v2_comparison.json`. |
| `check_lab.py`, `README.md`, `GRADING_RUBRIC.md`, `requirements.txt` | Maintenance và docs bao quát repo. |

### 1.2 Điểm kỹ thuật nổi bật
- **Release Gate đa tiêu chí:** 8 checks chia thành 3 lớp
  1. Sàn tuyệt đối V2: `score ≥ 2.5`, `hit_rate ≥ 45%`, `mrr ≥ 40%`, `faithfulness ≥ 10%`.
  2. Giới hạn hiệu năng: `avg_latency ≤ 3.0s`.
  3. Chống hồi quy vs V1: `Δscore ≥ -0.2`, `Δhit_rate ≥ -10%`, `Δmrr ≥ -10%`.
  Toàn bộ override được qua ENV (`GATE_MIN_SCORE`, `GATE_MAX_LATENCY`, v.v.) → dễ điều chỉnh threshold không cần sửa code.
- **Regression delta có ý nghĩa kỹ thuật:** `score_delta`, `hr_delta`, `mrr_delta`, `latency_delta` được tính và format `+.2f`/`+.2%` → readable trong console và trong JSON report.
- **3 artifact lưu song song:**
  - `reports/summary.json` — metrics V2 + regression block.
  - `reports/benchmark_results.json` — raw 50 case results.
  - `reports/v1_v2_comparison.json` — merged V1/V2 theo question, dùng cho downstream failure analysis và `compute_quick_metrics.py`.
- **Decision explicit:** `decision = "APPROVE" / "BLOCK"` ghi vào JSON — phục vụ CI/CD nếu cần gate merge bằng exit code (action item).
- **Tuning latency chéo:** khi gate `latency_budget` fail (V2 = 4.17s), đã phối hợp Phương Anh giảm xuống 2.80s qua 3 vòng tuning (rerank log off, max_tokens=160, top_k=5, warm-up).

### 1.3 Chứng minh qua Git
Commit `72c01cc feat: ignore golden_set.json` + các fix cross-module (timeout runner, prompt llm, retrieval top_k). Run cuối cùng: **APPROVE** — tất cả 8 gate checks pass.

---

## 2. Technical Depth (15đ)

### 2.1 MRR ở tầng decision
MRR là input vào 2 gate:
- `mrr_floor`: `V2 MRR ≥ 0.40` — sàn tuyệt đối.
- `no_mrr_regression`: `V2 − V1 ≥ -0.10` — bảo vệ khỏi fine-tune gây hồi quy.
Thực tế: V2 = 0.77, Δ = +0.067 → PASS cả 2. Việc đặt **2 loại gate trên cùng metric** (absolute + relative) là thiết kế phòng hộ: kể cả V1 tốt, V2 cũng phải đủ tốt tuyệt đối; và ngược lại, V2 tốt tuyệt đối nhưng kém V1 vẫn không release.

### 2.2 Cohen's Kappa ở tầng decision
Agreement rate (proxy Kappa) = 0.905 trên V2 → báo hiệu metrics tin cậy. Nếu Kappa thấp (ví dụ <0.4), phải nghi ngờ `final_score` và có thể **không cho APPROVE dù pass các ngưỡng khác**. Hiện tại gate chưa include Kappa check — đây là future work (có thể thêm `min_agreement_rate` threshold).

### 2.3 Position Bias ở tầng decision
Position bias thường phát sinh khi dùng pair-wise judge. Pipeline hiện tại chấm điểm absolute (1-5) trên từng response độc lập → position bias không áp dụng. Tuy nhiên ở tầng `_build_v1_v2_compare` có tính `winner = v2 if V2_score >= V1_score else v1` — nếu tương lai chuyển sang pair-wise, cần chèn bias-detection trước khi tính winner rate.

### 2.4 Trade-off Cost vs Quality — cốt lõi Release Gate
Gate là hiện thân số hoá của trade-off:
- **Hạ `GATE_MAX_LATENCY` xuống 2.5s** → có thể PASS baseline nhưng không cho phép cải thiện chất lượng tốn latency → BLOCK V2.
- **Hạ `GATE_MIN_SCORE` xuống 2.0** → dễ release hơn nhưng rủi ro ship product kém chất lượng.
- **Tăng `GATE_MIN_FAITH` lên 0.5** → giảm hallucination rate ship ra nhưng có thể BLOCK phần lớn V2 vì heuristic faithfulness hiện còn đơn giản (token overlap).
Mỗi ngưỡng là 1 business decision được code hóa. Giá trị hiện tại (2.5 / 45% / 40% / 10% / 3.0s / -0.2 / -10% / -10%) là compromise giữa "strict đủ để bảo vệ user" và "lenient đủ để team iterate được".

### 2.5 Cost report
`COST_PER_1K_TOKENS = 0.002` + `estimated_cost_usd = (total_tokens / 1000) × rate`. Hiện `tokens_used` chưa expose từ LLM wrapper (= 0) nên cost hiển thị $0 — đây là **tech debt có ý thức** ghi trong report:
> "Current LLM wrapper does not expose usage" (comment trong `main_agent.py:105`).
Action item đã log: parse `response.usage.total_tokens` của OpenAI và gắn vào metadata → cost report thành metric thật.

---

## 3. Problem Solving (10đ)

### 3.1 Vấn đề: V2 vượt latency budget 4.17s (>3.0s)
Gate `latency_budget` fail liên tục dù chất lượng V2 rõ ràng tốt hơn V1.
**Giải quyết (điều phối chéo team):**
1. Đọc toàn repo, xác định 3 bottleneck: rerank print-spam, LLM max_tokens không giới hạn, retrieval_top_k quá rộng.
2. Phối hợp Phương Anh áp 3 fix: xoá print trong `CrossEncoderRerank`, thêm `max_tokens=160 + temperature=0`, giảm `retrieval_top_k` V2: 10 → 6 → 5.
3. Thêm warm-up CrossEncoder khử cost query đầu.
4. Kết quả: **4.17s → 3.67s → 2.80s** (pass gate).

### 3.2 Vấn đề: Pipeline treo giữa chừng không báo lỗi
Khi chạy V2, script dừng ở một Judge result và không tiếp tục.
**Giải quyết:** Phối hợp Trí và Dũng thêm timeout đa tầng: `AsyncOpenAI(timeout=30.0, max_retries=2)`, `asyncio.wait_for(_score_with_openai, 30s)`, `asyncio.wait_for(agent.query, 60s)`, `asyncio.wait_for(evaluate_multi_judge, 90s)`, tách `asyncio.TimeoutError` riêng. Sau đó pipeline không bao giờ treo quá 90s/case.

### 3.3 Vấn đề: Trade-off V2 score giảm nhẹ (-0.07) nhưng MRR tăng (+0.067)
2 gate "score regression" và "no mrr regression" có thể mâu thuẫn — làm sao justify APPROVE?
**Giải quyết:** Đọc kỹ failure cases trong `v1_v2_comparison.json`, xác nhận `Δscore = -0.07 > GATE_MAX_SCORE_REG = -0.2` → trong tolerance. Viết vào `failure_analysis.md` rằng score giảm đến từ 1-2 case adversarial mà V2 rerank đẩy context nhạy cảm lên top → fix bằng prompt refusal (P0 action) chứ không phải rollback rerank. Đó là *judgement call* của integration lead.

### 3.4 Vấn đề: Failure analysis dễ biến thành boilerplate
Template ban đầu chỉ có placeholder "X case hallucination, Y case incomplete".
**Giải quyết:** Dùng `reports/benchmark_results.json` (thật) + script Python 1-off filter `status=="fail"` → 8 case thật → cluster theo pattern (adversarial 5, retrieval miss 2, incomplete 1) → 5-Whys cho 3 case `final_score ≤ 1.5`. Action Plan gắn trực tiếp case ID → reproducible.

### 3.5 Vấn đề: Integration nhiều file, rủi ro trùng lặp MainAgent
Có 2 file `main_agent.py` và `response_wrapper.py` đều định nghĩa class `MainAgent` — dễ import nhầm.
**Giải quyết:** Xác định `main.py:7` import từ `agent.main_agent` (pipeline thật), `response_wrapper.py` giữ lại như mock cho test runner và được document rõ. Không xoá để không phá work trước đó của team.

---

## 4. Kết quả ở mức decision
- **Decision cuối:** **APPROVE** ✅ — 8/8 gate checks pass sau 3 vòng tuning.
- **Failure analysis** với 3 nhóm lỗi rõ ràng + Action Plan P0/P1/P2 có thể handoff cho sprint tiếp theo.
- **Toàn repo chạy end-to-end** bằng 1 lệnh `python main.py`, output 3 artifact JSON sẵn sàng cho post-processing / CI gate.
- **Documentation** (README, GRADING_RUBRIC) giữ cho teammate mới onboard nhanh.
