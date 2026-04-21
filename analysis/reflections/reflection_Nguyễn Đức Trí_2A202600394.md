# Reflection — Nguyễn Đức Trí (2A202600394)

**Vai trò trong nhóm:** Evaluation Runtime Engineer — xây benchmark runner async điều phối agent + evaluator + judge.
**Module đảm nhận:** `engine/runner.py` (`BenchmarkRunner`).

---

## 1. Engineering Contribution (15đ)

### 1.1 Module cụ thể
`engine/runner.py` — orchestration layer giữa 3 nguồn bất đồng bộ:
- **Agent** (`MainAgent.query`) — BM25 + (optional) rerank + GPT-4o-mini.
- **Expert Evaluator** (`ExpertEvaluator.score`) — tính Hit@K, MRR.
- **LLM Judge** (`LLMJudge.evaluate_multi_judge`) — chấm điểm bằng 2 model GPT.

### 1.2 Điểm kỹ thuật nổi bật
- **Batch-controlled concurrency:** `run_all(dataset, batch_size=5)` chia dataset thành batches, mỗi batch dùng `asyncio.gather(*tasks)`. Batch-size=5 là điểm cân bằng giữa **rate-limit** của OpenAI (tránh `429`) và **throughput** (gấp ~5x so với chạy tuần tự).
- **Structured result schema:** mỗi `run_single_test` trả về object chuẩn với đầy đủ fields cho downstream:
  - `question`, `expected_answer`, `expected_retrieval_ids` — cho `_build_v1_v2_compare`.
  - `agent_response`, `retrieved_ids`, `latency`, `tokens_used`, `agent_version` — cho `_build_summary`.
  - `ragas: {retrieval: {hit_rate, mrr, top_k}}` — từ Dũng.
  - `judge: {final_score, agreement_rate, individual_scores, score_gap, conflict_resolved, faithfulness, relevancy}` — từ Dũng.
  - `status: "pass"/"fail"` — threshold tại `judge.final_score < 3`.
- **Latency tracking per-case:** `time.perf_counter()` ôm toàn bộ `agent.query` → chính là metric đi vào `latency_budget` gate của Release Gate.
- **Logging chi tiết:** `logging.basicConfig(level=INFO)` in `Agent response`, `RAGAS scores`, `Judge result` → hỗ trợ debug và là evidence stream để phân tích failure cases.
- **Robust per-case:** `try/except` xung quanh toàn bộ test case + catch `asyncio.TimeoutError` riêng → 1 case lỗi không crash 49 case khác.

### 1.3 Chứng minh qua Git & output
Chạy 50 × 2 = 100 test cases trong ~11 phút, không có crash cấp batch. `reports/benchmark_results.json` đầy đủ 50 entries với schema đồng nhất → mọi downstream (`main.py`, `failure_analysis.md`, `compute_quick_metrics.py`) parse thẳng.

---

## 2. Technical Depth (15đ)

### 2.1 MRR (Mean Reciprocal Rank)
MRR là trung bình `1/rank` với rank = vị trí đầu tiên match. Tại runner, MRR lấy qua `evaluator.score(case, response)` rồi gắn vào `ragas.retrieval.mrr`. Trong batch-mode, MRR của 50 case tổng hợp bằng trung bình cộng (vì mỗi case contribute 1/rank độc lập, mean trên 50 samples là estimator không bias).

### 2.2 Cohen's Kappa
Kappa đo reliability giữa 2 rater. Trong context runner: judge_a (`gpt-4o`) và judge_b (`gpt-4o-mini`) chấm độc lập. `judge.agreement_rate` trong output là simplification:
`agreement = max(0, 1 - score_gap/4)` với `score_gap = |score_a - score_b|`.
Đây là proxy dễ tính, nhưng *thiếu* thành phần `p_e` (expected agreement). Để tính Kappa thật, cần matrix 2D các (score_a, score_b) trên toàn 50 case → áp `(p_o - p_e) / (1 - p_e)`. Runner hiện log đủ `individual_scores` để hậu kỳ tính Kappa offline.

### 2.3 Position Bias
Position bias xuất hiện khi so sánh 2 response A/B mà judge thiên vị vị trí A. Runner hiện không gọi `check_position_bias` (hàm stub trong `LLMJudge`). Mitigation ở cấp runner: chạy V1 và V2 **tách batch riêng** → judge không bao giờ thấy cả 2 response cùng lúc, nên position bias không xảy ra ở tầng này. Đánh đổi: mất khả năng đo bias trực tiếp, nhưng độc lập hoá 2 đánh giá.

### 2.4 Trade-off Cost vs Quality
- **Batch_size 5 vs 10:** tăng batch nhanh hơn nhưng OpenAI rate-limit dễ bị hit (cost cao hơn qua retry). Batch 5 đủ an toàn với Tier 1.
- **Timeout 60s agent / 90s judge:** cân bằng cho case chậm có cơ hội hoàn thành vs chặn hanging. Timeout quá ngắn → nhiều `{"error": "timeout"}` → sample size hữu ích tụt xuống.
- **Logging INFO vs DEBUG:** INFO đủ cho failure analysis nhưng phát sinh ~150KB/run; DEBUG gấp 10 lần. Tác động nhỏ ở 50 case, nhưng scale 500+ sẽ thành vấn đề I/O.

---

## 3. Problem Solving (10đ)

### 3.1 Vấn đề: Rate limit khi chạy song song 50 case
`asyncio.gather(*[run_single_test(c) for c in dataset])` với 50 concurrent → OpenAI trả `429 Too Many Requests`.
**Giải quyết:** Chia dataset thành chunks `batch_size=5`, chạy tuần tự giữa các batch:
```python
for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i+batch_size]
    ...
```
Throughput giảm ~2-3x so với full-async nhưng không bị 429.

### 3.2 Vấn đề: 1 case hanging làm chết cả batch
`asyncio.gather` chờ TẤT CẢ task xong → nếu 1 OpenAI call hang, 4 case khác bị giữ lại cho tới khi timeout mặc định của client (vài phút).
**Giải quyết:** Bọc `agent.query` trong `asyncio.wait_for(..., timeout=60)` và `evaluate_multi_judge` trong `asyncio.wait_for(..., timeout=90)`. Nếu timeout, case được đánh dấu `{"error": "timeout"}` và pipeline tiếp tục. Log `Timeout khi chạy test case: ...` cho phép truy vết.

### 3.3 Vấn đề: Phân biệt "case lỗi logic" vs "case timeout"
Nếu chỉ có 1 nhánh `except Exception`, timeout và lỗi code sẽ gộp chung.
**Giải quyết:** Tách `except asyncio.TimeoutError` riêng trước `except Exception` → `failure_analysis.md` của Tiền phân biệt được 2 nhóm root cause khác nhau.

### 3.4 Vấn đề: Response không đủ field bắt buộc
Agent trong quá trình dev có lần trả `{"response": ..., "contexts": ...}` thay vì `{"answer": ...}` → evaluator nhận `None`.
**Giải quyết:** Thêm check `if "answer" not in response: return {"error": "Agent response thiếu trường 'answer'"}` ngay sau khi gọi agent → fail-fast thay vì fail-silent downstream.

---

## 4. Kết quả
- 100 test cases (50 × 2 version) chạy ổn định, không crash.
- Tổng thời gian benchmark: ~11 phút — đủ nhanh để iterate nhiều lần trong buổi tuning V2.
- Output schema đồng nhất là foundation cho Release Gate và Failure Analysis downstream.
