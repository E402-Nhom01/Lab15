# Reflection — Nguyễn Đức Dũng (2A202600148)

**Vai trò trong nhóm:** Evaluation Metrics Engineer — xây dựng toàn bộ tầng đánh giá chất lượng output.
**Module đảm nhận:** `engine/llm_judge.py` (Multi-Judge), `engine/retrieval_eval.py` (MRR/Hit@K).

---

## 1. Engineering Contribution (15đ)

### 1.1 Module / file cụ thể
| File | Nội dung |
|------|----------|
| `engine/llm_judge.py` | `LLMJudge` với Multi-Judge (GPT-4o + GPT-4o-mini), rubric 3 chiều (accuracy/safety/tone), `evaluate_multi_judge`, `calculate_faithfulness`, `calculate_relevancy`, `_fallback_score`, stub `check_position_bias`. |
| `engine/retrieval_eval.py` | `RetrievalEvaluator` với `calculate_hit_rate` (top-k match), `calculate_mrr` (1/rank), `evaluate_batch` (tổng hợp per-case). |

### 1.2 Điểm kỹ thuật nổi bật
- **Multi-Judge song song:** `asyncio.gather(self._score_with_model(model_a), self._score_with_model(model_b))` — 2 judge chấm đồng thời, latency Judge = max(2 call) thay vì sum.
- **Rubric 3-tiêu-chí có weighting:** `accuracy 0.6 + safety 0.3 + tone 0.1`. Buộc judge tính điểm weighted rồi làm tròn về int [1..5] → output nhất quán, dễ đối chiếu.
- **Conflict resolution logic:** khi `score_gap > 1`, chọn `min(score_a, score_b)` thay vì trung bình → *conservative-bias* để giảm risk over-score khi 2 judge bất đồng lớn.
- **Fallback score không phụ thuộc network:** `_fallback_score` dùng heuristic token-overlap (`0.7 × faithfulness + 0.3 × relevancy`) → nếu OpenAI API key thiếu hoặc request lỗi, pipeline vẫn chạy được để debug.
- **Structured output:** return dict giàu `final_score`, `agreement_rate`, `individual_scores`, `score_gap`, `conflict_resolved`, `faithfulness`, `relevancy` → cả Release Gate (main.py) và Failure Analysis đều dùng trực tiếp.

### 1.3 Chứng minh qua Git
Module được `engine/runner.py` gọi trên 100% test cases. Output ổn định, không có case bị `agreement_rate = None`. `faithfulness` (mean 0.27) là metric **quyết định** gate `faithfulness_floor ≥ 10%` của Release Gate — trực tiếp do code của Dũng tính ra.

---

## 2. Technical Depth (15đ)

### 2.1 MRR (Mean Reciprocal Rank)
```python
for i, doc_id in enumerate(retrieved_ids):
    if doc_id in expected_ids:
        return 1.0 / (i + 1)
return 0.0
```
- `i + 1` để chuyển từ 0-indexed (Python) sang 1-indexed (MRR định nghĩa).
- Trả 0.0 khi không match — tương đương rank = ∞.
- MRR đo "chất lượng **vị trí** của kết quả đầu tiên đúng", khác Hit@K chỉ đo "có match hay không". Kết quả thực: V1 MRR 0.723 → V2 MRR 0.770 (+6.7%) — rerank đẩy đáp án đúng lên cao hơn trong top-k.

### 2.2 Hit@K (Top-k Hit Rate)
`any(doc_id in retrieved_ids[:top_k] for doc_id in expected_ids)` → 1.0 nếu ít nhất 1 expected_id nằm trong top-k, ngược lại 0.0.
Hit@K trên tập 50 cases: V1 = 0.84, V2 = 0.82 — có giảm nhẹ vì V2 rerank khiến 1-2 case fall-back khỏi top-3. Nhận xét: Hit@K "coarse" hơn MRR; hai chỉ số đi kèm cho bức tranh đầy đủ.

### 2.3 Cohen's Kappa
Kappa đo reliability giữa 2 rater beyond chance: `κ = (p_o − p_e) / (1 − p_e)`. Trong module hiện tại, `agreement_rate = max(0, 1 − score_gap/4)` là approximation nhanh, KHÔNG có thành phần `p_e`.
Nếu muốn tính Kappa thật: gom tất cả cặp `(score_a, score_b)` trên 50 case → ma trận confusion 5×5 → tính `p_o` từ đường chéo, `p_e` từ marginals, áp công thức. Module đã expose `individual_scores: {gpt-4o, gpt-4o-mini}` trong từng kết quả nên hậu kỳ chạy một script Kappa rất dễ. Agreement rate hiện đạt **0.905** trên 50 case — thể hiện 2 judge đồng thuận cao; Kappa thực có thể cao tương đương vì phân bố score không quá lệch về 1 giá trị.

### 2.4 Position Bias
- **Định nghĩa:** LLM Judge có xu hướng thiên về response ở vị trí A hoặc B khi so sánh pair, không phụ thuộc chất lượng thực.
- **Detection pattern chuẩn:** đưa cặp (A, B) rồi hoán đổi (B, A) — nếu judge vẫn chọn position đầu tiên → bias.
- **Trạng thái code:** `LLMJudge.check_position_bias` hiện là stub. Design ban đầu: thực hiện A/B swap + so sánh tỉ lệ A_win. Lý do chưa implement: trong lab hiện tại, V1 và V2 được judge **độc lập** (không so pair-wise) → position bias không áp dụng. Nếu chuyển sang pair-wise comparison (e.g., head-to-head V1 vs V2), stub này là hook sẵn sàng.

### 2.5 Trade-off Cost vs Quality
| Khía cạnh | Single Judge | Multi-Judge (hiện tại) |
|-----------|--------------|------------------------|
| Cost | 1x API call | 2x API call |
| Reliability | Phụ thuộc 1 model | Có agreement_rate để đo noise |
| Latency | ~1s | ~1s (song song, không cộng) |
- Multi-Judge **gấp đôi cost** so với single nhưng cho phép phát hiện disagreement → rất đáng nếu target là production (reliability quan trọng hơn tiết kiệm vài xu).
- Conservative-bias (min khi conflict) tăng robustness nhưng có thể làm `avg_score` thấp hơn thực tế → trade-off giữa "không over-claim" và "không bỏ sót câu trả lời tốt".

---

## 3. Problem Solving (10đ)

### 3.1 Vấn đề: Judge thỉnh thoảng trả text có kèm giải thích
Dù prompt yêu cầu "chỉ trả về một số nguyên", GPT đôi khi vẫn kèm câu giải thích ("The answer is 4 because...").
**Giải quyết:** Dùng regex `re.search(r"[1-5]", text)` để bắt **số đầu tiên** trong [1..5] trong output → bỏ qua phần thừa. Nếu không match, fallback sang heuristic.

### 3.2 Vấn đề: Không có API key khi dev local
Nếu thiếu `OPENAI_API_KEY`, toàn bộ pipeline eval chết.
**Giải quyết:** `self.openai_client = AsyncOpenAI(...) if os.getenv(...) else None`, và `_fallback_score` dùng heuristic token-overlap → dev local hoặc CI không có key vẫn chạy được, chỉ là score kém chính xác.

### 3.3 Vấn đề: Judge timeout/hang
OpenAI responses API đôi khi hang.
**Giải quyết:** `AsyncOpenAI(api_key=..., timeout=30.0, max_retries=2)` + bọc call trong `asyncio.wait_for(..., timeout=30)` → nếu treo quá 30s, exception trả về và `_score_with_model` fallback sang heuristic. Không để 1 judge call kéo cả batch.

### 3.4 Vấn đề: Cohen's Kappa / Position Bias dễ bị hiểu sai
Các bạn trong nhóm ban đầu định dùng `agreement_rate` như Kappa.
**Giải quyết:** Đặt rõ trong code comment rằng đây là proxy (dựa trên `score_gap`), không phải Kappa chuẩn. Ghi lại `individual_scores` để tính Kappa offline khi cần. Tránh misreport trong báo cáo.

### 3.5 Vấn đề: Hàm `calculate_relevancy` phụ thuộc question length
Nếu question rất dài (nhiều token), overlap dễ đạt → score inflated.
**Giải quyết:** Chuẩn hoá bằng `set(...)` và `|overlap| / |question_tokens|` thay vì raw count → metric nằm trong [0..1] không phụ thuộc độ dài tuyệt đối. Pipeline công nhận tính này — relevancy V2 đạt 0.81 trên 50 cases (ổn định, không phân tán).

---

## 4. Kết quả
- **Faithfulness** trung bình V2 = 0.271 → ≥ 10% gate PASS ✅ (do code Dũng tính).
- **Agreement rate** 0.905 → 2 judge đồng thuận rất cao, tin cậy.
- **MRR** V2 = 0.77 → +6.7% so V1 → gate `no_mrr_regression` PASS ✅.
- **Hit@K** V2 = 0.82 → gate `hit_rate_floor ≥ 45%` PASS ✅.
Cả 4 metric quyết định trực tiếp Release Gate đều sản xuất từ module của Dũng.
