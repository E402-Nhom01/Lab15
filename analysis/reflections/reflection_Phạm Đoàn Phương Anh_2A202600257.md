# Reflection — Phạm Đoàn Phương Anh (2A202600257)

**Vai trò trong nhóm:** Agent Engineer — xây dựng agent benchmark và 2 phiên bản V1/V2 để phục vụ regression testing.
**Module đảm nhận:** `agent/main_agent.py`, tích hợp `agent/simple-rag/` (BM25 + CrossEncoder + GPT-4o-mini), tuning latency V2.

---

## 1. Engineering Contribution (15đ)

### 1.1 Module / file cụ thể
| File | Nội dung |
|------|----------|
| `agent/main_agent.py` | Lớp `MainAgent` với 2 phiên bản: V1 (BM25 only, top_k=3) và V2 (BM25 top_k=5 + CrossEncoder rerank top_k=3). Đọc PDF thành page-chunks, init retriever/rerank/LLM, expose `query()` async. |
| `agent/simple-rag/rag/rerank.py` | CrossEncoder rerank (`ms-marco-MiniLM-L-6-v2`) — đã được refactor để tắt log spam và làm sạch logic chọn top-k. |
| `agent/simple-rag/rag/llm.py` | `GPT4oMiniLLM` wrapper, thêm `max_tokens=160`, `temperature=0`, `timeout=30s`, `max_retries=2`. |
| `agent/response_wrapper.py` | Wrapper mock ban đầu (giờ không dùng) giúp Dũng & Trí test runner trước khi pipeline thật sẵn sàng. |

### 1.2 Điểm kỹ thuật nổi bật
- **Sync-to-async bridging:** `MainAgent.query()` dùng `asyncio.to_thread(self._run_sync, question)` để offload BM25 + CrossEncoder (đều là sync, dùng numpy/torch) khỏi event loop — cho phép `runner.py` chạy `asyncio.gather` batch_size=5 mà không bị block.
- **Chuẩn hoá output:** Mọi response đều có `{answer, contexts, metadata{sources, source_documents, latency_sec_agent, retrieval_top_k, rerank_top_k, agent_version}}`. Đây là contract mà `runner.py`, `ExpertEvaluator` (Hit@K, MRR), và `LLMJudge` (faithfulness) đều dựa vào.
- **V1/V2 khác biệt rõ ràng:**
  - V1: baseline đơn giản → `retrieval_top_k=3`, `rerank_top_k=0` → latency thấp (~2.2s), MRR 0.723.
  - V2: thêm rerank → `retrieval_top_k=5`, `rerank_top_k=3` → MRR **+6.7%** (0.77), trade-off latency +0.55s.
- **Warm-up CrossEncoder** ngay trong `__init__` bằng 1 predict dummy → khử cost lazy-init ở query đầu tiên (tiết kiệm ~0.3-0.8s trên query đầu của mỗi batch).

### 1.3 Chứng minh qua Git
Commit `update model to open ai only` (`a4ad14a`) — refactor chuyển từ Ollama local sang OpenAI để đồng bộ với pipeline eval của Dũng. Agent chạy ổn định qua 50 golden cases × 2 versions = 100 runs không crash.

---

## 2. Technical Depth (15đ)

### 2.1 MRR trong context rerank
Rerank của V2 trực tiếp cải thiện MRR: CrossEncoder nhận `[query, passage]` pair và trả về semantic similarity score, sắp xếp lại → `expected_id` có xu hướng nhảy lên vị trí cao hơn. Thực tế: MRR V1 = 0.723 → MRR V2 = 0.770 (Δ +0.047). Đây là bằng chứng số cho thấy rerank "giá trị gia tăng" ngay trên cùng tập retrieved candidates.

### 2.2 Cohen's Kappa (liên quan agent)
Khi đánh giá agent V1 vs V2, nếu 2 judge đều chấm V2 > V1 với Kappa cao, kết luận "V2 tốt hơn" mới vững. Kappa thấp → có thể do 1 judge bị bias bởi văn phong agent. Agent của Phương Anh giữ template trả lời thống nhất (same system prompt) để giảm confound với judge bias.

### 2.3 Position Bias
Với rerank: CrossEncoder rerank lại position của passages → bản thân việc rerank giúp **giảm position bias** của BM25 (BM25 chỉ sort theo lexical score, không hiểu semantic). Nhưng cũng cần cảnh báo: nếu prompt đưa top-3 context theo thứ tự rerank score, LLM có thể thiên về context đầu → cần xem xét shuffle hoặc đánh dấu rõ.

### 2.4 Trade-off Cost vs Quality
| Tiêu chí | V1 | V2 | Nhận xét |
|----------|----|----|----------|
| LLM cost / query | = GPT-4o-mini 1 call | = GPT-4o-mini 1 call (context dài hơn) | V2 tốn ~15-20% token input nhưng cùng model |
| Latency | 2.25s | 2.80s | +24% |
| Score | 3.74 | 3.67 | -0.07 (nhẹ, vẫn pass regression gate) |
| MRR | 0.723 | 0.770 | +6.7% |
| Hit Rate | 0.84 | 0.82 | -0.02 |
Trade-off rõ: V2 hy sinh nhẹ avg_score & hit_rate để đổi lấy MRR tốt hơn và latency vẫn trong budget 3.0s. Gate vẫn APPROVE.

---

## 3. Problem Solving (10đ)

### 3.1 Vấn đề: V2 ban đầu latency 4.17s (vượt budget 3.0s)
**Giải quyết (qua 2 vòng tuning):**
1. Tắt print-spam trong `CrossEncoderRerank.rerank` (~30 dòng/query × 50 query).
2. Thêm `max_tokens=160` và `temperature=0` cho GPT-4o-mini → giảm độ dài output.
3. Giảm `retrieval_top_k` V2 từ 10 → 5 (ít candidate cho rerank → nhanh hơn).
4. Warm-up CrossEncoder trong `__init__` bằng `self.rerank.model.predict([["warmup", "warmup"]])`.
Kết quả: **4.17s → 3.67s → 2.80s** (đạt gate).

### 3.2 Vấn đề: LLM call đôi khi treo không timeout
Khi chạy 50 case × 2 version, thỉnh thoảng 1 API call bị hang → `asyncio.gather` đợi vô hạn → cả batch chết.
**Giải quyết:** Thêm `OpenAI(api_key=..., timeout=30.0, max_retries=2)` trong `GPT4oMiniLLM.__init__` → request treo tự fail sau 30s và retry 2 lần trước khi bỏ.

### 3.3 Vấn đề: Output fallback khi reranker init lỗi
Nếu `sentence-transformers` chưa cài hoặc không tải được model, V2 sẽ crash ngay lúc init → cả benchmark ngừng.
**Giải quyết:** `try/except` quanh `CrossEncoderRerank(...)`, gán `self.rerank = None` và log warning. Hàm `_run_sync` check `if self.rerank and docs:` trước khi gọi rerank → V2 degrade gracefully về giống V1 nếu reranker lỗi.

### 3.4 Vấn đề: Page-level chunking mất thông tin ngắn
Page-level chunking khiến BM25 phải score page dài 500+ từ cho query ngắn → noise cao. Nhưng sub-chunking sẽ buộc đổi `expected_retrieval_ids` → không tương thích với golden set của Bảo.
**Giải quyết:** Giữ page-level để tương thích, bù lại bằng rerank semantic của CrossEncoder ở V2 → vừa không phá contract, vừa cải thiện retrieval quality.

---

## 4. Kết quả
V2 đạt **APPROVE** qua tất cả 8 gate checks:
- Latency 2.80s ≤ 3.0s ✅
- MRR +6.67% (không hồi quy) ✅
- Faithfulness 27.10% ≥ 10% ✅
Agent chạy ổn định 50 × 2 = 100 test cases không có case lỗi do agent side.
