# Reflection — Huỳnh Thái Bảo (2A202600373)

**Vai trò trong nhóm:** Data Engineer — phụ trách toàn bộ pipeline dữ liệu đánh giá.
**Module đảm nhận:** `data/synthetic_gen.py`, `data/HARD_CASES_GUIDE.md`, `data/golden_set.jsonl` (50 cases), `data/sample.pdf`.

---

## 1. Engineering Contribution (15đ)

### 1.1 Module / file cụ thể đã viết
| File | Nội dung |
|------|----------|
| `data/synthetic_gen.py` | Script sinh synthetic golden set bằng GPT-4o ở chế độ JSON. Gồm `extract_chunks_from_pdf()` (chia PDF theo page → gắn `chunk_id`) và `generate_qa_from_text()` (gọi OpenAI async, batch 10 pairs/lần). |
| `data/HARD_CASES_GUIDE.md` | Taxonomy 4 nhóm Hard Cases: Adversarial Prompts, Edge Cases, Multi-turn Complexity, Technical Constraints. Làm design-doc cho việc ra đề. |
| `data/golden_set.jsonl` | 50 test cases chuẩn hoá JSONL, mỗi case có `question`, `expected_answer`, `expected_retrieval_ids`, `metadata.difficulty`, `metadata.type`. |
| `data/sample.pdf` | PDF tài liệu nguồn (Ollama docs) làm ground-truth knowledge base. |

### 1.2 Điểm kỹ thuật nổi bật
- **JSON-mode generation:** Dùng `response_format={"type": "json_object"}` để buộc GPT-4o trả về JSON hợp lệ → parse bằng `json.loads` không sợ vỡ format, rất quan trọng khi sinh 50 case mà không cần retry thủ công.
- **Binding chunk_id → retrieval eval:** Mỗi chunk được gắn `[Chunk ID: page_X]` trước khi đưa vào prompt; model buộc phải tham chiếu đúng ID → cho phép downstream tính Hit@K và MRR một cách khách quan.
- **Batching an toàn:** Chia `num_pairs=50` thành các batch 10 với `num_batches = (num_pairs + batch_size - 1) // batch_size` — tránh vượt `max_tokens=4000` và giảm rủi ro 1 lỗi parse JSON làm mất toàn bộ dữ liệu.
- **Temperature 0.8:** Cố ý đẩy variance để bộ test phủ nhiều góc độ (factual, adversarial, ambiguous) thay vì lặp lại cùng một dạng.

### 1.3 Chứng minh qua Git
Commit tiêu biểu: `add data cua Bao` (`9f11b80`). Artifact `data/golden_set.jsonl` được `engine/runner.py` và `main.py:213` đọc trực tiếp qua `json.loads(line)`.

---

## 2. Technical Depth (15đ)

### 2.1 MRR (Mean Reciprocal Rank)
MRR = trung bình của `1/rank` với `rank` là vị trí (1-indexed) đầu tiên mà một `expected_retrieval_id` xuất hiện trong danh sách `retrieved_ids`. Nếu không xuất hiện, contribution = 0.
Ý nghĩa với **data**: MRR chỉ có nghĩa khi `expected_retrieval_ids` chính xác và không trùng lặp. Việc gán `page_X` cho từng chunk (không chunk nhỏ hơn) là một design choice — đơn giản hoá ground-truth nhưng làm MRR khắt khe hơn (retrieval phải trúng đúng page đó).

### 2.2 Cohen's Kappa (nối vào phần data)
Kappa đo agreement giữa 2 annotator, trừ đi expected-agreement do ngẫu nhiên:
`κ = (p_o − p_e) / (1 − p_e)`, với `p_o` = tỉ lệ đồng thuận thực tế, `p_e` = tỉ lệ đồng thuận kỳ vọng ngẫu nhiên.
Ứng dụng vào bộ data: nếu 2 người review 50 case và gán nhãn `pass/fail`, Kappa cho biết agreement là "thật" hay do cùng thiên vị. Mục tiêu của bộ hard-case là tránh "ai cũng pass" — Kappa ≥ 0.6 chứng tỏ câu hỏi có mức phân biệt rõ ràng.

### 2.3 Trade-off Cost vs Quality
- Dùng **GPT-4o** (không phải 4o-mini) cho sinh data vì **chất lượng ground truth** là nền móng của mọi metric sau này; sai ở đây không phát hiện được bằng test downstream.
- **Nhưng** chỉ chạy 1 lần để tạo `golden_set.jsonl` rồi commit — chi phí 1 lần, không phát sinh theo mỗi benchmark run.
- Đánh đổi: cost cao cho sinh data (~$0.5-1 cho 50 cases) nhưng tiết kiệm cho chi phí đánh giá 50 cases × mỗi iteration V1/V2 ở judge stage.

### 2.4 Position Bias (liên quan thiết kế data)
Position bias = LLM có xu hướng thiên vị câu trả lời ở vị trí A hoặc B. Với vai trò data, cần đảm bảo `expected_answer` không gợi ý vị trí (không luôn đặt thông tin quan trọng ở đầu chunk). Split theo page khiến thông tin phân bố tự nhiên, không bias vị trí.

---

## 3. Problem Solving (10đ)

### 3.1 Vấn đề: Golden set rỗng do path hard-coded
File `synthetic_gen.py:102` hard-code `C:\\assignments-main\\...`. Trên macOS của team, script không đọc được PDF → `golden_set.jsonl` rỗng.
**Giải quyết:** Tạm thời sinh data trên máy Windows rồi commit `golden_set.jsonl` để cả team dùng chung. Action item: refactor dùng `Path(__file__).resolve().parent / "sample.pdf"`.

### 3.2 Vấn đề: Một số case `expected_retrieval_ids = []`
"Edge Case — Out of Context" không có page nào chứa đáp án. Nếu để trống, công thức Hit@K / MRR có thể tính sai.
**Giải quyết:** Phối hợp với Dũng — `RetrievalEvaluator.calculate_hit_rate` dùng `any(...)` trả 0.0 khi `expected_ids` rỗng, `calculate_mrr` trả 0.0 khi không match. Các case này đóng vai trò test khả năng **refuse** của agent thay vì retrieval.

### 3.3 Vấn đề: Cần mix "hard" và "căn bản"
Nếu 100% adversarial thì không đánh giá được baseline retrieval. Nếu 100% factual thì không stress-test robustness.
**Giải quyết:** Trong prompt generation, liệt kê 4 categories với tỉ lệ ngầm và để GPT-4o tự phân bổ. Kết quả thực tế: ~15 adversarial/edge case chiếm đúng nhóm fail lớn nhất trong failure analysis → bộ data phân biệt được V1 vs V2 ở đúng trục quan trọng.

---

## 4. Kết quả liên quan trực tiếp
Bộ 50 golden cases là đầu vào duy nhất của toàn pipeline benchmark. Nhờ phân bố đa dạng:
- Avg score V2: **3.67 / 5** (tách biệt rõ pass/fail).
- Hit Rate: **0.82** — đủ cao để retrieval có nghĩa, đủ thấp để thấy room-for-improvement.
- 8 case fail cluster rõ thành 3 nhóm (adversarial / retrieval miss / incomplete) → bộ test không bị "đồng phục".
