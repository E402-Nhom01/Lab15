# 🚀 Lab Day 14: AI Evaluation Factory (Team Edition)

## 🎯 Tổng quan
"Nếu bạn không thể đo lường nó, bạn không thể cải thiện nó." — Nhiệm vụ của nhóm bạn là xây dựng một **Hệ thống đánh giá tự động** chuyên nghiệp để benchmark AI Agent. Hệ thống này phải chứng minh được bằng con số cụ thể: Agent đang tốt ở đâu và tệ ở đâu.

---

## 🕒 Lịch trình thực hiện (4 Tiếng)
- **Giai đoạn 1 (45'):** Thiết kế Golden Dataset & Script SDG. Tạo ra ít nhất 50 test cases chất lượng.
- **Giai đoạn 2 (90'):** Phát triển Eval Engine (RAGAS, Custom Judge) & Async Runner.
- **Giai đoạn 3 (60'):** Chạy Benchmark, Phân cụm lỗi (Failure Clustering) & Phân tích "5 Whys".
- **Giai đoạn 4 (45'):** Tối ưu Agent dựa trên kết quả & Hoàn thiện báo cáo nộp bài.

---

## 🛠️ Các nhiệm vụ chính (Expert Mission)

### 1. Retrieval & SDG (Nhóm Data)
- **Retrieval Eval:** Tính toán Hit Rate và MRR cho Vector DB. Bạn phải chứng minh được Retrieval stage hoạt động tốt trước khi đánh giá Generation.
- **SDG:** Tạo 50+ cases, bao gồm cả Ground Truth IDs của tài liệu để tính Hit Rate.

### 2. Multi-Judge Consensus Engine (Nhóm AI/Backend)
- **Consensus logic:** Sử dụng ít nhất 2 model Judge khác nhau. 
- **Calibration:** Tính toán hệ số đồng thuận (Agreement Rate) và xử lý xung đột điểm số tự động.

### 3. Regression Release Gate (Nhóm DevOps/Analyst)
- **Delta Analysis:** So sánh kết quả của Agent phiên bản mới với phiên bản cũ.
- **Auto-Gate:** Viết logic tự động quyết định "Release" hoặc "Rollback" dựa trên các chỉ số Chất lượng/Chi phí/Hiệu năng.

---

## 📤 Danh mục nộp bài (Submission Checklist)
Nhóm nộp 1 đường dẫn Repository (GitHub/GitLab) chứa:
1. [ ] **Source Code**: Toàn bộ mã nguồn hoàn chỉnh.
2. [ ] **Reports**: File `reports/summary.json` và `reports/benchmark_results.json` (được tạo ra sau khi chạy `main.py`).
3. [ ] **Group Report**: File `analysis/failure_analysis.md` (đã điền đầy đủ).
4. [ ] **Individual Reports**: Các file `analysis/reflections/reflection_[Tên_SV].md`.

---

## 🏆 Bí kíp đạt điểm tuyệt đối (Expert Tips)

### ✅ Đánh giá Retrieval (15%)
Nhóm nào chỉ đánh giá câu trả lời mà bỏ qua bước Retrieval sẽ không thể đạt điểm tối đa. Bạn cần biết chính xác chunk nào đang gây ra lỗi Hallucination.

### ✅ Multi-Judge Reliability (20%)
Việc chỉ tin vào một Judge (ví dụ GPT-4o) là một sai lầm trong sản phẩm thực tế. Hãy chứng minh hệ thống của bạn khách quan bằng cách so sánh nhiều Judge model và tính toán độ tin cậy của chúng.

### ✅ Tối ưu hiệu năng & Chi phí (15%)
Hệ thống Expert phải chạy cực nhanh (Async) và phải có báo cáo chi tiết về "Giá tiền cho mỗi lần Eval". Hãy đề xuất cách giảm 30% chi phí eval mà không giảm độ chính xác.

### ✅ Phân tích nguyên nhân gốc rễ (Root Cause) (20%)
Báo cáo 5 Whys phải chỉ ra được lỗi nằm ở đâu: Ingestion pipeline, Chunking strategy, Retrieval, hay Prompting.

---

## 🔧 Hướng dẫn chạy

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Tạo Golden Dataset (chạy trước khi benchmark)
python data/synthetic_gen.py

# 3. Chạy Benchmark & tạo reports
python main.py

# 4. Kiểm tra định dạng trước khi nộp
python check_lab.py

# 5. Tính nhanh metrics V1 vs V2 (hit_rate / accuracy / hallucination_rate)
python analysis/compute_quick_metrics.py
```

---

## ⚠️ Lưu ý quan trọng
- **Bắt buộc** chạy `python data/synthetic_gen.py` trước để tạo file `data/golden_set.jsonl`. File này không được commit sẵn trong repo.
- Trước khi nộp bài, hãy chạy `python check_lab.py` để đảm bảo định dạng dữ liệu đã chuẩn. Bất kỳ lỗi định dạng nào dẫn đến việc script chấm điểm tự động không chạy được sẽ bị trừ 5 điểm thủ tục.
- File `.env` chứa API Key **KHÔNG** được push lên GitHub.

---
*Chúc nhóm bạn xây dựng được một Evaluation Factory thực sự mạnh mẽ!*


## ✅ Checklist Lab14 — Các bước cần làm theo hướng dẫn giáo viên
**Mục tiêu cuối cùng:** Chứng minh bằng benchmark rằng **Version 2 tốt hơn Version 1**.

### PHASE 1 — DATASET (Quan trọng nhất)
1. **Chuẩn bị source data**
   - Document gốc, knowledge base, vector DB (nếu có), chunk text, chunk ID.
   - Nếu đã có vector DB: export chunk ra để đối chiếu ground truth.
2. **Chunk dữ liệu**
   - Mỗi chunk có: `chunk_id`, `chunk_text`, `source_document`.
3. **Thiết kế prompt tạo dataset**
   - Bắt buộc sinh: question, expected answer, correct chunk ID, difficulty, category, metadata.
   - Bắt buộc có Good Example + Hard Case Example.
4. **Sinh Golden Dataset bằng LLM**
   - Mục tiêu 30–50 câu, gồm easy/medium/hard, multi-hop, retrieval dễ sai, hallucination dễ xảy ra.
5. **Manual review dataset (bắt buộc)**
   - Kiểm tra: question đúng, answer đúng, chunk ID đúng, source đúng.

### PHASE 2 — AGENT VERSION
6. **Version 1 (baseline)**
   - Retrieval đơn giản (ví dụ BM25), logic cũ, prompt chưa tối ưu.
7. **Version 2 (nâng cấp rõ ràng)**
   - Retrieval tốt hơn + rerank + prompt tốt hơn + answer synthesis tốt hơn.
   - Kỳ vọng: `V2 > V1`.

### PHASE 3 — TRUST / JUDGE
8. **Xây LLM Judge**
   - Chấm: correct/incorrect, partial correct, hallucination, bias/fairness, consistency.
9. **Verify Judge**
   - Manual spot-check để tránh lỗi do chính Judge LLM.

### PHASE 4 — BENCHMARK
10. **Chạy benchmark cho V1** và lưu kết quả.
11. **Chạy benchmark cho V2** trên cùng dataset để so sánh công bằng.
12. **Tính metric**
   - Retrieval Accuracy, Hit Rate, Avg Hit Rate, Final Answer Accuracy, Hallucination Rate, Avg Score, Latency, Cost, User Satisfaction.

### PHASE 5 — ANALYSIS
13. **Phân tích nguyên nhân**
   - Không chỉ nói “V2 tốt hơn”, mà phải chỉ ra **tốt ở đâu, vì sao tốt hơn, rủi ro còn lại**.

### PHASE 6 — REPORT
14. **Final report**
   - Executive Summary, Benchmark Comparison, Metric Table, Trust Analysis, Risk Analysis, Recommendation, Next Action.

**Final Deliverable:** Dataset + Agent V1 + Agent V2 + LLM Judge + Benchmark Results + Final Report.

---

## ⚡ Minimal plan để pass nhanh (khi thiếu thời gian)
1. **Dataset (1–2 giờ):** tạo 30 câu, manual fix 10 câu quan trọng.
2. **V1 đơn giản:** BM25, `top_k=3`, không rerank.
3. **V2 nâng cấp rõ:** `top_k=10`, rerank top 3, prompt tốt hơn.
4. **Run benchmark** trên cùng dataset.
5. **Tính metrics chính:** `hit_rate`, `accuracy`, `hallucination_rate`.
6. **Viết analysis (rất quan trọng):** giải thích trade-off (quality ↑, latency/cost ↑ nhẹ nhưng chấp nhận được).

### Logging format khuyến nghị cho mỗi sample
```json
{
  "question": "...",
  "expected_answer": "...",
  "ground_truth_chunk_ids": ["chunk_001"],
  "v1_answer": "...",
  "v1_retrieved_chunk_ids": ["chunk_003", "chunk_001"],
  "v2_answer": "...",
  "v2_retrieved_chunk_ids": ["chunk_001", "chunk_010", "chunk_003"],
  "judge": {
    "winner": "v2",
    "v1_correct": false,
    "v2_correct": true,
    "hallucination_v1": true,
    "hallucination_v2": false,
    "reason": "V2 bám đúng chunk ground truth và trả lời đầy đủ hơn"
  },
  "latency_ms": {"v1": 820, "v2": 1210},
  "cost_usd": {"v1": 0.0019, "v2": 0.0034}
}
```

### Analysis mẫu (ngắn gọn, đúng trọng tâm)
- V2 cải thiện retrieval nên hit rate cao hơn.
- Reranker giảm chunk nhiễu nên accuracy tăng.
- Hallucination giảm nhờ prompt tốt hơn + context liên quan hơn.
- Latency tăng nhẹ do top_k lớn hơn và thêm rerank, nhưng vẫn chấp nhận được.
