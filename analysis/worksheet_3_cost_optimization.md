# Worksheet 3 — Cost Optimization Debate

> **Scenario:** University Admission & Student Support Agent (Trường ĐH F)
> **Hoạt động 4 (10:15–10:50)** | Nhóm: Tiền (Product) · Phương Anh (Presenter) · Bảo (Cost) · Dũng (Architect) · Thức (Reliability)
> **Mục tiêu:** chọn đúng chiến lược tối ưu cho hệ thống Admission Agent, không tối ưu theo phong trào.

---

## 0. Cơ sở ra quyết định (lấy từ Worksheet 1 + 2)

**Cost driver đã xác định ở Worksheet 2:**
1. **Token API top funnel** — câu hỏi lặp lại rất nhiều: ngành, điểm chuẩn, học phí, deadline.
2. **Human review** — bắt buộc vì sai học phí/quy chế = khủng hoảng truyền thông.
3. **Logging / audit / privacy** — bắt buộc do có PII (CCCD, điểm thi, học bạ).

**Cơ cấu cost token hiện tại ($165/tháng):**

| Layer | % traffic | Cost/tháng | % tổng cost token |
|---|---|---|---|
| GPT-5.4 mini (primary) | 80% | $36.41 | 22% |
| **GPT-5.4 (escalation)** | **15%** | **$113.77** | **69%** |
| Gemini 3.1 (fallback) | 5% | ~$15.00 | 9% |

**Quan sát then chốt:**
- Layer escalation 15% đang "ăn" **69%** cost token → đòn bẩy lớn nhất nằm ở đây.
- Traffic spike mùa cao điểm × câu hỏi lặp lại (top 100 FAQ) → cache hit rate kỳ vọng rất cao.
- Đã chọn **Cloud (Worksheet 1)** với lý do pay-as-you-go → chiến lược self-hosted GPU mâu thuẫn với quyết định này.
- Faithfulness / không hallucinate học phí là ràng buộc cứng → **không** được cắt context bừa.

---

## 1. Đánh giá 5 chiến lược thông dụng

| # | Chiến lược | Tiết kiệm phần nào | Lợi ích | Trade-off | Khi nào áp dụng |
|---|---|---|---|---|---|
| A | **Semantic caching** | Token API lặp lại ở top funnel | Cắt 30–50% call mini, giảm latency cho peak hour | Stale cache khi đổi học phí/quy chế → sai nguy hiểm; phải bind TTL vào version KB | **Ngay** — cost driver #1 + spike pattern |
| B | **Model routing — siết policy escalation** | Giảm tỉ lệ escalation 15% → ~8% | Cắt ~40% cost layer GPT-5.4 (layer đang chiếm 69%) | Misrouting → câu adversarial/PII lọt xuống mini → brand risk | **Ngay** — đòn bẩy lớn nhất |
| C | **Selective inference / phân tầng user-request** | Compute + human review ở đường realtime | Tách UI Cán bộ (async/batch) khỏi UI Thí sinh (realtime); giảm tải peak | Cần queue + worker, kiến trúc phức tạp thêm; cần thống nhất SLA mỗi UI | **Sau** — phụ thuộc Product chốt SLA từng UI |
| D | **Prompt compression** | Token input của RAG context | Nhỏ: tiết kiệm ~10–15% | Context 2500 token đang cố tình dày để **không hallucinate học phí** — cắt = tăng risk | **Không làm** ở giai đoạn này |
| E | **Smaller / self-hosted model** | Đơn giá/token | 0 ở volume hiện tại (peak 12k req/ngày, 26/30 ngày nhàn rỗi) | CapEx GPU + vận hành; mâu thuẫn quyết định Cloud ở Worksheet 1 | **Chưa** — volume chưa đạt breakeven |

---

## 2. Chi tiết 3 chiến lược được chọn

### ⭐ Chiến lược 1 — Semantic Caching *(làm ngay · P0)*
**Owner:** Cost Lead (Bảo) + Architect (Dũng)

- **Tiết kiệm phần nào:** toàn bộ call LLM cho các câu hỏi top funnel lặp lại. Key cache = `embed(question)` + similarity ≥ 0.92 + `kb_version`.
- **Lợi ích cụ thể:**
  - Top 100–200 câu FAQ (ngành CNTT bao nhiêu điểm, học phí, deadline, phương thức xét tuyển) ước tính cover **40–60% lượng câu hỏi UI Thí sinh**.
  - Cache hit → trả lời < 200ms, giúp giải quyết luôn vấn đề "phản hồi ngay" của enterprise constraint.
  - Giảm tải đỉnh khi spike (2 tuần trước deadline): spike chính là lúc câu hỏi **giống nhau** nhất.
  - Ước tính cắt **~$15–20/tháng** token mini + nhẹ tải human review ở câu lặp.
- **Trade-off & cách kiểm soát:**
  - **Rủi ro stale cache** → thông tin học phí/điểm sai = khủng hoảng. Kiểm soát bằng:
    - TTL ngắn (24h) cho thông tin số (học phí, điểm chuẩn).
    - Cache key gắn `kb_version` → mỗi đợt cập nhật quy chế, cache bị invalidate tự động.
    - Whitelist chỉ cache câu thuộc nhóm FAQ public; **không cache** câu liên quan hồ sơ cá nhân / tra cứu PII.
- **Khi áp dụng:** tuần này, trước khi vào mùa cao điểm.

### ⭐ Chiến lược 2 — Model Routing: siết chặt chính sách escalation *(làm ngay · P0)*
**Owner:** Architect (Dũng) + Cost Lead (Bảo); Reliability (Thức) ký off rule-set

- **Tiết kiệm phần nào:** layer **GPT-5.4 escalation** đang là 69% cost token → mục tiêu giảm tỉ lệ escalation từ **15% → 8%**.
- **Logic đề xuất (không phải làm mới routing, mà là siết rule):**
  1. **Intent classifier rẻ** (mini hoặc rule-based) phân câu hỏi thành `{simple_faq, personal_record, compliance/adversarial, complex_reasoning}`.
  2. Chỉ escalate GPT-5.4 khi:
     - Câu thuộc nhóm `compliance/adversarial` (quy chế, pháp lý, khiếu nại).
     - Câu cần reasoning đa bước trên hồ sơ cá nhân (xác suất đỗ, tư vấn ngành).
     - Mini trả về confidence thấp hoặc output không trích được nguồn.
  3. Các câu còn lại (FAQ, timeline, checklist) giữ ở mini.
- **Lợi ích dự kiến:**
  - Escalation 15% → 8% ⇒ cost GPT-5.4/tháng từ $113.77 xuống ~$60 → **tiết kiệm ~$50–55/tháng**.
  - Cộng hưởng: câu ở mini **nhanh hơn** → tốt cho UX ở peak.
- **Trade-off & cách kiểm soát:**
  - Misrouting = sai ở câu compliance → **không chấp nhận được**. Kiểm soát bằng:
    - **Whitelist escalate bắt buộc**: mọi câu match keyword {quy chế, khiếu nại, pháp lý, CCCD, điểm thi THPT} → escalate không điều kiện.
    - Offline eval trên golden set 50 case trước khi deploy rule mới.
    - Log `routing_decision` để Reliability audit hàng tuần.
- **Khi áp dụng:** song song với Chiến lược 1. Đây là **đòn bẩy cost lớn nhất** vì đánh trúng layer chiếm 69%.

### Chiến lược 3 — Selective Inference / phân tầng user-request *(làm sau · P1)*
**Owner:** Product Lead (Tiền) chốt SLA, Architect (Dũng) thiết kế queue

- **Ý tưởng:** 3 UI có SLA khác nhau rõ rệt → không nên dùng chung một pipeline sync.
  - **UI Thí sinh** (realtime chat, latency-critical) → synchronous, model mini, ưu tiên cache.
  - **UI SV năm nhất** (notification, checklist, reminder) → đa phần **async**, batch delivery → không tốn compute peak.
  - **UI Cán bộ** (dashboard, text-to-SQL, funnel analytics) → **async + aggregate cache daily**, không cần tính real-time mỗi lần refresh.
- **Lợi ích dự kiến:**
  - Giảm tải compute giờ peak (tránh cán bộ query dashboard đúng lúc thí sinh nộp hồ sơ).
  - Tiết kiệm ~20% compute tổng (Worksheet 2 đang ước $40–120 compute → save ~$15–25).
  - Tác động phụ: giảm human review ở những output non-critical (dashboard không cần review từng lần).
- **Trade-off:**
  - Cần thêm **queue + worker** → kiến trúc phức tạp hơn.
  - Cần Product chốt rõ: **ai được chờ bao lâu?** Nếu cán bộ kỳ vọng realtime analytics, không áp được async.
- **Vì sao làm sau:** phụ thuộc **thống nhất SLA từng UI** (việc của Product), và trùng effort với intent classifier ở Chiến lược 2 → làm nối tiếp sẽ tối ưu hơn làm song song.

---

## 3. Chốt thứ tự thực hiện

| Ưu tiên | Chiến lược | Tiết kiệm kỳ vọng | Effort | Rủi ro | Owner |
|---|---|---|---|---|---|
| **P0 — làm ngay** | Semantic caching | $15–20/tháng token + UX | Thấp (1–2 ngày) | Trung bình (stale) | Bảo + Dũng |
| **P0 — làm ngay** | Tiered routing tighten | $50–55/tháng (−40% escalation) | Trung bình (2–3 ngày) | Trung bình (misroute) | Dũng + Bảo |
| **P1 — làm sau** | Selective inference | ~$15–25/tháng compute + giảm peak | Cao (cần queue + SLA) | Thấp | Tiền + Dũng |
| **Không làm** | Prompt compression | <15%, hại faithfulness | — | Cao (hallucinate số) | — |
| **Chưa làm** | Self-hosted model | 0 ở volume này | Cao | Cao, mâu thuẫn WS1 | — |

**Tổng tiết kiệm P0:** ~**$65–75/tháng** trên tổng $165/tháng token ⇒ **giảm ~40% cost token** mà **không chạm** vào đường câu nhạy cảm (compliance/PII vẫn escalate đầy đủ, context RAG không bị cắt).

---

## 4. Điểm tranh luận đã giải quyết (từ Worksheet 0)

| Câu hỏi tranh luận | Kết luận nhóm |
|---|---|
| Cost driver lớn nhất ở đâu? | Layer escalation GPT-5.4 (69% cost token) + human review + token top-funnel — giải quyết bằng Chiến lược 1 + 2 |
| Có cần queue/fallback ở đâu không? | **Có** — queue cho UI Cán bộ + SV (Chiến lược 3, P1). Fallback đã có Gemini ở Worksheet 2 |
| Cloud có cần đủ hybrid không? | **Không** — giữ Cloud-native như Worksheet 1. Self-hosted/hybrid không cost-effective ở volume 12k/peak day |

---

**Đầu ra:** Worksheet 3 — quyết định chính thức cho Phase 4 cost optimization. Hai P0 triển khai trước mùa tuyển sinh cao điểm; P1 làm nối tiếp sau khi intent classifier + SLA mỗi UI được chốt.
