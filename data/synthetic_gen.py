import json
import asyncio
import os
from typing import List, Dict
from openai import AsyncOpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

async def generate_qa_from_text(chunks: List[Dict], num_pairs: int = 50) -> List[Dict]:
    """
    Sử dụng OpenAI API để tạo các cặp QA từ các đoạn văn bản (chunks) cho trước,
    dựa trên các chiến lược tạo Hard Cases. Bao gồm ground_truth_ids để đánh giá Retrieval.
    """
    print(f"Generating {num_pairs} QA pairs from {len(chunks)} chunks...")
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Gộp các chunk để đưa vào prompt theo định dạng có ID
    context_str = "\n\n".join([f"[Chunk ID: {c['chunk_id']}]\n{c['text']}" for c in chunks])
    # Giới hạn tránh token limit
    context_str = context_str[:30000]

    all_qa_pairs = []
    batch_size = 10
    num_batches = (num_pairs + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, num_pairs - i * batch_size)
        print(f"Generating batch {i+1}/{num_batches} ({current_batch_size} pairs)...")
        prompt = f"""
        Bạn là một chuyên gia đánh giá và kiểm thử AI (AI Evaluation Expert).
        Dựa vào tập hợp các đoạn tài liệu có kèm [Chunk ID] dưới đây, hãy tạo ra đúng {current_batch_size} test case 
        theo các tiêu chí "Hard Cases" để đánh giá độ bền bỉ của AI:
        
        1. Adversarial Prompts: Gài bẫy prompt injection, goal hijacking.
        2. Edge Cases: Hỏi ngoài ngữ cảnh (AI phải biết nói "Tôi không biết"), câu hỏi mập mờ, hoặc cung cấp thông tin mâu thuẫn.
        3. Multi-turn Complexity / Multi-hop reasoning: Các câu hỏi phức tạp đòi hỏi thu thập và kết nối luồng logic từ nhiều trang (chunk) khác nhau.
        4. Căn bản: Một số câu hỏi truy xuất thực tế (fact-check) từ tài liệu với độ khó cao.
        
        Ví dụ (Few-shot Examples) để bạn học thiết kế câu hỏi dạng Hard Case dựa trên Metadata:
        - Good Example (Factual):
          Hỏi: "Tốc độ xử lý của mô hình là bao nhiêu?" -> Trả lời: "100 tokens/s" -> expected_retrieval_ids: ["page_1"]
        - Hard Case Example (Multi-hop reasoning):
          Hỏi: "Mô hình A được phát hành năm nào và yêu cầu RAM bao nhiêu?" -> Trả lời: "Năm 2024 và yêu cầu tối thiểu 16GB RAM." -> expected_retrieval_ids: ["page_1", "page_2"]
        - Edge Case Example (Out-of-context / Hallucination check):
          Hỏi: "Làm sao để tôi pha được một tách trà theo tài liệu?" -> Trả lời: "Tài liệu này không chứa thông tin về cách pha trà." -> expected_retrieval_ids: []
        
        Tài liệu:
        -------------
        {context_str}
        -------------
        
        Định dạng đầu ra BẮT BUỘC là một JSON Object có key "test_cases" chứa JSON Array của các object, mỗi object có cấu trúc:
        {{
            "test_cases": [
                {{
                    "question": "Nội dung câu hỏi tiếng Anh",
                    "expected_answer": "Câu trả lời kỳ vọng tiếng Anh",
                    "context": "Trích xuất một đoạn ngắn hoặc nội dung liên hệ từ tài liệu (tiếng Anh)",
                    "expected_retrieval_ids": ["Các Chunk ID trích xuất thông tin trả lời", "[] với câu hỏi lạc đề"],
                    "metadata": {{
                        "difficulty": "hard/edge-case/adversarial",
                        "type": "out-of-context/prompt-injection/factual/ambiguous/multi-hop"
                    }}
                }}
            ]
        }}
        """

        try:
            response = await client.chat.completions.create(
                model="gpt-4o", # Model phù hợp cho JSON generation và instruction following
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output strict JSON objects."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=4000
            )
            
            result = response.choices[0].message.content
            parsed_result = json.loads(result)
            all_qa_pairs.extend(parsed_result.get("test_cases", []))
        except Exception as e:
            print(f"Lỗi khi gọi OpenAI API hoặc parse JSON ở batch {i+1}: {e}")
            
    return all_qa_pairs

def extract_chunks_from_pdf(pdf_path: str) -> List[Dict]:
    """Đọc dữ liệu từ file PDF và chia thành các chunk (theo trang)"""
    chunks = []
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                # Gắn chunk_id cho mỗi trang (cách đơn giản nhất)
                chunks.append({
                    "chunk_id": f"page_{i+1}",
                    "text": text.strip(),
                    "source_document": os.path.basename(pdf_path)
                })
        return chunks
    except Exception as e:
        print(f"Lỗi khi đọc file PDF: {e}")
        return chunks

async def main():
    pdf_path = "data\sample.pdf" # Đường dẫn tới sample.pdf (so với thư mục data/)
    print(f"Reading text from {pdf_path}...")
    chunks = extract_chunks_from_pdf(pdf_path)
    
    if not chunks:
        print("Không có nội dung trong file PDF hoặc đọc file thất bại. Vui lòng kiểm tra lại.")
        return

    # Tạo 50 test case
    qa_pairs = await generate_qa_from_text(chunks, num_pairs=50)
    
    if qa_pairs:
        # Tạo mapping để tra cứu nhanh nguyên văn (text) của các chunk ID
        chunk_map = {c["chunk_id"]: c["text"] for c in chunks}
        source_doc = os.path.basename(pdf_path)
        
        output_file = "golden_set.jsonl" # Vì file chạy trong folder data, lưu trực tiếp
        with open(output_file, "w", encoding="utf-8") as f:
            for pair in qa_pairs:
                # Đảm bảo dùng đúng key với json format của output
                gt_ids = pair.get("expected_retrieval_ids", [])
                
                # Fetch chunk text thực tế từ ID
                pair["chunk_text"] = [chunk_map.get(cid, "") for cid in gt_ids]
                pair["source_document"] = source_doc
                
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"Done! Saved {len(qa_pairs)} test cases to data/{output_file}")
    else:
        print("Không tạo được test cases nào.")

if __name__ == "__main__":
    asyncio.run(main())
