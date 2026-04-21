import asyncio
from typing import List, Dict, Any


class MainAgent:
    """
    Đây là Agent mẫu sử dụng kiến trúc RAG đơn giản.
    Phiên bản này đã chuẩn hóa output để phục vụ evaluation benchmark.
    """

    def __init__(self, top_k: int = 2):
        self.name = "SupportAgent-v1"
        self.top_k = top_k

    async def query(self, question: str) -> Dict[str, Any]:
        """
        Mô phỏng quy trình RAG:
        1. Retrieval: Tìm kiếm context liên quan.
        2. Generation: Gọi LLM để sinh câu trả lời.
        """

        # ================= SIMULATE LATENCY =================
        await asyncio.sleep(0.5)

        # ================= FAKE RETRIEVAL =================
        contexts = [
            "Để đổi mật khẩu, người dùng cần truy cập vào phần cài đặt tài khoản và chọn mục 'Đổi mật khẩu'.",
            "Hệ thống yêu cầu xác thực bằng email hoặc OTP trước khi cho phép thay đổi mật khẩu."
        ]

        # giả lập doc ids (quan trọng cho evaluation)
        retrieved_ids = [f"doc_{i}" for i in range(len(contexts))]

        # ================= FAKE GENERATION =================
        answer = (
            f"Dựa trên tài liệu hệ thống, câu trả lời cho '{question}' là: "
            "Bạn cần vào phần cài đặt tài khoản, chọn 'Đổi mật khẩu' "
            "và xác thực bằng email hoặc mã OTP."
        )

        # ================= STANDARDIZED OUTPUT =================
        return {
            "answer": answer,

            # context dùng cho faithfulness / grounding eval
            "contexts": contexts[: self.top_k],

            # REQUIRED for retrieval evaluation (hit@k, MRR)
            "retrieved_docs": retrieved_ids[: self.top_k],

            # metadata cho tracking experiment
            "metadata": {
                "model": "gpt-4o-mini",
                "retrieval": "mock",
                "reranker": None,
                "top_k": self.top_k,
                "pipeline": "MockRAGPipeline",

                # optional but useful
                "tokens_used": 150,
                "sources": ["policy_handbook.pdf"],
            },
        }


# ================= TEST =================

if __name__ == "__main__":
    agent = MainAgent(top_k=2)

    async def test():
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print("\n=== OUTPUT ===")
        print(resp)

    asyncio.run(test())