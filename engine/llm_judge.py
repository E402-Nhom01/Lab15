import asyncio
from typing import Dict, Any, List

class LLMJudge:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        # TODO: Định nghĩa rubrics chi tiết cho các tiêu chí: Accuracy, Professionalism, Safety
        self.rubrics = {
            "accuracy": "Chấm điểm từ 1-5 dựa trên độ chính xác so với Ground Truth...",
            "tone": "Chấm điểm từ 1-5 dựa trên sự chuyên nghiệp của ngôn ngữ..."
        }

    def _normalize_text(self, text: str) -> List[str]:
        """
        Chuẩn hóa text thành danh sách token đơn giản để so sánh overlap.
        Lưu ý: Đây là heuristic baseline; bản production nên dùng LLM judge hoặc embeddings.
        """
        cleaned = text.lower()
        for ch in [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", '"', "'"]:
            cleaned = cleaned.replace(ch, " ")
        return [tok for tok in cleaned.split() if tok]

    def calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Faithfulness (0-1): đo mức độ answer được "neo" vào retrieved contexts.
        Công thức baseline:
            |tokens(answer) ∩ tokens(contexts)| / |tokens(answer)|
        """
        answer_tokens = set(self._normalize_text(answer))
        if not answer_tokens:
            return 0.0

        context_tokens = set(self._normalize_text(" ".join(contexts)))
        supported_tokens = answer_tokens.intersection(context_tokens)
        return len(supported_tokens) / len(answer_tokens)

    def calculate_relevancy(self, question: str, answer: str) -> float:
        """
        Relevancy (0-1): đo mức độ answer trả đúng trọng tâm câu hỏi.
        Công thức baseline:
            |tokens(question) ∩ tokens(answer)| / |tokens(question)|
        """
        question_tokens = set(self._normalize_text(question))
        if not question_tokens:
            return 0.0

        answer_tokens = set(self._normalize_text(answer))
        overlap = question_tokens.intersection(answer_tokens)
        return len(overlap) / len(question_tokens)

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        EXPERT TASK: Gọi ít nhất 2 model (ví dụ GPT-4o và Claude).
        Tính toán sự sai lệch. Nếu lệch > 1 điểm, cần logic xử lý.
        """
        # Giả lập gọi 2 model
        score_a = 4
        score_b = 3
        
        avg_score = (score_a + score_b) / 2
        agreement = 1.0 if score_a == score_b else 0.5
        faithfulness = self.calculate_faithfulness(answer, [ground_truth])
        relevancy = self.calculate_relevancy(question, answer)
        
        return {
            "final_score": avg_score,
            "agreement_rate": agreement,
            "individual_scores": {"gpt-4o": score_a, "claude-3-5": score_b},
            "faithfulness": faithfulness,
            "relevancy": relevancy
        }

    async def check_position_bias(self, response_a: str, response_b: str):
        """
        Nâng cao: Thực hiện đổi chỗ response A và B để xem Judge có thiên vị vị trí không.
        """
        pass
