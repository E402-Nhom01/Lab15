import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

# Load real RAG modules from simple-rag package in this repository.
SIMPLE_RAG_ROOT = Path(__file__).resolve().parent / "simple-rag"
if str(SIMPLE_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMPLE_RAG_ROOT))

from rag.llm import GPT4oMiniLLM  # noqa: E402
from rag.prompt import ANSWER_PROMPT  # noqa: E402
from rag.rerank import CrossEncoderRerank  # noqa: E402
from rag.retrieval import BM25Retrieval  # noqa: E402


class MainAgent:
    """
    Agent benchmark wrapper using the real simple-rag stack:
    - Retrieval: BM25
    - Optional rerank: CrossEncoder (V2)
    - Generation: GPT-4o-mini
    """

    def __init__(self, version: str = "v1"):
        self.version = version
        self.name = f"SimpleRAG-{version}"

        self.retrieval_top_k = 3 if version == "v1" else 10
        self.rerank_top_k = 0 if version == "v1" else 3

        pdf_path = Path(__file__).resolve().parents[1] / "data" / "sample.pdf"
        chunks = self._build_page_chunks(pdf_path)
        self.retrieval = BM25Retrieval(documents=chunks)
        self.llm = GPT4oMiniLLM(model_name="gpt-4o-mini")

        self.rerank = None
        if self.rerank_top_k > 0:
            try:
                self.rerank = CrossEncoderRerank(
                    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
            except Exception as exc:
                print(f"⚠️ Cannot initialize reranker, fallback to no-rerank: {exc}")

    def _build_page_chunks(self, pdf_path: Path) -> List[Dict]:
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing source PDF: {pdf_path}")

        reader = PdfReader(str(pdf_path))
        chunks: List[Dict] = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            chunks.append(
                {
                    "chunk_id": f"page_{i}",
                    "chunk_text": text,
                    "source_document": pdf_path.name,
                }
            )

        if not chunks:
            raise ValueError(f"No text extracted from PDF: {pdf_path}")
        return chunks

    def _run_sync(self, question: str) -> Dict:
        start = time.perf_counter()
        retrieved = self.retrieval.retrieve(question, top_k=self.retrieval_top_k)

        docs = [r["chunk_text"] for r in retrieved]
        meta = [
            {"chunk_id": r["chunk_id"], "source_document": r["source_document"]}
            for r in retrieved
        ]

        final_docs = docs
        final_meta = meta
        if self.rerank and docs:
            reranked_docs, _ = self.rerank.rerank(
                question,
                docs,
                top_k=min(self.rerank_top_k, len(docs)),
                metadata=meta,
            )
            doc_to_meta = {d: m for d, m in zip(docs, meta)}
            final_docs = reranked_docs
            final_meta = [doc_to_meta[d] for d in reranked_docs if d in doc_to_meta]

        prompt = ANSWER_PROMPT.format(query=question, context="\n".join(final_docs))
        answer = self.llm.generate(prompt)

        latency = time.perf_counter() - start
        return {
            "answer": answer,
            "contexts": final_docs,
            "metadata": {
                "model": "gpt-4o-mini",
                "agent_version": self.version,
                "tokens_used": 0,  # Current LLM wrapper does not expose usage.
                "sources": [m["chunk_id"] for m in final_meta],
                "source_documents": [m["source_document"] for m in final_meta],
                "latency_sec_agent": latency,
                "retrieval_top_k": self.retrieval_top_k,
                "rerank_top_k": self.rerank_top_k if self.rerank else 0,
            },
        }

    async def query(self, question: str) -> Dict:
        # Offload sync calls (BM25 + rerank + LLM HTTP) from event loop.
        return await asyncio.to_thread(self._run_sync, question)
