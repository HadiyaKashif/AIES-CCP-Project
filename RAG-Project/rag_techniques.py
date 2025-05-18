from config import gemini_llm
from langchain_core.documents import Document as LangchainDocument
from collections import defaultdict
from typing import List


# Prompt Helpers
def generate_query_variations(q: str) -> List[str]:
    prompt = f"Generate 3 different rephrasings of this query:\nOriginal: {q}\n\nVariations:\n1. "
    return [q] + [v.strip() for v in gemini_llm.invoke(prompt).content.split("\n") if v.strip()][:3]

def generate_sub_questions(q: str) -> List[str]:
    prompt = f"Break this question into 2-3 standalone sub-questions:\nOriginal: {q}\n\nSub-questions:\n1. "
    return [v.strip() for v in gemini_llm.invoke(prompt).content.split("\n") if v.strip()][:3]

def generate_reasoning_steps(q: str) -> List[str]:
    prompt = f"To answer this, list steps to retrieve required info:\nQuestion: {q}\n\nSteps:\n1. "
    return [v.strip() for v in gemini_llm.invoke(prompt).content.split("\n") if v.strip()][:3]

def reciprocal_rank_fusion(results: List[List[LangchainDocument]], k=60) -> List[LangchainDocument]:
    scores = defaultdict(float)
    for docs in results:
        for rank, doc in enumerate(docs, 1):
            doc_id = doc.page_content
            scores[doc_id] += 1.0 / (rank + k)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [LangchainDocument(page_content=doc_id, metadata={}) for doc_id, _ in sorted_docs[:4]]