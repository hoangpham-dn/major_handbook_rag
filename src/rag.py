from typing import Any, Tuple, Optional, Dict
import math
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

PROMPT_TEMPLATE = """
You are a helpful AI assistant that answers questions based on provided documents.

CONTEXT FROM DOCUMENTS:
{context}

CHAT HISTORY:
{chat_history}
---
QUESTION: {question}

INSTRUCTIONS:
1. If the context contains relevant information, answer based on that.
2. If the context is insufficient, use your knowledge.
3. Be concise unless more detail is needed.

ANSWER:
"""

QUERY_REWRITE_PROMPT = """
Given the history and the latest user question,
rewrite the latest question into a standalone question.

INSTRUCTIONS:
- Resolve references to previous topics; preserve user's intent.
- Include relevant context from previous messages.
- Do not answer the question.
- Return unchanged if the question is already standalone.

CHAT HISTORY:
{chat_history}

lATEST QUESTION:
{question}

Standalone question:
"""



def ask(
        question: str,
        db: Chroma, 
        llm: OllamaLLM,
        top_k: int,
        score_threshold: float,
        chat_history: Optional[list[Tuple[str, str]]] = None
        ) -> Dict[str, Any]:

    chat_history = chat_history or []

    # 1. Format conversation history
    history = format_history(chat_history)

    # 2. Rewrite question using conversation history
    if chat_history:
        rewrite_prompt = QUERY_REWRITE_PROMPT.format(
            chat_history=history,
            question=question
        )

        search_query = llm.invoke(rewrite_prompt).strip()
    else:
        search_query = question

    raw_results = db.similarity_search_with_score(search_query, k=top_k)
    sources = [
        {
            'source': str(doc.metadata.get('source', 'Unknown')),
            'score': 1 / (1 + score),
            'content': doc.page_content
        }
        for doc, score in raw_results
        if 1 / (1 + score) >= score_threshold
    ]

    used_fallback = not sources
    if used_fallback:
        context = "No sufficiently relevant documents found. Use your general knowledge."
    else:
        context = "\n---\n".join(
            str(s['source']) + "\n" 
            + str(s['content']) 
            for s in sources
        )

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        chat_history=history
        )
    
    answer = llm.invoke(prompt).strip()
    updated_history = chat_history + [(question, answer)]
    return {
        'answer': answer,  
        'sources': sources, 
        'used_fallback': used_fallback,
        "chat_history": updated_history,
        "search_query": search_query
        }


def format_history(chat_history: Optional[list[Tuple[str, str]]] = None) -> str:
    if not chat_history:
        return "No previous conversation"
    
    return "\n".join(
        f"User: {turn[0]} \n Assistant: {turn[1]}"
        for turn in chat_history[-5:]
    )
