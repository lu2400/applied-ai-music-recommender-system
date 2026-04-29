"""
RAG retriever for the music recommender system.

Loads a local music knowledge base (data/music_knowledge.json) and returns
the most relevant documents for a given user query using keyword overlap scoring.
This context is injected into the LLM prompt before preference extraction,
giving the model accurate genre/mood/era guidance from the knowledge base.
"""

import json
import os
import re
from typing import List, Dict

_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "music_knowledge.json")
_knowledge_base: List[Dict] = []


def _load_kb() -> List[Dict]:
    global _knowledge_base
    if not _knowledge_base:
        with open(_KB_PATH, encoding="utf-8") as f:
            _knowledge_base = json.load(f)["documents"]
    return _knowledge_base


def retrieve(query: str, k: int = 3) -> List[Dict]:
    """Return the top-k most relevant knowledge documents for the query.

    Scoring uses token-level overlap between the query and each document's
    title, content, and tags. Ties broken by document order (stable).
    """
    docs = _load_kb()
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))

    scored = []
    for doc in docs:
        doc_text = " ".join([
            doc.get("title", ""),
            doc.get("content", ""),
            " ".join(doc.get("tags", [])),
        ]).lower()
        doc_tokens = set(re.findall(r"[a-z0-9]+", doc_text))
        overlap = len(tokens & doc_tokens)
        if overlap > 0:
            scored.append((overlap, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]]


def format_context(docs: List[Dict]) -> str:
    """Format retrieved documents into a compact context string for the LLM."""
    if not docs:
        return ""
    lines = ["Relevant music knowledge (use this to guide your extraction):"]
    for doc in docs:
        lines.append(f"- {doc['title']}: {doc['content']}")
    return "\n".join(lines)
