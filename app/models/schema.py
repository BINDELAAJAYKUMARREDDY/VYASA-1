from typing import List, Optional

from pydantic import BaseModel


class AskResult(BaseModel):
    question: str
    emotion: str
    intent: str
    confidence: float = 0.0
    answer: str
    chunks: int
    verses_found: int
    sources: List[str]
    used_ollama: bool
    error: Optional[str] = None

