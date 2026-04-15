import json
import time
from typing import Tuple, Optional

import requests

from app.core.config import OLLAMA_HOST, OLLAMA_MODEL, LLM_TIMEOUT_S


def _call_ollama(prompt: str, timeout_s: int = 35) -> Tuple[bool, str, Optional[str]]:
    url = OLLAMA_HOST.rstrip("/") + "/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
        },
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            return False, "", f"ollama_http_{r.status_code}"
        data = r.json()
        text = (data.get("response") or "").strip()
        if not text:
            return False, "", "ollama_empty"
        return True, text, None
    except requests.Timeout:
        return False, "", "ollama_timeout"
    except Exception as ex:
        return False, "", f"ollama_error:{type(ex).__name__}"


def generate_answer(question: str, context: str, emotion: str = "neutral", intent: str = "advice", mode: str = "default") -> Tuple[bool, str, Optional[str]]:
    q = (question or "").strip()
    ctx = (context or "").strip()

    prompt = "\n".join(
        [
            "You are VYASA-1, a calm, dharma-guided assistant.",
            "Very strict grounding rules:",
            "- Use ONLY the provided CONTEXT below.",
            "- Do NOT use outside knowledge. Do NOT guess.",
            "- If the context does NOT contain the information, say exactly:",
            "  \"This is not found in the available texts\"",
            "- When you reference a teaching, write it like: \"According to the Mahabharata...\"",
            "",
            "CONTEXT (retrieved scripture passages):",
            ctx[:14000],
            "",
            "USER QUESTION:",
            q,
            "",
            f"DETECTED EMOTION: {emotion}",
            f"DETECTED INTENT: {intent}",
            f"MODE: {mode}",
            "",
            "MANDATORY answer format:",
            "Short Answer: (one direct line)",
            "Key Insight: (one strong line)",
            "### Your question",
            "### Scriptural insight",
            "### Detailed explanation",
            "### Simple explanation",
            "### What you can do (if applicable)",
            "### Comparison with modern science (if applicable)",
            "### Final takeaway",
            "### Conclusion",
            "",
            "CITATIONS:",
            "- Include at least 2 citations using: \"According to <text name>...\" and include the source label in brackets like (Rigveda 1.1.1).",
            "- Also include 1 short direct quote (<= 25 words) copied from the CONTEXT, and explain what it means in your own words.",
            "- In Scholar mode: include Sanskrit + transliteration + verse number if available in context.",
        ]
    ).strip()

    ok, text, err = _call_ollama(prompt, timeout_s=LLM_TIMEOUT_S)
    if ok:
        return True, text, None

    # retry once, shorter prompt (sometimes long prompt triggers issues)
    prompt2 = "\n".join(
        [
            "You are VYASA-1. Use ONLY the provided context. Be concise and structured.",
            "If not found, say: \"This is not found in the available texts\"",
            "CONTEXT:",
            ctx[:8000],
            "QUESTION:",
            q,
            f"EMOTION: {emotion}",
            f"INTENT: {intent}",
            f"MODE: {mode}",
        ]
    ).strip()
    ok2, text2, err2 = _call_ollama(prompt2, timeout_s=max(10, int(LLM_TIMEOUT_S * 0.7)))
    if ok2:
        return True, text2, None

    return False, "", err2 or err

