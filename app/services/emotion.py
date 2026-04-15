_pipe = None


def detect_emotion(text: str):
    # HF model + fallback keyword match.
    # Returns: {"emotion": "...", "confidence": float}
    global _pipe
    t = (text or "").strip()
    if not t:
        return {"emotion": "neutral", "confidence": 0.0}

    try:
        if _pipe is None:
            from transformers import pipeline

            _pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
        out = _pipe(t[:512])
        # pipeline output shape can differ depending on transformers versions
        if isinstance(out, list) and out and isinstance(out[0], list):
            best = out[0][0]
        elif isinstance(out, list) and out and isinstance(out[0], dict):
            best = out[0]
        else:
            best = {"label": "neutral", "score": 0.0}

        label = (best.get("label") or "neutral").lower()
        score = float(best.get("score") or 0.0)
        return {"emotion": label, "confidence": score}
    except Exception:
        low = t.lower()
        if any(w in low for w in ["anxious", "panic", "worried", "fear", "scared", "stress"]):
            return {"emotion": "anxiety", "confidence": 0.6}
        if any(w in low for w in ["sad", "depressed", "hopeless", "grief", "lonely"]):
            return {"emotion": "sadness", "confidence": 0.6}
        if any(w in low for w in ["angry", "furious", "rage", "hate"]):
            return {"emotion": "anger", "confidence": 0.6}
        if any(w in low for w in ["confused", "lost", "unclear", "dont know", "don't know"]):
            return {"emotion": "confusion", "confidence": 0.55}
        return {"emotion": "neutral", "confidence": 0.4}

