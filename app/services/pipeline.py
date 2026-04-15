from typing import Dict, Any

from app.core.config import CHUNK_WORDS, TOP_K_PER_CHUNK, MAX_CONTEXT_CHARS
from app.core.logger import get_logger
from app.services.chunker import chunk_text, chunk_verses
from app.services.emotion import detect_emotion
from app.services.intent import detect_intent
from app.services.memory import add_turn, summarize_memory, is_followup
from app.services.source_mapper import get_clean_source_name
from app.services.transliterate import to_iast
from app.services.retriever import (
    context_only_answer,
    format_top_passages,
    retrieve_and_rerank,
    find_exact_verse,
)
from app.services.llm import generate_answer
from app.services.vedic_concepts import detect_canonical


log = get_logger("vyasa.pipeline")


def answer_question(user_text: str, session_id: str = "default", mode: str = "default") -> Dict[str, Any]:
    q = (user_text or "").strip()
    if not q:
        return {
            "emotion": "neutral",
            "intent": "advice",
            "answer": "Please type a question.",
            "chunks": 0,
            "verses_found": 0,
            "sources": [],
            "used_ollama": False,
            "error": None,
        }

    mem_text = summarize_memory(session_id, limit=4)
    followup = is_followup(q)

    emo = detect_emotion(q)
    emotion = emo.get("emotion", "neutral")
    intent = detect_intent(q)
    qtype = _query_type(q, emotion, intent)
    chunks = chunk_text(q, max_words=CHUNK_WORDS)

    # retrieval + cross-encoder reranking + diversity
    best = retrieve_and_rerank(q, chunks, retrieve_k=20, final_k=5)
    sources = [d.get("source", "") for d in best if d.get("source")]
    canonical = detect_canonical(q)

    # contextual chunking: 3-5 verses, overlap 1
    verse_blocks = chunk_verses(best, min_verses=3, max_verses=5, overlap=1)

    # build grounded context (no external knowledge)
    ctx_parts = []
    for b in verse_blocks[:4]:
        ctx_parts.append(b[:3500])
    llm_context = "\n\n---\n\n".join(ctx_parts).strip()

    top3_text = format_top_passages(best, limit=3)
    ctx = llm_context

    log.info(
        f"query_len={len(q)} emotion={emotion} intent={intent} chunks={len(chunks)} "
        f"retrieved={len(best)} ctx_chars={len(llm_context)} canonical={bool(canonical)}"
    )

    prompt_question = q
    if followup and mem_text:
        prompt_question = "Follow-up question (use memory for continuity):\n" + q

    if mem_text:
        llm_context = mem_text + "\n\n" + llm_context

    # add dharma reasoning hints (keeps it lightweight but less shallow)
    hard_truth = _hard_truth_line(q)
    dharma_hint = _dharma_reasoning_hint(q, qtype, hard_truth)
    llm_context2 = llm_context
    if dharma_hint:
        llm_context2 = llm_context + "\n\nDharma reasoning frame:\n" + dharma_hint

    # exact verse request: return exact or explicit not-found message
    exact = find_exact_verse(q, best)
    if any(k in q.lower() for k in ["specific line", "which verse", "exact reference", "where exactly", "exact verse"]):
        if canonical:
            out = {
                "emotion": emotion,
                "intent": intent,
                "confidence": 0.78,
                "answer": _canonical_answer(q, canonical),
                "chunks": len(chunks),
                "verses_found": len(best),
                "sources": sources[:20],
                "used_ollama": False,
                "error": None,
            }
            add_turn(session_id, q, out["answer"])
            return out
        if exact is None:
            out = {
                "emotion": emotion,
                "intent": intent,
                "confidence": 0.45 if best else 0.15,
                "answer": "No exact verse found in indexed corpus",
                "chunks": len(chunks),
                "verses_found": len(best),
                "sources": sources[:20],
                "used_ollama": False,
                "error": None,
            }
            add_turn(session_id, q, out["answer"])
            return out
        else:
            ans_exact = _exact_verse_answer(q, exact, mode=mode)
            out = {
                "emotion": emotion,
                "intent": intent,
                "confidence": 0.88,
                "answer": ans_exact,
                "chunks": len(chunks),
                "verses_found": len(best),
                "sources": sources[:20],
                "used_ollama": False,
                "error": None,
            }
            add_turn(session_id, q, out["answer"])
            return out

    ok, ans, err = generate_answer(prompt_question, llm_context2, emotion=emotion, intent=intent, mode=mode)
    if ok and ans.strip():
        ans2, valid = _validate_and_fix_answer(ans.strip(), best)
        if not valid:
            err = "llm_invalid_output"
        else:
            ans = ans2

        # grounding verification + one regenerate
        if err is None:
            grounded = _grounding_ok(ans, best)
            sem_ok, sem_sim = _semantic_grounding_ok(ans, best)
            log.info(f"grounding lexical={grounded} semantic={sem_ok} sim={sem_sim:.3f}")
            if not grounded:
                log.info("grounding_failed -> regenerate_once")
                stronger_ctx = llm_context2 + "\n\nIMPORTANT: Use a direct quote from the context and explain its meaning clearly."
                ok2, ans_retry, err_retry = generate_answer(prompt_question, stronger_ctx, emotion=emotion, intent=intent, mode=mode)
                if ok2 and ans_retry.strip():
                    ans3, valid3 = _validate_and_fix_answer(ans_retry.strip(), best)
                    if valid3 and _grounding_ok(ans3, best) and _semantic_grounding_ok(ans3, best)[0]:
                        ans = ans3
                    else:
                        err = "grounding_failed"
                else:
                    err = err_retry or "grounding_failed"
            elif not sem_ok:
                log.info(f"semantic_grounding_failed sim={sem_sim:.3f} -> regenerate_once")
                stronger_ctx = llm_context2 + "\n\nIMPORTANT: Your explanation must match the meaning of the context. Stay close to the retrieved text."
                ok2, ans_retry, err_retry = generate_answer(prompt_question, stronger_ctx, emotion=emotion, intent=intent, mode=mode)
                if ok2 and ans_retry.strip():
                    ans3, valid3 = _validate_and_fix_answer(ans_retry.strip(), best)
                    sem_ok3, _ = _semantic_grounding_ok(ans3, best)
                    if valid3 and sem_ok3:
                        ans = ans3
                    else:
                        err = "semantic_grounding_failed"
                else:
                    err = err_retry or "semantic_grounding_failed"

        # conflict resolution note (only if multiple sources)
        if best and len({get_clean_source_name(d.get("source", "")) for d in best if d.get("source")}) >= 2:
            conflict_line = _conflict_resolution_line(best)
            if conflict_line and conflict_line not in ans:
                ans = ans.strip() + "\n\n" + conflict_line

        # action quality control (make steps specific and today-actionable)
        ans = _action_qc(ans, emotion, intent)

        # comparison block when applicable
        cmp_block = generate_comparison_block(q, best)
        if cmp_block and "### comparison with modern science" not in ans.lower():
            ans = ans.strip() + "\n\n" + cmp_block

        # follow-up suggestion
        ans = ans.strip() + "\n\nYou may reflect further on: Which one small dharmic action can you take today?"

        # keep answer concise (soft limit)
        ans = _compress_answer(ans, max_chars=2400)

        log.info("used_ollama=true fallback=false")
        conf = _confidence_score(best, err is None)
        out = {
            "emotion": emotion,
            "intent": intent,
            "confidence": conf,
            "answer": ans.strip(),
            "chunks": len(chunks),
            "verses_found": len(best),
            "sources": sources[:20],
            "used_ollama": True,
            "error": None,
        }
        add_turn(session_id, q, out["answer"])
        return out

    # fallback: never fail
    log.info(f"used_ollama=false fallback=true err={err}")
    steps = _personal_steps(emotion, intent)
    if top3_text:
        fallback = "\n".join(
            [
                "I could not reach the local LLM right now, so I will answer using the best matching passages found.",
                f"Detected emotion: {emotion}",
                f"Detected intent: {intent}",
                "",
                top3_text,
                "",
                "Practical steps (tailored):",
                "- " + steps[0],
                "- " + steps[1],
                "- " + steps[2],
            ]
        ).strip()
    else:
        fallback = context_only_answer(q, ctx)
    out = {
        "emotion": emotion,
        "intent": intent,
        "confidence": _confidence_score(best, False),
        "answer": fallback,
        "chunks": len(chunks),
        "verses_found": len(best),
        "sources": sources[:20],
        "used_ollama": False,
        "error": err,
    }
    add_turn(session_id, q, out["answer"])
    return out


def _canonical_answer(question: str, row: Dict[str, Any]):
    return "\n".join(
        [
            "Short Answer: Canonical teaching detected and matched.",
            f"Key Insight: {row.get('meaning', '').strip()}",
            "",
            "### Scriptural insight",
            f"> \"{row.get('matched', '')}\" — {row.get('source', '')} ({row.get('reference', '')})",
            "",
            "### Simple explanation",
            "In simple terms: this is a core Vedantic statement and is answered from canonical knowledge even when exact corpus verse metadata is missing.",
            "",
            "### Final takeaway",
            "The concept is valid and grounded in canonical Vedantic tradition.",
        ]
    ).strip()


def _validate_and_fix_answer(answer: str, docs):
    a = _ensure_markdown_sections((answer or "").strip())
    if not a or len(a) < 80:
        return "", False

    # Must include at least 1 "According to ..." citation if we have docs
    if docs:
        if a.lower().count("according to") < 1:
            refs = _refs_human(docs, limit=3)
            if refs:
                a = a.strip() + "\n\nReferences:\n" + refs

        if a.lower().count("according to") < 1:
            return "", False

        # Must roughly follow the 5-section format
        must = [
            "short answer:",
            "key insight:",
            "### your question",
            "### scriptural insight",
            "### detailed explanation",
            "### simple explanation",
            "### final takeaway",
            "### conclusion",
        ]
        if sum(1 for m in must if m.lower() in a.lower()) < 4:
            return "", False

    return a, True


def _refs_human(docs, limit: int = 3):
    rows = []
    for d in docs[:limit]:
        src = (d.get("source") or "").strip()
        clean = get_clean_source_name(src)
        sa = to_iast(d.get("sanskrit") or "")
        en = (d.get("english") or "").strip()
        piece = []
        piece.append(f"According to {clean} ({src}):")
        if sa:
            piece.append("IAST: " + sa[:260])
        if en:
            piece.append("English: " + en[:360])
        rows.append("\n".join(piece).strip())
    return "\n\n".join(rows).strip()


def _personal_steps(emotion: str, intent: str):
    e = (emotion or "neutral").lower()
    i = (intent or "advice").lower()

    if e == "anxiety":
        base = [
            "Write down your top 3 fears tonight, then circle the 1 thing you can control in the next 24 hours.",
            "Do 5 minutes of slow breathing: inhale 4 seconds, exhale 6 seconds, then take one small action immediately.",
            "Study/work in a 25-minute timer (no phone). After the timer, write 1 line: what improved?",
        ]
    elif e == "sadness":
        base = [
            "Tell one trusted person what you are going through in one sentence (message is enough).",
            "Do one small body action: 10-minute walk or shower, then eat something light and simple.",
            "Pick one duty that keeps life stable today (sleep, food, work). Do it gently, not perfectly.",
        ]
    elif e == "anger":
        base = [
            "Delay any big message or decision for 30 minutes; write the angry version in notes, do not send it.",
            "Do a physical reset: 20 pushups or a brisk 5-minute walk to reduce the heat in the body.",
            "Then choose one boundary or one truthful sentence you will say calmly, without insult.",
        ]
    elif e == "confusion":
        base = [
            "Write 2 columns: facts vs assumptions. Move anything uncertain into the assumptions column.",
            "List 3 options and 1 small test for each option you can do this week.",
            "Choose the option that reduces harm and regret, even if it is slower.",
        ]
    else:
        base = [
            "Write the problem in one clear sentence. If it’s too big, split it into 3 smaller problems.",
            "Pick the smallest next action (10 minutes). Do it now, then reassess.",
            "Read one cited passage again and extract one rule you can follow today.",
        ]

    if i == "explanation":
        base[2] = "After reading the references, write a 3-line summary in your own words (what it means, why it matters, what to do)."
    elif i == "philosophical_question":
        base[1] = "Spend 10 minutes in quiet reflection: ask 'What is my duty today?' and write the first honest answer."

    return base[:3]


def _query_type(q: str, emotion: str, intent: str) -> str:
    t = (q or "").lower()
    if intent == "technical" or any(x in t for x in ["specific line", "which verse", "exact reference", "chapter", "canto", "verse"]):
        return "technical"
    if "why" in t or "meaning" in t or intent in ["philosophical_question", "explanation"]:
        return "philosophy"
    if "what should i do" in t or "how do i" in t or intent == "advice":
        return "guidance"
    if emotion in ["sadness", "guilt", "anxiety"] or intent == "emotional_support":
        return "emotional"
    return "general"


def _grounding_ok(answer: str, docs) -> bool:
    # Very simple grounding check: answer should overlap with at least one doc meaning.
    # If it doesn't, it often means it went generic/hallucinated.
    a = (answer or "").lower()
    if not a or not docs:
        return True

    # must contain a short direct quote marker
    if "\"" not in answer and "“" not in answer:
        return False

    # overlap with at least one doc english
    import re

    aw = set(re.findall(r"[a-z]{4,}", a))
    if len(aw) < 10:
        return False

    best = 0
    for d in docs[:5]:
        en = (d.get("english") or "").lower()
        dw = set(re.findall(r"[a-z]{4,}", en))
        if not dw:
            continue
        overlap = len(aw & dw)
        best = max(best, overlap)

    return best >= 6


def _semantic_grounding_ok(answer: str, docs, threshold: float = 0.20):
    # semantic alignment check: cosine(answer_embedding, docs_embedding)
    # lightweight, uses the same sentence-transformers stack if available.
    try:
        from sentence_transformers import SentenceTransformer
        import math

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        a = (answer or "").strip()
        if not a or not docs:
            return True, 1.0

        docs_text = " ".join([(d.get("english") or "")[:500] for d in docs[:5]])
        if not docs_text.strip():
            return True, 1.0

        v1 = model.encode([a[:1200]])[0]
        v2 = model.encode([docs_text[:2000]])[0]

        dot = float((v1 * v2).sum())
        n1 = math.sqrt(float((v1 * v1).sum())) + 1e-9
        n2 = math.sqrt(float((v2 * v2).sum())) + 1e-9
        sim = dot / (n1 * n2)
        return sim >= threshold, float(sim)
    except Exception:
        # if model isn't available, don't block output
        return True, 0.0


def _conflict_resolution_line(docs):
    # If sources emphasize different angles, mention it briefly (adds depth).
    # Use similarity between top 2 english texts as a signal.
    try:
        en = [(d.get("english") or "").strip() for d in docs[:4] if (d.get("english") or "").strip()]
        if len(en) < 2:
            return ""
        s = _quick_jaccard(en[0], en[1])
        if s < 0.10:
            n1 = get_clean_source_name(docs[0].get("source", ""))
            n2 = get_clean_source_name(docs[1].get("source", ""))
            return f"Note: while {n1} emphasizes one angle, {n2} may emphasize another (for example, detachment vs duty). Both can be true depending on your situation."
        return ""
    except Exception:
        return ""


def _quick_jaccard(a: str, b: str):
    import re

    sa = set(re.findall(r"[a-z]{4,}", (a or "").lower()))
    sb = set(re.findall(r"[a-z]{4,}", (b or "").lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / (len(sa | sb) + 1e-9)


def _action_qc(answer: str, emotion: str, intent: str):
    # Ensure steps are specific + today-actionable + not generic repeats.
    a = (answer or "").strip()
    if not a:
        return a

    if "Practical steps" not in a:
        steps = _personal_steps(emotion, intent)
        return a + "\n\nPractical steps\n- " + steps[0] + "\n- " + steps[1] + "\n- " + steps[2]

    # If it contains very generic phrases, append better steps
    low = a.lower()
    generic_flags = ["stay calm", "be calm", "be patient", "take deep breaths", "don't worry"]
    if any(g in low for g in generic_flags):
        steps = _personal_steps(emotion, intent)
        a = a + "\n\nImproved practical steps (specific)\n- " + steps[0] + "\n- " + steps[1] + "\n- " + steps[2]
    return a


def _compress_answer(answer: str, max_chars: int = 2400):
    a = (answer or "").strip()
    if len(a) <= max_chars:
        return a

    # simple compression: remove repeated blank lines + keep first occurrence of headings
    lines = [x.rstrip() for x in a.splitlines()]
    out = []
    seen = set()
    for ln in lines:
        key = ln.strip().lower()
        if key in seen and key in [
            "understanding your situation",
            "what went wrong",
            "what scriptures teach",
            "practical steps",
            "final calm guidance",
        ]:
            continue
        seen.add(key)
        out.append(ln)
        if sum(len(x) + 1 for x in out) > max_chars:
            break

    return "\n".join(out).strip()


def _ensure_markdown_sections(answer: str):
    a = (answer or "").strip()
    if not a:
        return a
    mapping = [
        ("Your question", "### Your question"),
        ("Scriptural insight", "### Scriptural insight"),
        ("Detailed explanation", "### Detailed explanation"),
        ("Simple explanation", "### Simple explanation"),
        ("What you can do", "### What you can do (if applicable)"),
        ("Comparison with modern science", "### Comparison with modern science (if applicable)"),
        ("Final takeaway", "### Final takeaway"),
        ("Conclusion", "### Conclusion"),
    ]
    for plain, md in mapping:
        a = a.replace("\n" + plain + "\n", "\n" + md + "\n")
        if a.startswith(plain + "\n"):
            a = md + a[len(plain):]
    if "### Conclusion" not in a:
        a = a.strip() + "\n\n### Conclusion\nStay steady in dharma, apply one clear step today, and review the cited teaching calmly."
    return a


def _exact_verse_answer(q: str, d: Dict[str, Any], mode: str = "default"):
    src = (d.get("source") or "").strip()
    can = str(d.get("canto") or "").strip()
    ch = str(d.get("chapter") or "").strip()
    v = str(d.get("verse") or "").strip()
    sa = (d.get("sanskrit") or "").strip()
    en = (d.get("english") or "").strip()
    iast = to_iast(sa)

    ref = src
    if can or ch or v:
        ref = f"{src} (Canto {can}, Chapter {ch}, Verse {v})".strip()

    lines = []
    lines.append("Short Answer: Here is the closest exact verse found in the indexed corpus.")
    lines.append("Key Insight: The answer is grounded in one direct verse reference.")
    lines.append("")
    lines.append("### Your question")
    lines.append(q)
    lines.append("")
    lines.append("### Scriptural insight")
    if sa:
        lines.append(f'> "{sa[:300]}" — {ref}')
    elif en:
        lines.append(f'> "{en[:300]}" — {ref}')
    lines.append("")
    if mode == "scholar":
        if sa:
            lines.append("Sanskrit:")
            lines.append(sa[:600])
            lines.append("")
        if iast and iast != sa:
            lines.append("Transliteration (IAST):")
            lines.append(iast[:600])
            lines.append("")
    lines.append("### Simple explanation")
    lines.append("In simple terms: this verse directly answers your request using the indexed text.")
    if en:
        lines.append(en[:500])
    lines.append("")
    lines.append("### Final takeaway")
    lines.append("This exact reference is from the indexed corpus, without hallucinated verse numbers.")
    return "\n".join(lines).strip()


def generate_comparison_block(query: str, docs):
    t = (query or "").lower()
    if not any(k in t for k in ["science", "physics", "modern", "relativity", "cosmos", "universe", "time"]):
        return ""
    if not docs:
        return ""
    return "\n".join(
        [
            "### Comparison with modern science (if applicable)",
            "",
            "| Aspect | Vedic View | Modern Science |",
            "| ------ | ---------- | -------------- |",
            "| Cause | Consciousness/cosmic order and hierarchy of lokas | Gravity, velocity, physical laws |",
            "| Observer | Different realms can experience time differently | Different frames observe time differently |",
            "| Mechanism | Cyclical cosmic time + qualitative metaphysics | Space-time curvature and measurable equations |",
            "| Nature | Includes meaning, ethics, and purpose | Primarily quantitative and physical |",
        ]
    ).strip()


def _confidence_score(docs, grounded_ok: bool):
    # A simple score based on retrieval strength + grounding checks.
    if not docs:
        return 0.10
    base = 0.35
    base += min(0.30, len(docs) * 0.05)
    if grounded_ok:
        base += 0.20
    # average score if present
    scores = [float(d.get("score") or 0.0) for d in docs if d.get("score") is not None]
    if scores:
        avg = sum(scores) / max(1, len(scores))
        base += max(0.0, min(0.15, avg / 10.0))
    return float(max(0.0, min(0.95, base)))


def _hard_truth_line(q: str) -> str:
    t = (q or "").lower()
    bad = [
        "lie",
        "lying",
        "cheat",
        "cheating",
        "manipulate",
        "manipulation",
        "gaslight",
        "betray",
        "betrayed",
        "affair",
        "steal",
        "stole",
        "abuse",
        "threaten",
        "blackmail",
    ]
    if any(w in t for w in bad):
        return "Hard truth: this action broke trust and created harm. It is adharma if it is done knowingly and selfishly."
    return ""


def _dharma_reasoning_hint(q: str, qtype: str, hard_truth: str) -> str:
    # This is not "external knowledge". It's a reasoning checklist for the model.
    # It forces deeper analysis instead of shallow formatting.
    lines = []
    lines.append("Step 1: What exactly happened? (facts only)")
    lines.append("Step 2: Is it adharma? Why? (harm, truth, self-control)")
    lines.append("Step 3: What principle applies? (satya/truth, ahimsa/non-harm, non-attachment, duty)")
    lines.append("Step 4: What consequence does this create if repeated?")
    lines.append("Step 5: What correction is dharmic right now (small, realistic)?")
    if hard_truth:
        lines.append("")
        lines.append(hard_truth)

    if qtype == "philosophy":
        lines.append("")
        lines.append("Depth note: explain the principle clearly, avoid only giving steps.")
    elif qtype == "technical":
        lines.append("")
        lines.append("Depth note: include exact references and avoid broad generalization.")
    elif qtype == "emotional":
        lines.append("")
        lines.append("Tone note: be compassionate but not vague. Name the mistake gently if needed.")

    return "\n".join(lines).strip()

