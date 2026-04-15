import time


_mem = {}


def add_turn(session_id: str, user_text: str, answer_text: str, max_turns: int = 5):
    sid = (session_id or "default").strip() or "default"
    if sid not in _mem:
        _mem[sid] = []

    _mem[sid].append(
        {
            "t": int(time.time()),
            "user": (user_text or "").strip(),
            "answer": (answer_text or "").strip(),
        }
    )
    _mem[sid] = _mem[sid][-max_turns:]


def get_recent(session_id: str, limit: int = 5):
    sid = (session_id or "default").strip() or "default"
    return list(_mem.get(sid, []))[-limit:]


def summarize_memory(session_id: str, limit: int = 4):
    turns = get_recent(session_id, limit=limit)
    if not turns:
        return ""

    lines = []
    lines.append("Recent conversation (for continuity, do not invent details):")
    for i, t in enumerate(turns[-limit:], 1):
        u = (t.get("user") or "")[:240]
        a = (t.get("answer") or "")[:240]
        if u:
            lines.append(f"- User: {u}")
        if a:
            lines.append(f"  Assistant: {a}")
    return "\n".join(lines).strip()


def is_followup(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return False
    keys = [
        "why",
        "explain more",
        "explain again",
        "tell me more",
        "more details",
        "what next",
        "what should i do next",
        "next step",
        "continue",
    ]
    return any(k in t for k in keys)

