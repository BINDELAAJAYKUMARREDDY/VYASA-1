import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from app.models.schema import AskResult
from app.services.pipeline import answer_question
from app.core.config import MAX_CONCURRENT_REQUESTS, ASK_TIMEOUT_S
from app.core.logger import get_logger
from app.core.state import sessions


router = APIRouter()
log = get_logger("vyasa.api")
_sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


class AskBody(BaseModel):
    message: str
    session_id: str | None = None
    mode: str | None = "default"


@router.get("/")
async def root():
    return {
        "name": "VYASA-1",
        "status": "running",
        "endpoints": ["/ask?q=...", "/ui"],
        "open_ui": "http://127.0.0.1:8000/ui",
    }


@router.get("/ui", response_class=HTMLResponse)
async def ui():
    html_path = Path(__file__).resolve().parents[2] / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8", errors="ignore"))
    return HTMLResponse("<h2>UI not found. Missing frontend/index.html</h2>")


@router.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@router.get("/ask", response_model=AskResult)
async def ask(q: str, session_id: str = "default", mode: str = "default"):
    async with _sem:
        try:
            result = await asyncio.wait_for(_to_thread(answer_question, q, session_id, mode), timeout=ASK_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.info("ask_timeout")
            raise HTTPException(status_code=504, detail="Request timed out")

        return AskResult(
            question=q,
            emotion=result.get("emotion", "neutral"),
            intent=result.get("intent", "advice"),
            confidence=float(result.get("confidence", 0.0) or 0.0),
            answer=result["answer"],
            chunks=result["chunks"],
            verses_found=result["verses_found"],
            sources=result["sources"],
            used_ollama=result["used_ollama"],
            error=result.get("error"),
        )


@router.post("/ask")
async def ask_post(body: AskBody):
    msg = (body.message or "").strip()
    sid = (body.session_id or "default").strip() or "default"
    mode = (body.mode or "default").strip() or "default"

    # keep last 3 turns in sessions dict
    if sid not in sessions:
        sessions[sid] = []

    async with _sem:
        try:
            result = await asyncio.wait_for(_to_thread(answer_question, msg, sid, mode), timeout=ASK_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out")

    sessions[sid].append({"user": msg, "ai": result.get("answer", "")})
    sessions[sid] = sessions[sid][-3:]

    return {
        "session_id": sid,
        "response": result.get("answer", ""),
        "emotion": result.get("emotion", "neutral"),
        "intent": result.get("intent", "advice"),
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "sources": result.get("sources", []),
        "verses_found": result.get("verses_found", 0),
        "used_ollama": result.get("used_ollama", False),
        "error": result.get("error"),
    }


async def _to_thread(fn, *args, **kwargs):
    import asyncio

    return await asyncio.to_thread(fn, *args, **kwargs)

