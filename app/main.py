from fastapi import FastAPI

from app.api.routes import router
from app.core.logger import get_logger
from app.core.state import sessions


log = get_logger("vyasa")

app = FastAPI(title="VYASA-1", version="1.0")
app.include_router(router)

# Simple in-memory sessions store is in app/core/state.py


if __name__ == "__main__":
    import uvicorn

    log.info("Starting VYASA-1 on http://127.0.0.1:8000")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
