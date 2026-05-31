"""
Cricket Match Intelligence — FastAPI Application
"""
from __future__ import annotations

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os

from app.routes.parse import router as parse_router
from app.routes.analytics import router as analytics_router
from app.routes.qa import router as qa_router
from app.routes.summary import router as summary_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import traceback

app = FastAPI(
    title="Cricket Match Intelligence",
    description="NLP-powered cricket commentary analysis with RAG Q&A",
    version="1.0.0",
)

# Catch ALL errors and return JSON — must be added before other middleware
class CatchAllMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            logging.getLogger(__name__).error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"detail": str(exc)})

app.add_middleware(CatchAllMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Also handle FastAPI's own exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.getLogger(__name__).error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})

# API routes
app.include_router(parse_router,     prefix="/api", tags=["Parse"])
app.include_router(analytics_router, prefix="/api", tags=["Analytics"])
app.include_router(qa_router,        prefix="/api", tags=["Q&A"])
app.include_router(summary_router,   prefix="/api", tags=["Summary"])

# Serve frontend — resolve relative to repo root (parent of app/)
_HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(_HERE, "..", "frontend"))

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
else:
    import logging
    logging.getLogger(__name__).warning(f"Frontend dir not found at: {FRONTEND_DIR}")


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "Cricket Match Intelligence"}
