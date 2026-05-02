from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes_sessions, routes_chat, routes_traces
from app.api.errors import HelixError, helix_error_handler
from app.db.session import init_db
from app.obs.logging import configure_logging

UI_DIR = Path(__file__).parent.parent / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    configure_logging()
    await init_db()
    yield


app = FastAPI(title="Helix SROP", version="0.1.0", lifespan=lifespan)

# CORS — allow browser UI to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_sessions.router, prefix="/v1")
app.include_router(routes_chat.router, prefix="/v1")
app.include_router(routes_traces.router, prefix="/v1")

# Register typed error handler — converts HelixError subclasses to RFC 7807 JSON
app.add_exception_handler(HelixError, helix_error_handler)  # type: ignore[arg-type]

# Serve the chat UI
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/index.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
