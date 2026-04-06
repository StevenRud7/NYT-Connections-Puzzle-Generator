"""
api/main.py
FastAPI application entry point.

Run:
  uvicorn api.main:app --reload --port 8000

Interactive docs:
  http://localhost:8000/docs
  http://localhost:8000/redoc
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensure both project root and ml/ are importable from wherever uvicorn is launched
ROOT   = Path(__file__).parent.parent
ML_DIR = ROOT / "ml"
for p in [str(ROOT), str(ML_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from api.routes.generate import router as generate_router
from api.routes.puzzles  import router as puzzles_router
from api.store           import PuzzleStore


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the puzzle store so first request isn't slow
    store = PuzzleStore()
    print(f"[startup] Puzzle store loaded — {store.count()} puzzles available.")
    yield
    print("[shutdown] API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Connections Puzzle Generator",
    description=(
        "Generates NYT-style Connections puzzles using ML/NLP trained on "
        "historical puzzle data. Serves puzzles and validates player guesses."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the static site (any localhost port) to call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",   # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "null",                    # file:// origin (opening index.html directly)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(generate_router, prefix="/api")
app.include_router(puzzles_router,  prefix="/api")


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    store = PuzzleStore()
    return {
        "status":        "ok",
        "puzzles_stored": store.count(),
    }


@app.get("/", tags=["meta"])
def root():
    return {
        "message": "Connections Puzzle Generator API",
        "docs":    "/docs",
        "health":  "/health",
    }