# src/server.py
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi import Security
from fastapi.security import APIKeyHeader

# --- Config ---
API_KEY = os.getenv("BOOKS_API_KEY")  # set in your .env and loaded by serve.sh

# Swagger "Authorize" will prompt for this header once and reuse it
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# --- App ---
app = FastAPI(title="Kids Book Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you like
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utilities ---
def df_json_response(df: pd.DataFrame) -> JSONResponse:
    """Make DataFrame JSON-safe: replace inf/NaN and cast numpy scalars."""
    safe = df.replace([np.inf, -np.inf], np.nan).where(pd.notna(df), None)
    # Cast numpy scalars -> Python types so JSONResponse is happy
    for col in safe.columns:
        if pd.api.types.is_numeric_dtype(safe[col]):
            safe[col] = safe[col].astype(object).apply(
                lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x
            )
    return JSONResponse(content=safe.to_dict(orient="records"))

def require_key(header_key: str | None, query_key: str | None):
    """Accept key via Swagger header (X-API-Key) or ?key=... (for curl)."""
    if not API_KEY:  # dev mode: allow if unset; for prod, enforce by removing this
        return
    provided = header_key or query_key
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# --- Routes ---
@app.get("/", include_in_schema=False)
def home():
    # Friendly landing: redirect to interactive docs
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Quiet the browser's favicon probe
    return Response(status_code=204)

@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}

# NOTE: No /login endpoints anymore. Auth is done via Swagger "Authorize" (X-API-Key).

from src.search import find_similar_books  # import after app to avoid heavy import on tools

@app.get("/search")
def search(
    request: Request,
    title: str = Query(..., description="Seed book title"),
    intent: str = Query("", description="What you are looking for"),
    k: int = 20,
    year_min: int | None = None,
    year_max: int | None = None,
    min_votes: int | None = None,
    min_rating: float | None = None,
    x_api_key: str | None = Security(api_key_header),  # set once via Swagger "Authorize"
    key: str | None = None,  # optional ?key=... support for curl
):
    require_key(x_api_key, key)

    df = find_similar_books(
        query_title=title,
        intent_text=intent,
        k=k,
        year_min=year_min,
        year_max=year_max,
        min_votes=min_votes,
        min_rating=min_rating,
        exclude_self=True,
        exclude_same_work=True,
        same_language_as_seed=True,
    )
    return df_json_response(df)
