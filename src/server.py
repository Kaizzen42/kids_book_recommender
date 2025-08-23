# src/server.py
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from src.search import find_similar_books
from fastapi.responses import JSONResponse

API_KEY = os.getenv("BOOKS_API_KEY")  # set via environment

app = FastAPI(title="Kids Book Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

def require_key(x_api_key: str | None, key_param: str | None):
    """Allow API key via header X-API-Key or ?key=..."""
    if not API_KEY:
        # dev convenience; for production, enforce
        return
    provided = x_api_key or key_param
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/search")
def search(
    title: str = Query(..., description="Seed book title"),
    intent: str = Query("", description="What you are looking for"),
    k: int = 20,
    year_min: int | None = None,
    year_max: int | None = None,
    min_votes: int | None = None,
    min_rating: float | None = None,
    x_api_key: str | None = Header(None),  # header key
    key: str | None = None,                # query key
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
    return df.to_dict(orient="records")

def df_json_response(df: pd.DataFrame) -> JSONResponse:
    # replace +/-inf -> NaN, then NaN -> None (JSON null)
    safe = (df.replace([np.inf, -np.inf], np.nan)
              .where(pd.notna(df), None))
    # (optional) cast numpy scalars to native python types
    safe = safe.applymap(
        lambda x: float(x) if isinstance(x, (np.floating,)) else
                  int(x)   if isinstance(x, (np.integer,))  else x
    )
    return JSONResponse(content=safe.to_dict(orient="records"))
