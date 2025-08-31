# src/server.py
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from src.search import find_similar_books
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse, Response, HTMLResponse

# For logging in just once instead of every time
from fastapi import FastAPI, HTTPException, Query, Header, Request, Response, Form
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse


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


@app.get("/", include_in_schema=False)
def home():
    # Send people to the interactive docs (Swagger UI)
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Return empty 204 so browsers stop retrying, keeps logs clean
    return Response(status_code=204)

@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
def login_form(request: Request, key: str | None = None):
    """
    GET /login?key=... → sets cookie & redirects to /docs (one-time magic link)
    GET /login          → shows a tiny form to enter the key once
    """
    if key:
        # Set cookie then bounce to docs
        resp = RedirectResponse(url="/docs", status_code=302)
        if not API_KEY or key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
        resp.set_cookie(
            key="api_key",
            value=key,
            httponly=True,
            samesite="lax",
            secure=True,           # browser sees https on ngrok → set True
            max_age=60*60*24*30,   # 30 days
        )
        return resp

    # Show a minimal login form
    return HTMLResponse("""
<!doctype html><html><head><meta charset="utf-8"><title>Login</title>
<style>body{font-family:sans-serif;max-width:480px;margin:3rem auto}</style></head>
<body>
  <h3>Enter API Key</h3>
  <form method="post" action="/login">
    <input type="password" name="key" placeholder="API key" style="width:100%;padding:.6rem" />
    <button style="margin-top:1rem;padding:.6rem 1rem">Login</button>
  </form>
</body></html>
""")

@app.post("/login", include_in_schema=False)
def login_post(key: str = Form(...)):
    if not API_KEY or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    resp = RedirectResponse(url="/docs", status_code=302)
    resp.set_cookie(
        key="api_key",
        value=key,
        httponly=True,
        samesite="lax",
        secure=True,             # https via ngrok
        max_age=60*60*24*30,
    )
    return resp
