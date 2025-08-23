import os, time, sys

import numpy as np
import pandas as pd

import re
import unicodedata
from typing import Callable, Iterable, List


def project_root():
    # adjust if your layout differs
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, ".."))

# Load Vectors (After embedding Generation)
def load_vectors_and_meta():
    root = project_root()
    vec_dir = os.path.join(root, "data", "vectors")

    title_path  = os.path.join(vec_dir, "title_embeddings.npy")
    desc_path   = os.path.join(vec_dir, "description_embeddings.npy")
    review_path = os.path.join(vec_dir, "reviews_embeddings.npy")
    meta_path   = os.path.join(vec_dir, "book_metadata.parquet")

    # Load
    title_embs  = np.load(title_path, mmap_mode="r")   # mmap: fast start, low RAM
    desc_embs   = np.load(desc_path, mmap_mode="r")
    review_embs = np.load(review_path, mmap_mode="r")
    meta = pd.read_parquet(meta_path)

    # Basic checks
    n = len(meta)
    assert title_embs.shape[0] == n,  f"title rows {title_embs.shape[0]} != meta {n}"
    assert desc_embs.shape[0]  == n,  f"desc rows {desc_embs.shape[0]} != meta {n}"
    assert review_embs.shape[0] == n, f"review rows {review_embs.shape[0]} != meta {n}"
    dim = title_embs.shape[1]
    assert desc_embs.shape[1] == dim == review_embs.shape[1], "embedding dims mismatch"

    return {
        "title": title_embs,
        "description": desc_embs,
        "reviews": review_embs,
        "meta": meta,
        "dim": dim,
        "n": n,
    }

# Cosine similarity utilities
def dot_product_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity via dot product (assumes normalized vectors).
    Why normalized? 
    Because cosine(a,b) = (a.b) / (||a|| ||b||) 
    and if ||a||=||b||=1 then cos(a,b)=a.b
    
    We alreadty normalize embeddings at generation time, by setting normalize_embeddings=True.
    
    Args:
        query: shape (dim,)  is the normalized query vector
        matrix: shape (n, dim) is the normalized matrix of vectors to compare against
    Returns:
        similarities: shape (n,) array of cosine similarities
    """
    return matrix @ query  # (n,dim) @ (dim,) -> (n,) Fast vectorized dot product.

def dot_similarity_batch(
    queries: np.ndarray,  # (m, dim)
    matrix: np.ndarray    # (n, dim)
) -> np.ndarray:
    """
    Batched cosine similarity via dot product (assumes normalized vectors).
    """
    return queries @ matrix.T  # (m,dim) @ (dim,n) -> (m,n)

# -------------------- Title lookups --------------------
def resolve_book_by_title(meta: pd.DataFrame, title: str) -> int:
    """
    
    Returns the row index (int) of the best match for the title.
    Strategy:
        - casefold match on substring containment
        - tie-break by highest sum_n_votes, then highest n_reviews
        
    Raises:
        ValueError if no match found.
    """
    t = title.casefold().strip()
    candidates = meta[meta['title'].fillna('').str.casefold().str.contains(t, na=False)]
    if candidates.empty:
        raise ValueError(f"No match found for title {title!r}")

    candidates = candidates.copy()
    candidates['_rank']  = -candidates['sum_n_votes'].fillna(0)
    candidates['_rank2'] = -candidates['n_reviews'].fillna(0)
    candidates = candidates.sort_values(by=['_rank', '_rank2']).drop(columns=['_rank', '_rank2'])
    return int(candidates.index[0])

def seed_vectors_by_row(data: dict, row: int) -> dict:
    """
    Returns the three vectors (title/description/reviews) for a given row index.
    """
    return {
        "title": data["title"][row],          # (dim,)
        "description": data["description"][row],
        "reviews": data["reviews"][row],
    }

_punct_re = re.compile(r"[\W_]+", re.UNICODE)
_paren_re = re.compile(r"\([^)]*\)")       # remove (...) such as (Harry Potter, #3), (2012 ed.)
_series_hash_re = re.compile(r"#\s*\d+")
_multi_space_re = re.compile(r"\s+")

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def canonical_title(title: str) -> str:
    """
    Normalize visually identical titles (across editions) to the same key.
    Heuristics:
      - lowercase, strip accents
      - remove parenthetical tags (...), e.g., (Harry Potter, #3)
      - remove '#3' style series numbers
      - keep portion before ':' (often drops edition subtitles)
      - strip punctuation & collapse spaces
    """
    if not isinstance(title, str):
        title = "" if title is None else str(title)
    s = _strip_accents(title.casefold().strip())
    s = _paren_re.sub(" ", s)
    s = _series_hash_re.sub(" ", s)
    if ":" in s:
        s = s.split(":", 1)[0]
    s = _punct_re.sub(" ", s)
    s = _multi_space_re.sub(" ", s)
    return s.strip()

def select_unique_by_key(
    meta: pd.DataFrame,
    ranked_idx: Iterable[int],
    key_fn: Callable[[str], str],
    k: int,
) -> List[int]:
    """
    Walk the ranked indices and keep the first occurrence per canonical title.
    Returns up to k indices from `ranked_idx` preserving order.
    """
    seen = set()
    out: List[int] = []
    for idx in ranked_idx:
        t = meta.iloc[idx]["title"]
        key = key_fn(t)
        if key in seen:
            continue
        seen.add(key)
        out.append(idx)
        if len(out) >= k:
            break
    return out


# ---------- Metadata features ----------
def _minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def _log1p_minmax(x: np.ndarray) -> np.ndarray:
    return _minmax(np.log1p(x.astype(float)))

def build_meta_features(meta: pd.DataFrame) -> dict:
    year     = meta["publication_year"].fillna(meta["publication_year"].median()).to_numpy()
    rating   = meta["average_rating"].fillna(meta["average_rating"].median()).to_numpy()
    n_votes  = meta["sum_n_votes"].fillna(0).to_numpy()

    recency  = _minmax(year)            # newer → closer to 1
    quality  = _minmax(rating)          # higher rating → closer to 1
    popularity = _log1p_minmax(n_votes) # diminishing returns

    return {"recency": recency, "quality": quality, "popularity": popularity}

# ---------- Intent heuristics ----------
def intent_to_params(intent_text: str | None) -> dict:
    """
    Very light keyword mapping.
    Returns: dict with keys:
      - weights: (w_title, w_desc, w_rev)
      - year_min/year_max/min_votes/min_rating (optional)
      - boosts: dict(alpha_recency, beta_popularity, gamma_quality)
    """
    intent = (intent_text or "").casefold()

    # defaults
    params = {
        "weights": (0.50, 0.35, 0.15),  # title, description, reviews
        "boosts": {"alpha_recency": 0.05, "beta_popularity": 0.10, "gamma_quality": 0.08},
    }

    # Weight nudges
    if any(k in intent for k in ["plot", "story", "theme", "genre", "synopsis", "like this book"]):
        params["weights"] = (0.30, 0.55, 0.15)
    if any(k in intent for k in ["writing", "style", "voice", "tone", "prose"]):
        params["weights"] = (0.20, 0.35, 0.45)
    if any(k in intent for k in ["series", "title", "name", "sequel"]):
        params["weights"] = (0.60, 0.30, 0.10)

    # Filters
    if "recent" in intent or "new" in intent or "latest" in intent:
        params["year_min"] = 2018
        params["boosts"]["alpha_recency"] = 0.10
    if "classic" in intent or "older" in intent or "before" in intent:
        params["year_max"] = 2005
        params["boosts"]["alpha_recency"] = 0.00  # don't boost recency

    if any(k in intent for k in ["popular", "bestseller", "widely read", "trending"]):
        params["min_votes"] = 1000
        params["boosts"]["beta_popularity"] = 0.15

    if any(k in intent for k in ["highly rated", "well rated", "quality"]):
        params["min_rating"] = 4.0
        params["boosts"]["gamma_quality"] = 0.12

    return params

# ---------- Core scoring ----------
def weighted_similarity_from_row(row: int, data: dict, weights: tuple[float, float, float]) -> np.ndarray:
    """
    Blend similarities from the three embedding spaces using the seed book at `row`.
    All embeddings were saved unit-normalized, so cosine = dot.
    """
    w_title, w_desc, w_rev = weights
    q_title = data["title"][row]        # (dim,)
    q_desc  = data["description"][row]
    q_rev   = data["reviews"][row]

    sim = (
        w_title * (data["title"] @ q_title) +
        w_desc  * (data["description"] @ q_desc) +
        w_rev   * (data["reviews"] @ q_rev)
    )
    return sim  # (N,)

def apply_filters(meta: pd.DataFrame,
                  year_min: int | None = None,
                  year_max: int | None = None,
                  min_votes: int | None = None,
                  min_rating: float | None = None) -> np.ndarray:
    mask = np.ones(len(meta), dtype=bool)
    if year_min is not None:
        mask &= (meta["publication_year"] >= year_min)
    if year_max is not None:
        mask &= (meta["publication_year"] <= year_max)
    if min_votes is not None:
        mask &= (meta["sum_n_votes"] >= min_votes)
    if min_rating is not None:
        mask &= (meta["average_rating"] >= min_rating)
    return mask

def hybrid_score(sim: np.ndarray, meta_feats: dict,
                 alpha_recency=0.05, beta_popularity=0.10, gamma_quality=0.08) -> np.ndarray:
    """
    final = sim + α*recency + β*popularity + γ*quality
    """
    return (sim
            + alpha_recency * meta_feats["recency"]
            + beta_popularity * meta_feats["popularity"]
            + gamma_quality * meta_feats["quality"])

# ---------- Public API ----------
def find_similar_books(
    query_title: str,
    intent_text: str = "",
    k: int = 20,
    weights: tuple[float, float, float] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    min_votes: int | None = None,
    min_rating: float | None = None,
    exclude_self: bool = True,
):
    data = load_vectors_and_meta()
    meta = data["meta"].reset_index(drop=False)  # keep original row index as 'index'
    meta.rename(columns={"index": "_row"}, inplace=True)

    # Resolve the seed book row (row is the original index used by npy files)
    row = resolve_book_by_title(data["meta"], query_title)

    # Intent defaults and merges
    intent_params = intent_to_params(intent_text)
    if weights is None:
        weights = intent_params["weights"]

    # Merge optional filters: explicit args win over intent-derived
    y_min = year_min if year_min is not None else intent_params.get("year_min")
    y_max = year_max if year_max is not None else intent_params.get("year_max")
    v_min = min_votes if min_votes is not None else intent_params.get("min_votes")
    r_min = min_rating if min_rating is not None else intent_params.get("min_rating")

    # Compute similarities
    sim = weighted_similarity_from_row(row, data, weights)  # (N,)

    # Build metadata features & boosts
    feats = build_meta_features(data["meta"])
    boosts = intent_params["boosts"]
    final = hybrid_score(sim, feats,
                         alpha_recency=boosts["alpha_recency"],
                         beta_popularity=boosts["beta_popularity"],
                         gamma_quality=boosts["gamma_quality"])

    # Apply filters
    mask = apply_filters(data["meta"], year_min=y_min, year_max=y_max, min_votes=v_min, min_rating=r_min)
    if exclude_self:
        mask[row] = False

    # Rank all passing candidates
    idx = np.argwhere(mask).ravel()
    ranked = idx[np.argsort(final[idx])[::-1]]   # descending by score

    # Over-fetch to have room after dedup
    overfetch = ranked[: max(200, 10*k)]

    # Keep first occurrence per canonical title
    unique_top = select_unique_by_key(
        data["meta"], overfetch, key_fn=canonical_title, k=k
    )

    out = data["meta"].iloc[unique_top].copy()
    out["score"] = final[unique_top]
    out["similarity"] = sim[unique_top]
    
    out = out[["book_id", "title", "score", "similarity", "average_rating", "publication_year", "n_reviews", "sum_n_votes"]]
    return out.reset_index(drop=True)

if __name__ == "__main__":
    data = load_vectors_and_meta()
    print("Loaded OK:", data["n"], "rows,", data["dim"], "dim")

    try:
        # Example 1: default intent
        res = find_similar_books("Harry Potter", intent_text="", k=10)
        print("\nTop-10 (default):")
        print(res.head(10).to_string(index=False))

        # Example 2: “similar plot/theme”
        res2 = find_similar_books("Harry Potter", intent_text="similar plot and theme, popular, highly rated", k=10)
        print("\nTop-10 (plot/theme + popularity + quality):")
        print(res2.head(10).to_string(index=False))

        # Example 3: “similar writing style”
        res3 = find_similar_books("Harry Potter", intent_text="similar writing style / voice, recent", k=10, year_min=2010)
        print("\nTop-10 (style + recent):")
        print(res3.head(10).to_string(index=False))

    except Exception as e:
        print("Search test error:", e)
