import os, time, sys

import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    data = load_vectors_and_meta()
    print("Loaded:")
    print("  n:", data["n"])
    print("  dim:", data["dim"])
    print("  title:", data["title"].shape)
    print("  description:", data["description"].shape)
    print("  reviews:", data["reviews"].shape)
    print("  meta cols:", list(data["meta"].columns))

    # Smoke test: try resolving a common word from your corpus, e.g., "Harry Potter"
    try:
        row = resolve_book_by_title(data["meta"], "Harry Potter")
        vecs = seed_vectors_by_row(data, row)
        print("Resolved row:", row, "title:", data["meta"].loc[row, "title"])
        print("Vec norms (should be ~1.0):",
              np.linalg.norm(vecs["title"]),
              np.linalg.norm(vecs["description"]),
              np.linalg.norm(vecs["reviews"]))
    except Exception as e:
        print("Lookup test:", e)
