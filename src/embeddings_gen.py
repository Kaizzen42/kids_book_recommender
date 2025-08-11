
from typing import List
import pandas as pd
import os, time
import numpy as np

# For embeddings
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

MAX_CHARS = 3400 # P95 of length of review_texts

def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"

def load_model():
    device = get_device()
    print(f"Using device: {device}")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print(f"Embedding dim: {model.get_sentence_embedding_dimension()}")
    return model

def read_data():
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, '..', 'data', 'clean', 'books_with_reviews.csv')
    df = pd.read_csv(csv_path)
    return df

def embed_texts(model: SentenceTransformer, texts: List[str]):
    # Basic, no-frills helper for tiny batches
    return model.encode(
        texts,
        normalize_embeddings=True,      # good for cosine similarity
        convert_to_numpy=True,
        show_progress_bar=False
    )

def clean_series_text(s):
    # Ensure strings, drop NaNs, Trim very long text fields. 
    s = s.fillna('').astype(str)
    s = s.str.strip()
    return s.apply(lambda x: x[:MAX_CHARS])

def embed_column(model, series: pd.Series,
                 batch_size:int = 64,
                 desc: str = "") -> np.ndarray:
    
    texts = clean_series_text(series).tolist()
    dim = model.get_sentence_embedding_dimension()
    out = np.empty((len(texts), dim), dtype = np.float32)
    
    t0 = time.time()
    for i in tqdm(range(0, len(texts), batch_size),
                  desc=f"Embedding {desc or series.name}",
                  unit="batch"):
        batch = texts[i:i + batch_size]
        embs = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # good for cosine similarity
            show_progress_bar=False #tqdm handles this
        )
        
        out[i:i+len(batch)] = embs
    
    dt = time.time() - t0
    print(f"Finished {desc or series.name} — {len(texts)} rows in {dt/60:.1f} min "
          f"(~{dt/max(1, len(texts)):.3f}s/row)")
    return out 

def save_arrays_and_meta(
    df: pd.DataFrame,
    title_embs: np.ndarray,
    desc_embs: np.ndarray,
    review_embs: np.ndarray,
    out_dir: str
):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "title_embeddings.npy"), title_embs)
    np.save(os.path.join(out_dir, "description_embeddings.npy"), desc_embs)
    np.save(os.path.join(out_dir, "reviews_embeddings.npy"), review_embs)

    # Keep a slim metadata table (no gigantic text fields)
    meta_cols = ["book_id", "title", "average_rating", "publication_year", "n_reviews", "sum_n_votes"]
    meta = df[meta_cols].copy()
    meta.to_parquet(os.path.join(out_dir, "book_metadata.parquet"), index=False)

    print(f"Saved to {out_dir}")
    
if __name__ == "__main__":
    df = read_data()
    print(f"Rows in DataFrame: {len(df)}")
    print(f"Columns in DataFrame: {df.columns.tolist()}")
    required = {"book_id","title","average_rating","description","publication_year","n_reviews","review_texts","sum_n_votes"}
    missing = required - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"
    
    model = load_model()
     # === Full column embeddings ===
    title_embs  = embed_column(model, df["title"],        batch_size=64, desc="title")
    desc_embs   = embed_column(model, df["description"],  batch_size=64, desc="description")
    review_embs = embed_column(model, df["review_texts"], batch_size=32, desc="review_texts")  # smaller batch for long text

    # === Save arrays + metadata ===
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vectors")
    save_arrays_and_meta(df, title_embs, desc_embs, review_embs, out_dir)
