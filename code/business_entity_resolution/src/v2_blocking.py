"""
v2 — blocking (Multi-Key with Hot-Key Capping)
==============================================
Guarantees memory safety by extracting highly specific blocking keys
and dropping any key that generates >50,000 cross-join pairs.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os, re, time, pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import jellyfish

# ---------------------------------------------------------------------------
# Paths & Hyperparams
# ---------------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))
STUDENT_RES = os.path.join(REPO_ROOT, "6ab10eb3b23ba_student_resource", "student_resource")
TRAIN_DIR   = os.path.join(STUDENT_RES, "dataset", "mini_train") # <--- USING MINI TRAIN
TEST_DIR    = os.path.join(STUDENT_RES, "dataset", "test")
OUTPUT_DIR  = os.path.join(STUDENT_RES, "output")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

K               = 50
MATCH_THRESHOLD = 0.05     # extremely loose threshold for blocking
MAX_FEATURES    = 100_000
MAX_PAIRS_PER_KEY = 250_000   

# ---------------------------------------------------------------------------
# Text normalisation & Keys
# ---------------------------------------------------------------------------
_PUNCT_RE  = re.compile(r'[^\w\s]')
_SPACE_RE  = re.compile(r'\s+')

def normalise_name(text: str) -> str:
    text = str(text).lower().strip()
    return _SPACE_RE.sub(' ', _PUNCT_RE.sub(' ', text)).strip()

def get_keys(text: str) -> list:
    words = text.split()
    if not words: return []
    
    keys = set()
    
    # 1. Acronym of the first few words (usually the name)
    name_words = words[:8]
    if len(name_words) > 1:
        acr = "".join([w[0] for w in name_words if w])
        if len(acr) >= 3: keys.add(f"acr:{acr}")
        
    # 2. Every word and its soundex (if >= 4 chars to avoid tiny common words)
    for w in words:
        if len(w) >= 4:
            keys.add(f"w:{w}")
            keys.add(f"s:{jellyfish.soundex(w)}")
            
    return list(keys)

def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["norm_name"] = df["business_name"].apply(normalise_name)
    df["tfidf_text"] = df["norm_name"] + " " + df["business_address"].fillna("").str.lower()
    
    print("    Extracting blocking keys from full text (name + address)...")
    df["bkeys"] = df["tfidf_text"].apply(get_keys)
    df_exploded = df.explode("bkeys").dropna(subset=["bkeys"])
    df_exploded["bkey"] = df_exploded["bkeys"] + "|" + df_exploded["country"].str.strip()
    
    return df, df_exploded[["entity_id", "bkey"]]

# ---------------------------------------------------------------------------
# Process pipeline
# ---------------------------------------------------------------------------
def process_split(s1: pd.DataFrame, s23: pd.DataFrame, s1_keys: pd.DataFrame, s23_keys: pd.DataFrame, vec: TfidfVectorizer) -> dict:
    t0 = time.time()
    
    print("    Counting key frequencies...")
    c1 = s1_keys["bkey"].value_counts()
    c23 = s23_keys["bkey"].value_counts()
    
    valid_keys = []
    for k, count1 in c1.items():
        count23 = c23.get(k, 0)
        if count1 * count23 <= MAX_PAIRS_PER_KEY:
            valid_keys.append(k)
            
    valid_keys = set(valid_keys)
    print(f"    Kept {len(valid_keys):,} safe keys (dropped {len(c1)-len(valid_keys):,} hot keys).")
    
    s1_keys = s1_keys[s1_keys["bkey"].isin(valid_keys)]
    s23_keys = s23_keys[s23_keys["bkey"].isin(valid_keys)]
    
    print("    Merging on safe blocking keys...")
    pairs = s1_keys.merge(s23_keys.rename(columns={"entity_id": "cid"}), on="bkey", how="inner")
    pairs = pairs[["entity_id", "cid"]].drop_duplicates()
    print(f"    Generated {len(pairs):,} unique candidate pairs in {time.time()-t0:.1f}s.")
    
    if len(pairs) == 0: return {}
    
    print("    Calculating TF-IDF similarities for candidates...")
    t1 = time.time()
    
    s1_vecs = vec.transform(s1["tfidf_text"])
    s23_vecs = vec.transform(s23["tfidf_text"])
    
    s1_idx_map = {eid: idx for idx, eid in enumerate(s1["entity_id"])}
    s23_idx_map = {eid: idx for idx, eid in enumerate(s23["entity_id"])}
    
    pairs["s1_idx"] = pairs["entity_id"].map(s1_idx_map)
    pairs["s23_idx"] = pairs["cid"].map(s23_idx_map)
    
    CHUNK = 2_000_000
    scores = []
    
    for start in range(0, len(pairs), CHUNK):
        chunk = pairs.iloc[start:start+CHUNK]
        v1_chunk = s1_vecs[chunk["s1_idx"].values]
        v23_chunk = s23_vecs[chunk["s23_idx"].values]
        sims = np.array(v1_chunk.multiply(v23_chunk).sum(axis=1)).flatten()
        scores.extend(sims)
        
    pairs["score"] = scores
    print(f"    Calculated similarities in {time.time()-t1:.1f}s.")
    
    print(f"    Filtering top {K} matches...")
    pairs = pairs[pairs["score"] > 0.05]
    pairs = pairs.sort_values(["entity_id", "score"], ascending=[True, False])
    pairs = pairs.groupby("entity_id").head(K)
    
    cands = {}
    for _, row in pairs.iterrows():
        sid = row["entity_id"]
        if sid not in cands: cands[sid] = {}
        cands[sid][row["cid"]] = row["score"]
        
    return cands

def score_recall(cands: dict, gt_path: str):
    gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    found = total = 0
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        gt_ids = set(row["matched_entity_ids"].split(",")) - {""}
        if not gt_ids: continue
        total += len(gt_ids)
        pred_ids = set(cands.get(s1_id, {}).keys())
        found += len(gt_ids & pred_ids)
    
    recall = found / total if total else 0.0
    print(f"\n  Blocking recall: {found:,} / {total:,} = {recall:.4f}  ({recall:.1%})")
    print("  ✅ GATE PASSED" if recall >= 0.95 else "  ⚠️ GATE FAILED (Target ≥ 95%)")

def main():
    print("=" * 60)
    print("v2 — blocking (Multi-Key with Hot-Key Capping) [MINI DATASET]")
    print("=" * 60)
    
    print("\n[1/3] Loading MINI TRAIN data...")
    s1_tr = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    s23_tr = pd.concat([
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", dtype=str).fillna(""),
        pd.read_csv(os.path.join(TRAIN_DIR, "train_source3.tsv"), sep="\t", dtype=str).fillna("")
    ], ignore_index=True)
    
    s1_tr, s1_tr_keys = prep_df(s1_tr)
    s23_tr, s23_tr_keys = prep_df(s23_tr)
    
    print("\n[2/3] Fitting TF-IDF vectorizer...")
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=MAX_FEATURES, sublinear_tf=True)
    sample_texts = pd.concat([s1_tr["tfidf_text"], s23_tr["tfidf_text"]])
    vec.fit(sample_texts)
    
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)
        
    print("\n[3/3] Processing TRAIN split...")
    cands_tr = process_split(s1_tr, s23_tr, s1_tr_keys, s23_tr_keys, vec)
    score_recall(cands_tr, os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))
    
    print("\n[Done] Skipping TEST split during local development.")

if __name__ == "__main__":
    main()
