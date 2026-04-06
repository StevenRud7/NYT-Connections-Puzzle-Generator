"""
Phase 3 — Embeddings
Generates and caches sentence-transformer embeddings for every unique
word and group theme label found in connections.json.

Outputs:
  ml/models/embeddings_cache.npz  — numpy arrays (words + themes)
  ml/models/embeddings_index.json — maps word/theme strings to array indices

Run:
  pip install sentence-transformers numpy
  python ml/embeddings.py
"""

import json
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit(
        "sentence-transformers not installed.\n"
        "Run: pip install sentence-transformers"
    )

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("data/connections.json")
MODELS_DIR   = Path("ml/models")
CACHE_PATH   = MODELS_DIR / "embeddings_cache.npz"
INDEX_PATH   = MODELS_DIR / "embeddings_index.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Best balance of speed vs quality for semantic similarity tasks
MODEL_NAME = "all-MiniLM-L6-v2"

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_corpus(data_path: Path) -> tuple[list[str], list[str]]:
    """
    Extract two distinct corpora from connections.json:
      words  — every unique member word (uppercase)
      themes — every unique group label (uppercase)
    """
    with open(data_path, "r") as f:
        puzzles = json.load(f)

    words_set  = set()
    themes_set = set()

    for puzzle in puzzles:
        for group in puzzle["answers"]:
            themes_set.add(group["group"].upper().strip())
            for member in group["members"]:
                words_set.add(member.upper().strip())

    return sorted(words_set), sorted(themes_set)


def embed(model: SentenceTransformer, texts: list[str],
          label: str, batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of texts with a progress log.
    Returns a float32 numpy array of shape (len(texts), embedding_dim).
    """
    print(f"  Encoding {len(texts):,} {label} ...")
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit-norm → cosine sim == dot product
    )
    return vectors.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Quick sanity-check helper (vectors already unit-normed)."""
    return float(np.dot(a, b))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("  Phase 3 — Embeddings")
    print("═" * 60)

    # 1. Load corpus
    print(f"\n[1/4] Loading corpus from {DATA_PATH} ...")
    words, themes = load_corpus(DATA_PATH)
    print(f"      {len(words):,} unique words | {len(themes):,} unique theme labels")

    # 2. Load model
    print(f"\n[2/4] Loading sentence-transformer model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"      Embedding dimension: {dim}")

    # 3. Encode
    print("\n[3/4] Generating embeddings ...")
    word_vectors  = embed(model, words,  "member words")
    theme_vectors = embed(model, themes, "theme labels")

    # 4. Save
    print(f"\n[4/4] Saving cache → {CACHE_PATH}")
    np.savez_compressed(
        CACHE_PATH,
        word_vectors=word_vectors,
        theme_vectors=theme_vectors,
    )

    index = {
        "model":          MODEL_NAME,
        "embedding_dim":  dim,
        "word_count":     len(words),
        "theme_count":    len(themes),
        "words":          words,   # position i → word_vectors[i]
        "themes":         themes,  # position i → theme_vectors[i]
    }
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)

    print(f"      Index saved  → {INDEX_PATH}")

    # ── Quick sanity checks ──────────────────────────────────────────────────
    print("\n── Sanity checks ───────────────────────────────────────────")

    # Check 1: BARK should be closer to TREE/DOG themes than to MUSIC
    check_words   = ["BARK", "BLUSH", "SCORE"]
    check_themes  = ["TREE PARTS", "EXHIBIT NERVOUSNESS", "EVALUATE"]

    w_idx = {w: i for i, w in enumerate(words)}
    t_idx = {t: i for i, t in enumerate(themes)}

    for cw in check_words:
        if cw not in w_idx:
            print(f"  '{cw}' not in vocabulary — skipping")
            continue
        wv = word_vectors[w_idx[cw]]
        sims = []
        for ct in check_themes:
            if ct in t_idx:
                tv = theme_vectors[t_idx[ct]]
                sims.append((ct, cosine_similarity(wv, tv)))
        if sims:
            best = max(sims, key=lambda x: x[1])
            print(f"  '{cw}' → closest theme: '{best[0]}' (sim={best[1]:.3f})")

    # Check 2: Nearest neighbours for a few words (among all theme vectors)
    print("\n  Nearest theme for each word (top-1 cosine sim):")
    sample_words = ["GRADE", "FIDGET", "MASCARA", "TIGER"]
    for sw in sample_words:
        if sw not in w_idx:
            continue
        wv   = word_vectors[w_idx[sw]]
        sims = theme_vectors @ wv          # dot product == cosine (unit normed)
        best_i = int(np.argmax(sims))
        print(f"  '{sw}' → '{themes[best_i]}' (sim={sims[best_i]:.3f})")

    print("\n✓ Embeddings complete.")
    print("═" * 60)


# ── Public API ────────────────────────────────────────────────────────────────

class EmbeddingStore:
    """
    Lightweight wrapper used by downstream modules (group_generator, validator).
    Usage:
        store = EmbeddingStore()
        vec   = store.word_vector("BARK")
        sims  = store.word_theme_similarities("BARK")   # {theme: score}
    """

    def __init__(
        self,
        cache_path: Path = CACHE_PATH,
        index_path: Path = INDEX_PATH,
    ):
        data = np.load(cache_path)
        self._word_vectors  = data["word_vectors"]   # (W, D) float32
        self._theme_vectors = data["theme_vectors"]  # (T, D) float32

        with open(index_path) as f:
            idx = json.load(f)
        self._words      = idx["words"]
        self._themes     = idx["themes"]
        self._word_idx   = {w: i for i, w in enumerate(self._words)}
        self._theme_idx  = {t: i for i, t in enumerate(self._themes)}
        self.dim         = idx["embedding_dim"]

    # ── word vectors ──────────────────────────────────────────────────────────

    def word_vector(self, word: str) -> np.ndarray | None:
        i = self._word_idx.get(word.upper())
        return self._word_vectors[i] if i is not None else None

    def theme_vector(self, theme: str) -> np.ndarray | None:
        i = self._theme_idx.get(theme.upper())
        return self._theme_vectors[i] if i is not None else None

    def word_theme_similarities(self, word: str) -> dict[str, float]:
        """Cosine similarity between one word and ALL known themes."""
        wv = self.word_vector(word)
        if wv is None:
            return {}
        sims = self._theme_vectors @ wv
        return {t: float(sims[i]) for i, t in enumerate(self._themes)}

    def group_coherence(self, members: list[str]) -> float:
        """
        Average pairwise cosine similarity within a group of words.
        High coherence → easy (Yellow). Low coherence → hard (Purple).
        """
        vecs = [self.word_vector(m) for m in members if self.word_vector(m) is not None]
        if len(vecs) < 2:
            return 0.0
        total, count = 0.0, 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                total += float(np.dot(vecs[i], vecs[j]))
                count += 1
        return total / count

    def cross_group_contamination(
        self, members: list[str], other_members: list[str]
    ) -> float:
        """
        Max cosine similarity between any word in `members` and
        any word in `other_members`. High score → red herring potential.
        """
        max_sim = 0.0
        for m in members:
            vm = self.word_vector(m)
            if vm is None:
                continue
            for o in other_members:
                vo = self.word_vector(o)
                if vo is None:
                    continue
                sim = float(np.dot(vm, vo))
                if sim > max_sim:
                    max_sim = sim
        return max_sim

    def nearest_themes(self, word: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Top-k most similar themes to a given word."""
        sims = self.word_theme_similarities(word)
        return sorted(sims.items(), key=lambda x: -x[1])[:top_k]


if __name__ == "__main__":
    main()