"""
Phase 4 — Difficulty Classifier
Trains a model that predicts difficulty level (0=Yellow → 3=Purple)
for any candidate group, given its theme label and 4 member words.

Features used:
  - Group coherence (avg pairwise cosine sim between members)
  - Theme-member similarity (avg cosine sim of members → theme label)
  - Cross-group contamination proxy (variance of theme-member sims)
  - Theme type one-hot (from pattern_analyzer)
  - Label length, member word length stats
  - Red herring potential (how many members appear in 2+ themes)

Outputs:
  ml/models/difficulty_classifier.pkl  — serialized sklearn pipeline
  ml/models/difficulty_report.json     — evaluation metrics

Run:
  pip install scikit-learn sentence-transformers numpy
  python ml/difficulty_model.py
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH       = Path("data/connections.json")
MODELS_DIR      = Path("ml/models")
PATTERNS_PATH   = MODELS_DIR / "pattern_clusters.json"
CACHE_PATH      = MODELS_DIR / "embeddings_cache.npz"
INDEX_PATH      = MODELS_DIR / "embeddings_index.json"
CLF_PATH        = MODELS_DIR / "difficulty_classifier.pkl"
REPORT_PATH     = MODELS_DIR / "difficulty_report.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Theme type list (must stay consistent with pattern_analyzer.py) ──────────
THEME_TYPES = [
    "suffix_compound", "prefix_compound", "sandwich_compound",
    "connective_pair", "descriptive_category", "proper_noun_set",
    "verb_phrase_category", "synonym_set",
    "domain_sports", "domain_food", "domain_music", "domain_geography",
    "domain_pop_culture", "domain_language", "domain_science", "domain_money",
    "general_category",
]

import re
RE_PREFIX = re.compile(r"^([A-Z]+)\s+_{2,}$")
RE_SUFFIX = re.compile(r"^_{2,}\s+([A-Z]+)$")
RE_BOTH   = re.compile(r"^([A-Z]+)\s+_{2,}\s+([A-Z]+)$")

DOMAIN_KEYWORDS = {
    "sports":      ["TEAM","PLAYER","SPORT","GAME","LEAGUE","COACH","BALL","CUP","RACE","MATCH"],
    "food":        ["FOOD","DISH","MEAL","INGREDIENT","COOK","BAKE","EAT","DRINK","RECIPE","CUISINE"],
    "music":       ["SONG","BAND","ARTIST","ALBUM","NOTE","CHORD","GENRE","LYRIC","TUNE","BEAT"],
    "geography":   ["COUNTRY","CITY","CAPITAL","RIVER","MOUNTAIN","STATE","CONTINENT","ISLAND","OCEAN","LAKE"],
    "pop_culture": ["MOVIE","FILM","TV","SHOW","CHARACTER","ACTOR","CELEBRITY","BRAND","GAME","BOOK"],
    "language":    ["WORD","PHRASE","LETTER","SYNONYM","RHYME","HOMOPHONE","PREFIX","SUFFIX","SLANG","IDIOM"],
    "science":     ["ELEMENT","CHEMICAL","BIOLOGY","PHYSICS","MATH","FORMULA","ATOM","CELL","GENE","PLANET"],
    "money":       ["MONEY","PRICE","COST","PAY","BANK","STOCK","TAX","COIN","FEE","DOLLAR"],
}


def detect_theme_type(label: str, members: list[str]) -> str:
    """Mirror of pattern_analyzer.detect_theme_type — kept local to avoid circular imports."""
    label = label.upper().strip()
    if RE_PREFIX.match(label):   return "suffix_compound"
    if RE_SUFFIX.match(label):   return "prefix_compound"
    if RE_BOTH.match(label):     return "sandwich_compound"
    if re.search(r"\bOR\b|\bAND\b", label): return "connective_pair"
    if re.match(r"^(THINGS|WAYS|WORDS|TYPES|KINDS|FORMS)\b", label): return "descriptive_category"
    proper = ["TEAM","PLAYER","MEMBER","BRAND","NAME","CITY","COUNTRY","STATE",
              "SINGER","BAND","ACTOR","CHARACTER","SHOW","MOVIE"]
    if any(w in label for w in proper): return "proper_noun_set"
    if re.match(r"^[A-Z]+\s+[A-Z]", label):
        verb_starters = ["EXHIBIT","SHOW","EXPRESS","DESCRIBE","INDICATE","SUGGEST",
                         "REPRESENT","SIGNAL","CONVEY","DEMONSTRATE"]
        if label.split()[0] in verb_starters: return "verb_phrase_category"
    if len(label.split()) == 1: return "synonym_set"
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in label for kw in keywords): return f"domain_{domain}"
    return "general_category"


# ── Embedding helpers ─────────────────────────────────────────────────────────

class _EmbeddingLookup:
    """Minimal embedding lookup — avoids importing embeddings.py to keep things clean."""

    def __init__(self):
        data = np.load(CACHE_PATH)
        self.wv = data["word_vectors"]   # (W, D)
        self.tv = data["theme_vectors"]  # (T, D)
        with open(INDEX_PATH) as f:
            idx = json.load(f)
        self.words  = idx["words"]
        self.themes = idx["themes"]
        self.wi     = {w: i for i, w in enumerate(self.words)}
        self.ti     = {t: i for i, t in enumerate(self.themes)}

    def word(self, w: str) -> np.ndarray | None:
        i = self.wi.get(w.upper())
        return self.wv[i] if i is not None else None

    def theme(self, t: str) -> np.ndarray | None:
        i = self.ti.get(t.upper())
        return self.tv[i] if i is not None else None


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(
    theme_label: str,
    members: list[str],
    emb: _EmbeddingLookup,
    red_herring_set: set[str],
) -> np.ndarray:
    """
    Build a fixed-length feature vector for one group.

    Feature layout (total = 7 + len(THEME_TYPES)):
      [0]  group_coherence          — avg pairwise cosine sim of members
      [1]  theme_member_sim_mean    — avg cosine sim(member, theme_vec)
      [2]  theme_member_sim_std     — std of above (spread = ambiguity)
      [3]  theme_member_sim_min     — worst-fitting member
      [4]  avg_member_word_length
      [5]  label_word_count
      [6]  red_herring_count        — how many members are red herring words
      [7:] theme_type_one_hot
    """
    # ── Coherence ────────────────────────────────────────────────────────────
    vecs = [emb.word(m) for m in members if emb.word(m) is not None]
    if len(vecs) >= 2:
        pairs = [(np.dot(vecs[i], vecs[j]))
                 for i in range(len(vecs))
                 for j in range(i + 1, len(vecs))]
        coherence = float(np.mean(pairs))
    else:
        coherence = 0.0

    # ── Theme-member similarity ───────────────────────────────────────────────
    tv = emb.theme(theme_label)
    if tv is not None:
        tm_sims = [float(np.dot(emb.word(m), tv))
                   for m in members if emb.word(m) is not None]
    else:
        tm_sims = [0.0]
    tm_mean = float(np.mean(tm_sims))
    tm_std  = float(np.std(tm_sims))
    tm_min  = float(np.min(tm_sims))

    # ── Surface features ──────────────────────────────────────────────────────
    avg_len     = float(np.mean([len(m) for m in members]))
    label_words = float(len(theme_label.split()))
    rh_count    = float(sum(1 for m in members if m.upper() in red_herring_set))

    # ── Theme type one-hot ────────────────────────────────────────────────────
    t_type = detect_theme_type(theme_label, members)
    one_hot = [1.0 if t == t_type else 0.0 for t in THEME_TYPES]

    return np.array(
        [coherence, tm_mean, tm_std, tm_min, avg_len, label_words, rh_count] + one_hot,
        dtype=np.float32,
    )


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(
    puzzles: list[dict],
    emb: _EmbeddingLookup,
    red_herring_set: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns X (features), y (labels 0-3) for all groups in the dataset.
    Skips groups with level outside 0-3.
    """
    X_rows, y_rows = [], []
    skipped = 0

    for puzzle in puzzles:
        for group in puzzle["answers"]:
            lvl = group["level"]
            if lvl not in range(4):
                skipped += 1
                continue
            feat = extract_features(
                group["group"], group["members"], emb, red_herring_set
            )
            X_rows.append(feat)
            y_rows.append(lvl)

    if skipped:
        print(f"      Warning: {skipped} groups skipped (invalid level).")

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("  Phase 4 — Difficulty Classifier")
    print("═" * 60)

    # 1. Load data
    print(f"\n[1/5] Loading data ...")
    with open(DATA_PATH) as f:
        puzzles = json.load(f)
    with open(PATTERNS_PATH) as f:
        patterns = json.load(f)
    print(f"      {len(puzzles)} puzzles loaded.")

    # 2. Setup embeddings + red herring set
    print("[2/5] Loading embedding cache ...")
    emb = _EmbeddingLookup()
    red_herring_set = set(patterns.get("red_herring_candidates", {}).keys())
    print(f"      {len(red_herring_set)} red herring candidates.")

    # 3. Build feature matrix
    print("[3/5] Extracting features ...")
    X, y = build_dataset(puzzles, emb, red_herring_set)
    print(f"      Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    from collections import Counter
    dist = Counter(y.tolist())
    level_names = {0: "Yellow", 1: "Green", 2: "Blue", 3: "Purple"}
    for lvl in range(4):
        print(f"        Level {lvl} ({level_names[lvl]:6s}): {dist.get(lvl, 0):4d} samples")

    # 4. Train
    print("\n[4/5] Training GradientBoostingClassifier ...")
    clf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])

    # Cross-validate first for honest metrics
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        clf_pipeline, X, y, cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
    )
    print(f"      CV accuracy : {np.mean(cv_results['test_accuracy']):.3f} "
          f"± {np.std(cv_results['test_accuracy']):.3f}")
    print(f"      CV F1-macro : {np.mean(cv_results['test_f1_macro']):.3f} "
          f"± {np.std(cv_results['test_f1_macro']):.3f}")

    # Fit on full dataset for production use
    clf_pipeline.fit(X, y)

    # Full-dataset classification report
    y_pred = clf_pipeline.predict(X)
    report = classification_report(
        y, y_pred,
        target_names=[level_names[i] for i in range(4)],
        output_dict=True,
    )
    print("\n      Full-dataset classification report:")
    for name in [level_names[i] for i in range(4)]:
        r = report[name]
        print(f"        {name:6s} | P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1-score']:.2f}")

    # 5. Save
    print(f"\n[5/5] Saving model → {CLF_PATH}")
    with open(CLF_PATH, "wb") as f:
        pickle.dump(clf_pipeline, f)

    report_out = {
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std":  float(np.std(cv_results["test_accuracy"])),
        "cv_f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
        "cv_f1_macro_std":  float(np.std(cv_results["test_f1_macro"])),
        "full_dataset_report": report,
        "feature_names": [
            "group_coherence", "theme_member_sim_mean", "theme_member_sim_std",
            "theme_member_sim_min", "avg_member_word_length", "label_word_count",
            "red_herring_count",
        ] + THEME_TYPES,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report_out, f, indent=2)
    print(f"      Report saved → {REPORT_PATH}")

    print("\n✓ Difficulty classifier trained and saved.")
    print("═" * 60)


# ── Public API ────────────────────────────────────────────────────────────────

class DifficultyClassifier:
    """
    Inference wrapper used by puzzle_assembler and validator.

    Usage:
        clf = DifficultyClassifier()
        level = clf.predict(theme_label="PAPER ___", members=["CLIP","TRAIL","TOWEL","TIGER"])
        proba = clf.predict_proba(...)
    """

    LEVEL_NAMES = {0: "yellow", 1: "green", 2: "blue", 3: "purple"}

    def __init__(
        self,
        clf_path:   Path = CLF_PATH,
        cache_path: Path = CACHE_PATH,
        index_path: Path = INDEX_PATH,
        patterns_path: Path = PATTERNS_PATH,
    ):
        with open(clf_path, "rb") as f:
            self._pipeline = pickle.load(f)
        self._emb = _EmbeddingLookup()
        with open(patterns_path) as f:
            patterns = json.load(f)
        self._rh_set = set(patterns.get("red_herring_candidates", {}).keys())

    def _featurize(self, theme_label: str, members: list[str]) -> np.ndarray:
        return extract_features(
            theme_label, members, self._emb, self._rh_set
        ).reshape(1, -1)

    def predict(self, theme_label: str, members: list[str]) -> int:
        """Returns predicted level 0-3."""
        return int(self._pipeline.predict(self._featurize(theme_label, members))[0])

    def predict_proba(self, theme_label: str, members: list[str]) -> dict[str, float]:
        """Returns probability distribution over all 4 levels."""
        proba = self._pipeline.predict_proba(self._featurize(theme_label, members))[0]
        return {self.LEVEL_NAMES[i]: float(p) for i, p in enumerate(proba)}

    def predict_color(self, theme_label: str, members: list[str]) -> str:
        """Returns color string: 'yellow' | 'green' | 'blue' | 'purple'."""
        return self.LEVEL_NAMES[self.predict(theme_label, members)]


if __name__ == "__main__":
    main()