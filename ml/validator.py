"""
Phase 6 — Validator (v2)
Quality gate for generated puzzles. Uses the shared morphology utilities
from group_generator so checks are consistent with generation rules.

Checks:
  1.  Exactly 4 groups of 4 members
  2.  All 16 words unique
  3.  One group at each level (0-3)
  4.  No single-word theme label appears verbatim as its own member
      (multi-word descriptive labels are fine — that's intentional NYT style)
  5.  No morphological variants within any group
  6.  Red herring presence (soft warning only)
  7.  Each group coherence ≥ MIN_COHERENCE
  8.  Coherence gradient Yellow ≥ Green ≥ Blue ≥ Purple (soft warning)
  9.  Not a duplicate of a historical puzzle (cosine sim threshold)
  10. No group is an exact match of a historical group
  11. Theme labels distinct from each other
  12. Diversity: ≥ 3 distinct effective types, ≤ 2 of any one type
"""

import json
import sys
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ML_DIR = Path(__file__).parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from group_generator import has_morph_overlap, label_contaminates, FILL_FAMILY, _SessionStore, STRUCTURAL_TYPES
FILL_BLANK_FAMILY = FILL_FAMILY

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH     = Path("data/connections.json")
MODELS_DIR    = Path("ml/models")
CACHE_PATH    = MODELS_DIR / "embeddings_cache.npz"
INDEX_PATH    = MODELS_DIR / "embeddings_index.json"
PATTERNS_PATH = MODELS_DIR / "pattern_clusters.json"

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_COHERENCE       = 0.20
DUPLICATE_THRESHOLD = 0.92
LABEL_SIM_THRESHOLD = 0.95
MIN_DISTINCT_TYPES  = 3
MAX_SAME_TYPE       = 2


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed:        bool
    failures:      list[str] = field(default_factory=list)
    warnings:      list[str] = field(default_factory=list)
    quality_score: float     = 0.0

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines  = [f"[{status}]  quality={self.quality_score:.3f}"]
        for f in self.failures:
            lines.append(f"  FAIL  {f}")
        for w in self.warnings:
            lines.append(f"  WARN  {w}")
        return "\n".join(lines)


# ── Embedding lookup ──────────────────────────────────────────────────────────

class _EmbLookup:
    def __init__(self):
        data        = np.load(CACHE_PATH)
        self.wv     = data["word_vectors"]
        self.tv     = data["theme_vectors"]
        with open(INDEX_PATH) as f:
            idx     = json.load(f)
        self.words  = idx["words"]
        self.themes = idx["themes"]
        self.wi     = {w: i for i, w in enumerate(self.words)}
        self.ti     = {t: i for i, t in enumerate(self.themes)}

    def wvec(self, w):
        i = self.wi.get(w.upper())
        return self.wv[i] if i is not None else None

    def tvec(self, t):
        i = self.ti.get(t.upper())
        return self.tv[i] if i is not None else None

    def centroid(self, members):
        vecs = [self.wvec(m) for m in members if self.wvec(m) is not None]
        if not vecs:
            return None
        c    = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(c)
        return c / norm if norm > 0 else c

    def coherence(self, members):
        vecs = [self.wvec(m) for m in members if self.wvec(m) is not None]
        if len(vecs) < 2:
            return 0.0
        pairs = [float(np.dot(vecs[i], vecs[j]))
                 for i in range(len(vecs))
                 for j in range(i + 1, len(vecs))]
        return float(np.mean(pairs))


# ── Historical index ──────────────────────────────────────────────────────────

class HistoricalIndex:
    def __init__(self, emb: _EmbLookup):
        with open(DATA_PATH) as f:
            puzzles = json.load(f)

        self._sigs: list[np.ndarray]  = []
        self._sets: list[frozenset]   = []

        for p in puzzles:
            centroids = []
            for g in p["answers"]:
                self._sets.append(frozenset(m.upper() for m in g["members"]))
                c = emb.centroid([m.upper() for m in g["members"]])
                if c is not None:
                    centroids.append(c)
            if centroids:
                sig  = np.mean(centroids, axis=0).astype(np.float32)
                norm = np.linalg.norm(sig)
                self._sigs.append(sig / norm if norm > 0 else sig)

        self._mat = np.stack(self._sigs) if self._sigs else np.zeros((0, 1))

    def max_sim(self, sig: np.ndarray) -> float:
        if self._mat.shape[0] == 0:
            return 0.0
        return float(np.max(self._mat @ sig))

    def exact_match(self, members: list[str]) -> bool:
        s = frozenset(m.upper() for m in members)
        return s in self._sets


# ── Validator ─────────────────────────────────────────────────────────────────

class PuzzleValidator:

    def __init__(self):
        print("  Loading validator ...")
        self._emb  = _EmbLookup()
        self._hist = HistoricalIndex(self._emb)
        with open(PATTERNS_PATH) as f:
            p = json.load(f)
        self._rh_set = set(p.get("red_herring_candidates", {}).keys())
        print(f"  Historical index: {len(self._hist._sets)} groups.")

    def _effective_type(self, theme_type: str) -> str:
        return "fill_in_blank" if theme_type in FILL_BLANK_FAMILY else theme_type

    # ── Checks ────────────────────────────────────────────────────────────────

    def _chk_structure(self, p):
        f = []
        if len(p.get("groups", [])) != 4:
            f.append(f"Expected 4 groups, got {len(p.get('groups',[]))}")
            return f
        for g in p["groups"]:
            if len(g.get("members", [])) != 4:
                f.append(f"Group '{g.get('theme')}' has {len(g.get('members',[]))} members")
        return f

    def _chk_unique(self, p):
        words = [m for g in p["groups"] for m in g["members"]]
        if len(words) != len(set(words)):
            dupes = [w for w, c in Counter(words).items() if c > 1]
            return [f"Duplicate words: {dupes}"]
        return []

    def _chk_levels(self, p):
        levels  = {g["level"] for g in p["groups"]}
        missing = {0,1,2,3} - levels
        if missing:
            names = {0:"Yellow",1:"Green",2:"Blue",3:"Purple"}
            return [f"Missing levels: {[names[l] for l in sorted(missing)]}"]
        return []

    def _chk_label_member(self, p):
        """Only flag single-word labels that appear verbatim as a member."""
        f = []
        for g in p["groups"]:
            theme   = g["theme"].upper().strip()
            members = [m.upper() for m in g["members"]]
            if "___" in theme:
                continue
            words = [w for w in theme.split() if len(w) > 2]
            if len(words) == 1 and words[0] in members:
                f.append(f"Single-word theme '{theme}' appears in its own members")
        return f

    def _chk_morph(self, p):
        f = []
        for g in p["groups"]:
            m = [x.upper() for x in g["members"]]
            if has_morph_overlap(m):
                f.append(f"Morphological variants in '{g['theme']}': {m}")
        return f

    def _chk_red_herrings(self, p):
        all_m = [m for g in p["groups"] for m in g["members"]]
        known = [m for m in all_m if m.upper() in self._rh_set]
        if not known and not p.get("red_herrings"):
            return [], ["No confirmed red herring words found."]
        return [], []

    def _chk_coherence(self, p):
        f, w = [], []
        names = {0:"Yellow",1:"Green",2:"Blue",3:"Purple"}
        cohs  = {}
        for g in p["groups"]:
            c = self._emb.coherence([m.upper() for m in g["members"]])
            cohs[g["level"]] = c
            if c < MIN_COHERENCE:
                f.append(f"{names[g['level']]} '{g['theme']}' coherence {c:.3f} < {MIN_COHERENCE}")
        for lvl in range(3):
            if lvl in cohs and (lvl+1) in cohs:
                if cohs[lvl] < cohs[lvl+1] - 0.05:
                    w.append(f"Coherence gradient: {names[lvl]}({cohs[lvl]:.3f}) < "
                             f"{names[lvl+1]}({cohs[lvl+1]:.3f})")
        return f, w

    def _chk_duplicate(self, p):
        centroids = [self._emb.centroid([m.upper() for m in g["members"]])
                     for g in p["groups"]]
        centroids = [c for c in centroids if c is not None]
        if not centroids:
            return []
        sig  = np.mean(centroids, axis=0).astype(np.float32)
        norm = np.linalg.norm(sig)
        if norm > 0:
            sig = sig / norm
        sim = self._hist.max_sim(sig)
        if sim >= DUPLICATE_THRESHOLD:
            return [f"Too similar to a historical puzzle (sim={sim:.3f})"]
        return []

    def _chk_exact_match(self, p):
        f = []
        for g in p["groups"]:
            if self._hist.exact_match(g["members"]):
                f.append(f"Group '{g['theme']}' is an exact historical group.")
        return f

    def _chk_label_sim(self, p):
        f      = []
        themes = [g["theme"].upper() for g in p["groups"]]
        for i in range(len(themes)):
            for j in range(i+1, len(themes)):
                ti = self._emb.tvec(themes[i])
                tj = self._emb.tvec(themes[j])
                if ti is not None and tj is not None:
                    sim = float(np.dot(ti, tj))
                    if sim >= LABEL_SIM_THRESHOLD:
                        f.append(f"Themes too similar: '{themes[i]}' ↔ '{themes[j]}' ({sim:.3f})")
        return f

    def _chk_diversity(self, p):
        f      = []
        eff    = [self._effective_type(g.get("theme_type","general_category"))
                  for g in p["groups"]]
        counts = Counter(eff)
        if max(counts.values()) > MAX_SAME_TYPE:
            f.append(f"Too many groups of same type: {dict(counts)}")
        if len(counts) < MIN_DISTINCT_TYPES:
            f.append(f"Too few distinct theme types: {len(counts)} (need {MIN_DISTINCT_TYPES})")
        return f

    # ── Quality score ─────────────────────────────────────────────────────────

    def _quality(self, p) -> float:
        cohs = [self._emb.coherence([m.upper() for m in g["members"]])
                for g in p["groups"]]
        spread     = (cohs[0] - cohs[-1]) if len(cohs) == 4 else 0.0
        spread_s   = min(max(spread, 0.0), 0.5) / 0.5
        all_m      = [m for g in p["groups"] for m in g["members"]]
        rh_s       = min(sum(1 for m in all_m if m.upper() in self._rh_set) / 4.0, 1.0)
        eff        = {self._effective_type(g.get("theme_type","general_category"))
                      for g in p["groups"]}
        div_s      = len(eff) / 4.0
        return round((spread_s + rh_s + div_s) / 3.0, 4)

    # ── Master validate ───────────────────────────────────────────────────────

    def validate(self, puzzle: dict) -> ValidationResult:
        failures, warnings = [], []

        structural = self._chk_structure(puzzle)
        if structural:
            return ValidationResult(passed=False, failures=structural)

        failures += self._chk_unique(puzzle)
        failures += self._chk_levels(puzzle)
        failures += self._chk_label_member(puzzle)
        failures += self._chk_morph(puzzle)
        failures += self._chk_duplicate(puzzle)
        failures += self._chk_exact_match(puzzle)
        failures += self._chk_label_sim(puzzle)
        failures += self._chk_diversity(puzzle)

        f, w = self._chk_red_herrings(puzzle)
        failures += f; warnings += w

        f, w = self._chk_coherence(puzzle)
        failures += f; warnings += w

        return ValidationResult(
            passed=len(failures) == 0,
            failures=failures,
            warnings=warnings,
            quality_score=self._quality(puzzle),
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("puzzle_file")
    args = parser.parse_args()
    with open(args.puzzle_file) as f:
        puzzle = json.load(f)
    v = PuzzleValidator()
    r = v.validate(puzzle)
    print(r)
    sys.exit(0 if r.passed else 1)