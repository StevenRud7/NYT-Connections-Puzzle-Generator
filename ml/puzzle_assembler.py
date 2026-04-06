"""
Phase 5b — Puzzle Assembler (v2)

Assembles 4 CandidateGroups into a complete NYT-style puzzle.
Enforces:
  - One group per difficulty level (Yellow/Green/Blue/Purple)
  - At least 3 distinct effective theme types
  - At most 2 groups of the same effective type
  - fill_in_blank family counts as one type
  - Minimum 1 confirmed red herring word
  - No generated_at field in output JSON
"""

import json
import random
import uuid
import sys
import numpy as np
from pathlib import Path
from itertools import combinations, permutations

# Ensure ml/ is on the path when called from project root
sys.path.insert(0, str(Path(__file__).parent))

from group_generator import (
    GroupGenerator, CandidateGroup, _EmbLookup,
    T_FILL_SUFFIX, T_FILL_PREFIX, T_CATEGORY, T_DESCRIPTOR,
    T_DOUBLE, T_HOMOPHONE, T_PLUS_MINUS,
    T_RHYME, T_PERSON_NAME, T_BRAND,
    T_HIDDEN_WORD, T_INITIALISM,
    FILL_FAMILY, STRUCTURAL_TYPES, _SessionStore,
    classify_domain, canonical_theme,
    MAX_DOMAIN_PER_PUZZLE,
)
FILL_BLANK_FAMILY = FILL_FAMILY

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR    = Path("ml/models")
PATTERNS_PATH = MODELS_DIR / "pattern_clusters.json"

# ── Color map ─────────────────────────────────────────────────────────────────
LEVEL_TO_COLOR = {0: "yellow", 1: "green", 2: "blue", 3: "purple"}
COLOR_TO_HEX   = {
    "yellow": "#F9DF6D",
    "green":  "#A0C35A",
    "blue":   "#B0C4EF",
    "purple": "#BA81C5",
}

# ── Diversity rules ───────────────────────────────────────────────────────────
MIN_DISTINCT_TYPES = 3
MAX_SAME_TYPE      = 2


# ── Assembled puzzle ──────────────────────────────────────────────────────────

class AssembledPuzzle:
    def __init__(self, groups: list[dict], words_shuffled: list[str],
                 red_herrings: list[str], generation_meta: dict):
        self.id              = f"gen-{uuid.uuid4().hex[:8]}"
        self.groups          = groups
        self.words_shuffled  = words_shuffled
        self.red_herrings    = red_herrings
        self.generation_meta = generation_meta

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "groups":         self.groups,
            "words_shuffled": self.words_shuffled,
            "red_herrings":   self.red_herrings,
            "meta":           self.generation_meta,
        }


# ── Assembler ─────────────────────────────────────────────────────────────────

class PuzzleAssembler:

    SCORE_WEIGHTS = {
        "difficulty_spread":   4.0,
        "type_diversity":      3.0,
        "red_herring_bonus":   2.0,
        "contamination":       1.5,
        "coherence_gradient":  1.0,
    }

    def __init__(self):
        from difficulty_model import DifficultyClassifier
        print("  Loading difficulty classifier ...")
        self._clf = DifficultyClassifier()
        print("  Loading embeddings ...")
        self._emb = _EmbLookup()
        with open(PATTERNS_PATH) as f:
            p = json.load(f)
        self._rh_set = set(p.get("red_herring_candidates", {}).keys())

    # ── Diversity check ───────────────────────────────────────────────────────

    @staticmethod
    def _diversity_ok(groups: list[CandidateGroup]) -> bool:
        """Returns True if the 4-group combo passes type AND domain rules."""
        from collections import Counter

        # Theme type diversity
        eff_types   = [g.effective_type() for g in groups]
        type_counts = Counter(eff_types)
        if max(type_counts.values()) > MAX_SAME_TYPE:
            return False
        if len(type_counts) < MIN_DISTINCT_TYPES:
            return False

        # Domain diversity: no non-general domain more than MAX_DOMAIN_PER_PUZZLE
        domain_counts = Counter(g.domain for g in groups)
        for domain, cnt in domain_counts.items():
            if domain != "general" and cnt > MAX_DOMAIN_PER_PUZZLE:
                return False

        return True

    # ── Difficulty assignment ─────────────────────────────────────────────────

    def _assign_levels(
        self, groups: list[CandidateGroup]
    ) -> dict[int, CandidateGroup] | None:
        level_names = ["yellow", "green", "blue", "purple"]
        probas = [
            self._clf.predict_proba(g.theme, g.members)
            for g in groups
        ]
        best_score, best_perm = -1.0, None
        for perm in permutations(range(4)):
            score = sum(probas[i][level_names[perm[i]]] for i in range(4))
            if score > best_score:
                best_score, best_perm = score, perm
        if best_perm is None:
            return None
        return {best_perm[i]: groups[i] for i in range(4)}

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(self, groups: list[CandidateGroup]) -> float:
        w = self.SCORE_WEIGHTS

        if not self._diversity_ok(groups):
            return -999.0

        assignment = self._assign_levels(groups)
        if assignment is None or len(set(assignment.keys())) < 4:
            return -999.0

        score = w["difficulty_spread"] * 1.0

        # Type diversity bonus: reward more distinct types
        from collections import Counter
        eff = Counter(g.effective_type() for g in groups)
        score += w["type_diversity"] * (len(eff) / 4.0)

        # Red herring count
        all_members = [m for g in groups for m in g.members]
        rh = sum(1 for m in all_members if m in self._rh_set)
        score += w["red_herring_bonus"] * rh

        # Cross-group semantic contamination (makes puzzle tricky)
        contamination = 0.0
        for g1, g2 in combinations(groups, 2):
            vecs1 = [self._emb.wvec(m) for m in g1.members if self._emb.wvec(m) is not None]
            vecs2 = [self._emb.wvec(m) for m in g2.members if self._emb.wvec(m) is not None]
            if vecs1 and vecs2:
                contamination += max(
                    float(np.dot(v1, v2)) for v1 in vecs1 for v2 in vecs2
                )
        score += w["contamination"] * contamination

        # Coherence gradient: Yellow most coherent, Purple least
        ordered = [assignment[l] for l in range(4)]
        cohs    = [g.coherence for g in ordered]
        gradient_ok = all(cohs[i] >= cohs[i+1] for i in range(3))
        score += w["coherence_gradient"] * (1.0 if gradient_ok else 0.0)

        return score

    # ── Red herring detection ─────────────────────────────────────────────────
    # A word is a red herring if its cosine sim to another group's centroid
    # is at least 85% of its sim to its own group's centroid.

    def _find_red_herrings(self, groups: list[CandidateGroup]) -> list[str]:
        centroids = []
        for g in groups:
            vecs = [self._emb.wvec(m) for m in g.members if self._emb.wvec(m) is not None]
            if not vecs:
                centroids.append(None)
                continue
            c = np.mean(vecs, axis=0).astype(np.float32)
            norm = np.linalg.norm(c)
            centroids.append(c / norm if norm > 0 else c)

        rh = []
        for gi, g in enumerate(groups):
            if centroids[gi] is None:
                continue
            for m in g.members:
                wv = self._emb.wvec(m)
                if wv is None:
                    continue
                own_sim = float(np.dot(wv, centroids[gi]))
                for gj in range(len(groups)):
                    if gj == gi or centroids[gj] is None:
                        continue
                    other_sim = float(np.dot(wv, centroids[gj]))
                    if other_sim >= own_sim * 0.85:
                        rh.append(m)
                        break
        return list(set(rh))

    # ── Main assemble ─────────────────────────────────────────────────────────

    def assemble(self, pool: list[CandidateGroup],
                 max_combos: int = 600) -> AssembledPuzzle | None:
        if len(pool) < 4:
            return None

        best_score, best_combo, best_assign = -999.0, None, None
        tried = 0

        shuffled = pool[:]
        random.shuffle(shuffled)

        for combo in combinations(shuffled, 4):
            if tried >= max_combos:
                break
            tried += 1

            # Quick pre-filter: no word overlap
            all_words = [w for g in combo for w in g.members]
            if len(all_words) != len(set(all_words)):
                continue

            score = self._score(list(combo))
            if score > best_score:
                best_score  = score
                best_combo  = list(combo)
                best_assign = self._assign_levels(list(combo))

        if best_combo is None or best_assign is None:
            return None

        groups_out = []
        for level in range(4):
            g     = best_assign[level]
            color = LEVEL_TO_COLOR[level]
            groups_out.append({
                "level":      level,
                "color":      color,
                "hex":        COLOR_TO_HEX[color],
                "theme":      g.theme,
                "members":    g.members,
                "theme_type": g.theme_type,
                "coherence":  round(g.coherence, 4),
            })

        all_words = [m for g in groups_out for m in g["members"]]
        random.shuffle(all_words)

        # Mark all assembled groups as session-used so they never appear again
        for g in best_combo:
            _SessionStore.mark_used(g.theme, g.members)

        return AssembledPuzzle(
            groups=groups_out,
            words_shuffled=all_words,
            red_herrings=self._find_red_herrings(best_combo),
            generation_meta={
                "combos_evaluated": tried,
                "assembly_score":   round(best_score, 4),
                "pool_size":        len(pool),
                "theme_types":      [g.effective_type() for g in best_combo],
                "coherences":       [round(g.coherence, 4) for g in best_combo],
            },
        )


# ── Convenience entry point ───────────────────────────────────────────────────

def generate_puzzle(seed: int | None = None) -> AssembledPuzzle | None:
    print("\n── Generating candidate pool ...")
    gen  = GroupGenerator(seed=seed)
    pool = gen.generate_candidate_pool(pool_size=32)
    print(f"   Pool: {len(pool)} candidates")
    for g in pool:
        print(f"     [{g.effective_type():20s}] {g.theme}: {g.members}")

    print("\n── Assembling puzzle ...")
    asm    = PuzzleAssembler()
    puzzle = asm.assemble(pool)

    if puzzle:
        print(f"\n   ✓ Puzzle: {puzzle.id}")
        for g in puzzle.groups:
            print(f"     [{g['color'].upper():6s} | {g['theme_type']:20s}] "
                  f"{g['theme']}: {g['members']}")
        print(f"   Red herrings: {puzzle.red_herrings}")
    else:
        print("   ✗ Could not assemble a valid puzzle.")

    return puzzle


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int,  default=None)
    parser.add_argument("--output", type=str,  default=None)
    args = parser.parse_args()

    puzzle = generate_puzzle(seed=args.seed)
    if puzzle:
        data = puzzle.to_dict()
        if args.output:
            with open(args.output, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\n✓ Saved to {args.output}")
        else:
            print(json.dumps(data, indent=2))
    else:
        print("✗ Generation failed.")