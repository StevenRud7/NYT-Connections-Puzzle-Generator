"""
Phase 2 — Pattern Analyzer
Mines recurring NYT Connections trick types and structural patterns
from connections.json. Outputs models/pattern_clusters.json.

Assumes connections.json is fully normalized:
  - All levels are 0/1/2/3 (Yellow/Green/Blue/Purple) in positional order.
  - All member words are uppercase strings.
"""

import json
import re
import os
from collections import defaultdict, Counter
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("data/connections.json")
MODELS_DIR   = Path("ml/models")
OUTPUT_PATH  = MODELS_DIR / "pattern_clusters.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Pattern detection helpers ────────────────────────────────────────────────

# Regex: theme label is "WORD ___" or "___ WORD" (compound word fill-in-the-blank)
RE_PREFIX = re.compile(r"^([A-Z]+)\s+_{2,}$")          # PAPER ___
RE_SUFFIX = re.compile(r"^_{2,}\s+([A-Z]+)$")          # ___ FISH
RE_BOTH   = re.compile(r"^([A-Z]+)\s+_{2,}\s+([A-Z]+)$") # HEAD ___ ACHE (rare)

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

def detect_theme_type(group_label: str, members: list[str]) -> str:
    """
    Classify a group into one of the NYT trick pattern types.
    Returns a string tag.
    """
    label = group_label.upper().strip()

    # 1. Fill-in-the-blank compound word patterns
    if RE_PREFIX.match(label):
        return "suffix_compound"     # e.g. PAPER ___ → CLIP, TRAIL, TOWEL, TIGER
    if RE_SUFFIX.match(label):
        return "prefix_compound"     # e.g. ___ FISH  → SWORD, STAR, CAT, BLOW
    if RE_BOTH.match(label):
        return "sandwich_compound"   # e.g. HEAD ___ ACHE

    # 2. "___ or ___" / "X and Y" connective patterns
    if re.search(r"\bOR\b|\bAND\b", label):
        return "connective_pair"

    # 3. "THINGS THAT ___" / "WAYS TO ___" descriptive phrase
    if re.match(r"^(THINGS|WAYS|WORDS|TYPES|KINDS|FORMS)\b", label):
        return "descriptive_category"

    # 4. "___S" — plural proper noun sets (teams, countries, etc.)
    proper_indicators = ["TEAM","PLAYER","MEMBER","BRAND","NAME","CITY","COUNTRY",
                         "STATE","SINGER","BAND","ACTOR","CHARACTER","SHOW","MOVIE"]
    if any(word in label for word in proper_indicators):
        return "proper_noun_set"

    # 5. Action / verb category  — "EXHIBIT ___", "SHOW ___", "WAYS TO ___"
    if re.match(r"^[A-Z]+\s+[A-Z]", label) and len(label.split()) >= 2:
        first_word = label.split()[0]
        # Heuristic: if the label reads like a verb phrase
        verb_starters = ["EXHIBIT","SHOW","EXPRESS","DESCRIBE","INDICATE","SUGGEST",
                         "REPRESENT","SIGNAL","CONVEY","DEMONSTRATE"]
        if first_word in verb_starters:
            return "verb_phrase_category"

    # 6. Single-word theme (pure synonym set)
    if len(label.split()) == 1:
        return "synonym_set"

    # 7. Domain detection via keyword match in label
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in label for kw in keywords):
            return f"domain_{domain}"

    return "general_category"


def detect_red_herring_potential(all_puzzles: list[dict]) -> dict[str, list[str]]:
    """
    Find words that appear across MULTIPLE different group themes —
    these are natural red herring candidates.
    Returns {word: [group_theme1, group_theme2, ...]}
    """
    word_to_themes: dict[str, list[str]] = defaultdict(list)
    for puzzle in all_puzzles:
        for group in puzzle["answers"]:
            theme = group["group"]
            for word in group["members"]:
                word_to_themes[word].append(theme)

    # Keep only words appearing in 2+ different themes
    return {
        word: list(set(themes))
        for word, themes in word_to_themes.items()
        if len(set(themes)) >= 2
    }


def analyze_difficulty_signals(puzzles: list[dict]) -> dict:
    """
    For each difficulty level 0-3, collect structural signals:
    - Average number of words in theme label
    - Most common theme types at that level
    - Average member word length
    - Presence of red herrings
    """
    level_data: dict[int, dict] = {i: {
        "theme_types": [],
        "theme_word_counts": [],
        "member_word_lengths": [],
        "label_lengths": [],
    } for i in range(4)}

    for puzzle in puzzles:
        for group in puzzle["answers"]:
            lvl = group["level"]
            if lvl not in range(4):
                continue
            label    = group["group"]
            members  = group["members"]
            t_type   = detect_theme_type(label, members)

            level_data[lvl]["theme_types"].append(t_type)
            level_data[lvl]["theme_word_counts"].append(len(label.split()))
            level_data[lvl]["label_lengths"].append(len(label))
            for m in members:
                level_data[lvl]["member_word_lengths"].append(len(m))

    signals = {}
    level_names = {0: "yellow", 1: "green", 2: "blue", 3: "purple"}
    for lvl, data in level_data.items():
        type_counts = Counter(data["theme_types"])
        signals[level_names[lvl]] = {
            "level": lvl,
            "most_common_theme_types": type_counts.most_common(5),
            "avg_theme_word_count":    round(
                sum(data["theme_word_counts"]) / max(len(data["theme_word_counts"]), 1), 2),
            "avg_label_length":        round(
                sum(data["label_lengths"]) / max(len(data["label_lengths"]), 1), 2),
            "avg_member_word_length":  round(
                sum(data["member_word_lengths"]) / max(len(data["member_word_lengths"]), 1), 2),
        }
    return signals


def build_theme_type_index(puzzles: list[dict]) -> dict:
    """
    For each theme type, store example groups so the generator
    can sample real NYT examples as templates.
    Returns {theme_type: [{group, members, level}, ...]}
    """
    index: dict[str, list[dict]] = defaultdict(list)
    for puzzle in puzzles:
        for group in puzzle["answers"]:
            lvl     = group["level"]
            label   = group["group"]
            members = group["members"]
            t_type  = detect_theme_type(label, members)
            index[t_type].append({
                "group":   label,
                "members": members,
                "level":   lvl,
                "puzzle_id": puzzle.get("id"),
            })
    return dict(index)


def extract_compound_roots(puzzles: list[dict]) -> dict:
    """
    For prefix/suffix compound patterns, extract the root word
    and the 4 completing words. Used as generation templates.
    Returns {root_word: {"type": "prefix"|"suffix", "completions": [...], "level": int}}
    """
    roots: dict[str, dict] = {}
    for puzzle in puzzles:
        for group in puzzle["answers"]:
            label   = group["group"]
            members = group["members"]
            lvl     = group["level"]

            m = RE_PREFIX.match(label)
            if m:
                root = m.group(1)
                roots[root] = {"type": "suffix", "completions": members, "level": lvl}
                continue

            m = RE_SUFFIX.match(label)
            if m:
                root = m.group(1)
                roots[root] = {"type": "prefix", "completions": members, "level": lvl}
                continue

    return roots


def build_vocabulary_universe(puzzles: list[dict]) -> dict:
    """
    Full vocabulary: every word that has appeared, with metadata.
    {word: {"levels": [int,...], "groups": [str,...], "frequency": int}}
    """
    universe: dict[str, dict] = defaultdict(lambda: {"levels": [], "groups": [], "frequency": 0})
    for puzzle in puzzles:
        for group in puzzle["answers"]:
            for word in group["members"]:
                universe[word]["levels"].append(group["level"])
                universe[word]["groups"].append(group["group"])
                universe[word]["frequency"] += 1
    return dict(universe)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("  Phase 2 — Pattern Analyzer")
    print("═" * 60)

    # Load data
    print(f"\n[1/6] Loading {DATA_PATH} ...")
    with open(DATA_PATH, "r") as f:
        puzzles = json.load(f)
    print(f"      {len(puzzles)} puzzles loaded.")

    # Red herring words
    print("[2/6] Detecting red herring candidates ...")
    red_herrings = detect_red_herring_potential(puzzles)
    print(f"      {len(red_herrings)} words appear in 2+ different group themes.")

    # Difficulty signals
    print("[3/6] Analyzing difficulty signals per level ...")
    difficulty_signals = analyze_difficulty_signals(puzzles)
    for color, data in difficulty_signals.items():
        top = data["most_common_theme_types"][0] if data["most_common_theme_types"] else ("N/A", 0)
        print(f"      {color.upper():8s} | avg label len: {data['avg_label_length']:5.1f} "
              f"| avg member len: {data['avg_member_word_length']:4.1f} "
              f"| top pattern: {top[0]} ({top[1]})")

    # Theme type index
    print("[4/6] Building theme type index ...")
    theme_index = build_theme_type_index(puzzles)
    print(f"      {len(theme_index)} distinct theme types found:")
    for t, groups in sorted(theme_index.items(), key=lambda x: -len(x[1])):
        print(f"        {t:30s} → {len(groups):4d} examples")

    # Compound roots
    print("[5/6] Extracting compound word roots ...")
    compound_roots = extract_compound_roots(puzzles)
    print(f"      {len(compound_roots)} compound root words extracted.")

    # Vocabulary universe
    print("[6/6] Building vocabulary universe ...")
    vocab = build_vocabulary_universe(puzzles)
    print(f"      {len(vocab)} unique words in universe.")

    # Assemble output
    output = {
        "meta": {
            "total_puzzles": len(puzzles),
            "total_unique_words": len(vocab),
            "total_compound_roots": len(compound_roots),
            "total_red_herring_candidates": len(red_herrings),
            "theme_type_counts": {k: len(v) for k, v in theme_index.items()},
        },
        "difficulty_signals":    difficulty_signals,
        "theme_type_index":      theme_index,
        "compound_roots":        compound_roots,
        "red_herring_candidates": red_herrings,
        "vocabulary_universe":   vocab,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Pattern clusters saved → {OUTPUT_PATH}")
    print("═" * 60)


if __name__ == "__main__":
    main()