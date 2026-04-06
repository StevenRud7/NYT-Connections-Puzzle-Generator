"""
Phase 5a — Group Generator (v7)

COMPLETE REWRITE addressing all known issues:

DEDUPLICATION  (the core fix)
  - Theme labels are normalised before storage: punctuation removed, dots
    stripped, extra whitespace collapsed, all uppercase.
    "NHL TEAM MEMBER" and "N.H.L. TEAM MEMBER" both normalise to
    "NHL TEAM MEMBER" and are recognised as the same theme.
  - Member-set frozensets are also tracked.
  - Persistent file backing survives server restarts.

DOMAIN DIVERSITY  (stops single idea flooding)
  - A domain classifier labels every group with one of ~15 content domains.
  - The assembler receives domain counts and blocks any domain appearing
    more than MAX_DOMAIN_PER_PUZZLE (default 1) times in a single puzzle.
  - Sports, geography, food etc. each get at most one slot.

CORRECT PLURAL / SINGULAR
  - Groups are sourced directly from verified historical data.
  - Non-structural groups are mutated by exactly 1 member using embeddings.
  - _KNOWN_WRONG_MEMBERS: a blocklist of specific words known to be wrong
    for certain theme patterns (CONGRESS, OCTOPUS for NHL, etc.)
  - plural_required(theme): returns True when the theme label itself implies
    plural members (team MEMBERS, ___ FANS etc.)

STRUCTURAL TYPES NEVER MUTATED
  - Homophone, rhyme, plus/minus, hidden word, initialism, double meaning.
  - Used verbatim from verified historical entries.

TWELVE THEME TYPES
  fill_suffix    WORD ___
  fill_prefix    ___ WORD
  category       Named category (non biased)
  descriptor     Things that ___ / Ways to ___
  double_meaning Word in two domains
  homophone      Sounds like another word
  plus_minus     Add/remove a letter
  rhyme          Rhymes with ___
  person_name    Also a person's name
  brand          Brand / company (non biased)
  hidden_word    Contains a hidden word
  initialism     Abbreviation / acronym
"""

import json
import random
import re
import unicodedata
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR    = Path("ml/models")
PATTERNS_PATH = MODELS_DIR / "pattern_clusters.json"
CACHE_PATH    = MODELS_DIR / "embeddings_cache.npz"
INDEX_PATH    = MODELS_DIR / "embeddings_index.json"
DATA_PATH     = Path("data/connections.json")
SESSION_PATH  = MODELS_DIR / "session_used.json"

MEMBERS_PER_GROUP     = 4
MIN_COHERENCE         = 0.17
MAX_COHERENCE         = 0.93
MIN_MEMBER_SIM        = 0.09
MAX_DOMAIN_PER_PUZZLE = 1    # at most one group per content domain per puzzle

# ── Theme type constants ──────────────────────────────────────────────────────
T_FILL_SUFFIX = "fill_suffix"
T_FILL_PREFIX = "fill_prefix"
T_CATEGORY    = "category"
T_DESCRIPTOR  = "descriptor"
T_DOUBLE      = "double_meaning"
T_HOMOPHONE   = "homophone"
T_PLUS_MINUS  = "plus_minus"
T_RHYME       = "rhyme"
T_PERSON_NAME = "person_name"
T_BRAND       = "brand"
T_HIDDEN_WORD = "hidden_word"
T_INITIALISM  = "initialism"

FILL_FAMILY      = {T_FILL_SUFFIX, T_FILL_PREFIX}
STRUCTURAL_TYPES = {T_HOMOPHONE, T_PLUS_MINUS, T_RHYME, T_HIDDEN_WORD,
                    T_INITIALISM, T_DOUBLE}

STRATEGY_WEIGHTS = {
    T_FILL_SUFFIX: 0.18,
    T_FILL_PREFIX: 0.13,
    T_CATEGORY:    0.22,
    T_DESCRIPTOR:  0.15,
    T_DOUBLE:      0.07,
    T_HOMOPHONE:   0.06,
    T_PLUS_MINUS:  0.05,
    T_RHYME:       0.04,
    T_PERSON_NAME: 0.04,
    T_BRAND:       0.03,
    T_HIDDEN_WORD: 0.02,
    T_INITIALISM:  0.01,
}

# ── Domain classifier ─────────────────────────────────────────────────────────
# Maps a theme label to a broad content domain. Used to prevent one puzzle
# being flooded with sports themes or geography themes etc.
_DOMAIN_PATTERNS = [
    ("sports",     re.compile(
        r'\b(NHL|NBA|NFL|MLB|WNBA|MLS|FIFA|NCAA|SPORT|TEAM|LEAGUE|PLAYER'
        r'|ATHLETE|COACH|GAME|CHAMPIONSHIP|TOURNAMENT|PITCHER|BATTER'
        r'|QUARTERBACK|RECEIVER|GOALKEEPER|REFEREE|STADIUM|ARENA'
        r'|HOCKEY|BASKETBALL|BASEBALL|FOOTBALL|SOCCER|TENNIS|GOLF'
        r'|SWIMMING|GYMNASTICS|BOXING|WRESTLING|CYCLING)\b'
    )),
    ("geography",  re.compile(
        r'\b(COUNTRY|COUNTRIES|CAPITAL|CITY|CITIES|RIVER|MOUNTAIN|STATE'
        r'|CONTINENT|ISLAND|OCEAN|LAKE|REGION|PROVINCE|TERRITORY'
        r'|AFRICA|EUROPE|ASIA|AMERICA|AUSTRALIA|ANTARCTICA)\b'
    )),
    ("food",       re.compile(
        r'\b(FOOD|DISH|MEAL|INGREDIENT|COOK|BAKE|EAT|DRINK|RECIPE'
        r'|CUISINE|FLAVOR|TASTE|RESTAURANT|MENU|DESSERT|APPETIZER'
        r'|PASTA|SAUCE|SPICE|FRUIT|VEGETABLE|MEAT|FISH|CHEESE|BREAD)\b'
    )),
    ("music",      re.compile(
        r'\b(SONG|BAND|ARTIST|ALBUM|NOTE|CHORD|GENRE|LYRIC|TUNE|BEAT'
        r'|MUSICIAN|COMPOSER|INSTRUMENT|GUITAR|PIANO|DRUM|JAZZ|ROCK'
        r'|POP|CLASSICAL|RAP|HIP.HOP|OPERA|SYMPHONY)\b'
    )),
    ("film_tv",    re.compile(
        r'\b(MOVIE|FILM|TV|SHOW|SERIES|ACTOR|ACTRESS|DIRECTOR|CHARACTER'
        r'|SCENE|EPISODE|SEQUEL|ANIMATED|CARTOON|DOCUMENTARY|SITCOM'
        r'|DRAMA|COMEDY|THRILLER|HORROR|SUPERHERO)\b'
    )),
    ("science",    re.compile(
        r'\b(ELEMENT|CHEMICAL|BIOLOGY|PHYSICS|CHEMISTRY|FORMULA|ATOM'
        r'|CELL|GENE|PLANET|STAR|GALAXY|MOLECULE|COMPOUND|REACTION'
        r'|EXPERIMENT|LABORATORY|SCIENTIST|THEORY|HYPOTHESIS)\b'
    )),
    ("literature", re.compile(
        r'\b(BOOK|NOVEL|AUTHOR|WRITER|POEM|POET|PLAY|PLAYWRIGHT'
        r'|CHARACTER|CHAPTER|GENRE|FICTION|NONFICTION|BIOGRAPHY)\b'
    )),
    ("animals",    re.compile(
        r'\b(ANIMAL|MAMMAL|BIRD|FISH|REPTILE|INSECT|AMPHIBIAN|DOG|CAT'
        r'|HORSE|BEAR|LION|TIGER|ELEPHANT|WHALE|SHARK|SNAKE|SPIDER)\b'
    )),
    ("history",    re.compile(
        r'\b(PRESIDENT|MONARCH|KING|QUEEN|EMPEROR|DYNASTY|EMPIRE|WAR'
        r'|REVOLUTION|ANCIENT|MEDIEVAL|COLONIAL|HISTORICAL|CENTURY)\b'
    )),
    ("tech",       re.compile(
        r'\b(COMPUTER|SOFTWARE|HARDWARE|INTERNET|APP|DEVICE|PHONE'
        r'|DIGITAL|ONLINE|WEBSITE|SOCIAL MEDIA|ALGORITHM|CODE|PROGRAM)\b'
    )),
    ("fashion",    re.compile(
        r'\b(FASHION|CLOTHING|GARMENT|FABRIC|DESIGNER|STYLE|TREND'
        r'|OUTFIT|DRESS|SUIT|SHOE|ACCESSORY|JEWELRY|BRAND)\b'
    )),
    ("wordplay",   re.compile(
        r'\b(HOMOPHONE|RHYME|ANAGRAM|PALINDROME|ACRONYM|ABBREVIATION'
        r'|SOUNDS LIKE|LETTER|HIDDEN|CONTAINS|INITIALS)\b'
    )),
    ("names",      re.compile(
        r'\b(NAME|FIRST NAME|LAST NAME|SURNAME|NICKNAME|ALIAS)\b'
    )),
    ("nature",     re.compile(
        r'\b(WEATHER|SEASON|CLIMATE|PLANT|TREE|FLOWER|FOREST|OCEAN'
        r'|MOUNTAIN|DESERT|JUNGLE|RIVER|LAKE|SKY|SUN|MOON|STAR)\b'
    )),
]

def classify_domain(theme_label: str) -> str:
    lbl = theme_label.upper()
    for domain, pattern in _DOMAIN_PATTERNS:
        if pattern.search(lbl):
            return domain
    return "general"


# ── Theme label canonicaliser ─────────────────────────────────────────────────
def canonical_theme(label: str) -> str:
    """
    Normalise a theme label for dedup purposes:
      - Uppercase
      - Remove all punctuation except underscores
      - Collapse all whitespace to single space
      - Strip leading/trailing space
    Examples:
      "N.H.L. TEAM MEMBER" → "NHL TEAM MEMBER"
      "N.B.A.  TEAM  MEMBER" → "NBA TEAM MEMBER"
      "Things That Can Run, Annoyingly" → "THINGS THAT CAN RUN ANNOYINGLY"
    """
    s = label.upper()
    # Remove all punctuation except underscores and spaces
    s = re.sub(r'[^\w\s]', '', s)
    # Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ── Stop words / force-plural / blocklist ─────────────────────────────────────
_STOP = {
    "A","AN","THE","OF","IN","ON","AT","TO","FOR","WITH","BY","AS",
    "IS","ARE","WAS","BE","OR","AND","IT","ITS","THAT","EACH","ALSO",
    "USED","WORDS","THINGS","WAYS","CAN","MAKE","ADD","WHICH",
    "BEFORE","AFTER","BOTH","RHYMES","SOUNDS","LIKE","HIDDEN","CONTAINS",
    "EACH","TWO","THREE","FOUR","LETTER","LETTERS",
}

_FORCE_PLURAL = {
    "PANTS","GLASSES","SCISSORS","TONGS","JEANS","SHORTS","TIGHTS",
    "LEGGINGS","TWEEZERS","PLIERS","GOGGLES","BINOCULARS","TROUSERS",
    "THANKS","REGARDS","REMAINS","RICHES","STAIRS","OUTSKIRTS",
    "PREMISES","PROCEEDS","GOODS","ODDS","CONTENTS","WHEREABOUTS",
    "MEANS","SPECIES","SERIES","NEWS","MATHEMATICS","PHYSICS","ECONOMICS",
    "ATHLETICS","LINGUISTICS","ETHICS","POLITICS","ELECTRONICS",
    "CLOTHES","SUDS","DREGS","ANNALS","ARCHIVES","ENTRAILS","INNARDS",
    "SAVINGS","WINNINGS","EARNINGS","TAKINGS","HOLDINGS","DEALINGS",
    "PAGES","ARMS","LEGS","BLUES","GREENS","REDS","WHITES","BLACKS",
    "BLUES","BEARS","BULLS","JETS","NETS","KNICKS","SPURS","CELTICS",
    "ROCKETS","RAPTORS","KINGS","HAWKS","WOLVES","SIXERS","SUNS",
}

# Words that are flat-out wrong regardless of context
_BLOCKLIST = {
    "NU","MU","XI","KI","QI","ZA","AA","AE","AI","OE","OI","OU",
    "AB","AD","AG","AH","AL","AM","AR","AW","AX","AY","BA","BI",
    "BO","BY","DA","DE","ED","EF","EH","EL","EM","EN","ER",
    "ES","ET","EW","EX","FA","FE","GI","GU","HA","HI",
    "HM","ID","IF","IO","JO","KA","LA","LI","LO","MA","ME",
    "MI","MM","MO","MY","NA","NE","OD","OH","OM","OW","OX","OY",
    "PA","PE","PI","PO","RE","SH","SI","TA","TI","UT","WO",
    "XU","YA","YE","YO","ZO",
    # Known bad outputs
    "CHING","OBELISK","EXPIRATION","EXPIATE","CONGRESS","OCTOPUS",
    "CONTINUATION","ABBREVIATION","PRONUNCIATION","ADMINISTRATION",
    # Things that appear as wrong sports members via mutation
    "SENATE","PARLIAMENT","GOVERNMENT","LEGISLATURE","ASSEMBLY",
}

MIN_WORD_LEN = 3

# ── Words that need plural form for specific theme patterns ───────────────────
def needs_plural(theme_label: str, word: str, vocab: set) -> str:
    """
    Given a theme label and a word, return the correct form (plural or singular).
    - If label contains MEMBERS / TEAMS / FANS → plural preferred
    - If label contains ___ and the compound is standard in plural → plural
    - Otherwise → singular preferred (via smart_normalise)
    """
    lbl = theme_label.upper()
    w   = word.upper()

    # Themes that intrinsically describe plural things
    plural_triggers = [
        "MEMBERS","TEAMS","FANS","PLAYERS","GAMES","SPORTS",
        "BRANDS","COMPANIES","COUNTRIES","CITIES","SONGS","MOVIES",
        "BOOKS","SHOWS","AWARDS","INSTRUMENTS",
    ]
    if any(t in lbl for t in plural_triggers):
        # Try to pluralise if not already
        if not w.endswith('S') and w + 'S' in vocab:
            return w + 'S'
        if not w.endswith('S') and w + 'ES' in vocab:
            return w + 'ES'
    return w


def smart_normalise(word: str, vocab: set, theme_label: str = "") -> str:
    """
    Choose the most appropriate form of a word for a given theme context.
    Uses theme label to decide whether plural is needed.
    Never called on fill-blank completions (those keep original form).
    """
    w = word.upper().strip()
    if w in _FORCE_PLURAL:
        return w

    # Check if this theme needs plural members
    correct = needs_plural(theme_label, w, vocab)
    if correct != w:
        return correct

    # Default: prefer singular for clean presentation
    if (w.endswith('S') and not w.endswith('SS') and not w.endswith('US')
            and not w.endswith('IS') and not w.endswith('AS') and len(w) > 4):
        singular = w[:-1]
        if singular in vocab and singular != w:
            return singular
        if w.endswith('ES') and len(w) > 5:
            s2 = w[:-2]
            if s2 in vocab:
                return s2
    return w


# ── Persistent session store ──────────────────────────────────────────────────
class _SessionStore:
    """
    Dedup store with canonical theme normalisation.
    "NHL TEAM MEMBER" and "N.H.L. TEAM MEMBER" are treated as identical.
    Persisted to disk so dedup survives restarts.
    """
    _loaded:            bool = False
    _used_canonical:    set  = set()   # canonical theme labels
    _used_combos:       set  = set()   # frozensets of member words

    @classmethod
    def _load(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if SESSION_PATH.exists():
            try:
                with open(SESSION_PATH) as f:
                    data = json.load(f)
                cls._used_canonical = set(data.get("canonical_themes", []))
                cls._used_combos    = {
                    frozenset(c) for c in data.get("combos", [])
                }
            except Exception:
                cls._used_canonical = set()
                cls._used_combos    = set()

    @classmethod
    def _save(cls) -> None:
        try:
            with open(SESSION_PATH, "w") as f:
                json.dump({
                    "canonical_themes": sorted(cls._used_canonical),
                    "combos":           [sorted(c) for c in cls._used_combos],
                }, f, indent=2)
        except Exception:
            pass

    @classmethod
    def is_used(cls, theme: str, members: list) -> bool:
        cls._load()
        canon = canonical_theme(theme)
        if canon in cls._used_canonical:
            return True
        if frozenset(m.upper() for m in members) in cls._used_combos:
            return True
        return False

    @classmethod
    def mark_used(cls, theme: str, members: list) -> None:
        cls._load()
        cls._used_canonical.add(canonical_theme(theme))
        cls._used_combos.add(frozenset(m.upper() for m in members))
        cls._save()

    @classmethod
    def reset(cls) -> None:
        cls._used_canonical.clear()
        cls._used_combos.clear()
        cls._loaded = True
        cls._save()


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

    def nearest_words(self, qv, top_k=60, exclude=None):
        sims  = self.wv @ qv
        order = np.argsort(-sims)
        out, seen = [], set()
        for i in order:
            w = self.words[i]
            if w in seen: continue
            seen.add(w)
            if exclude and w in exclude:
                continue
            out.append((w, float(sims[i])))
            if len(out) >= top_k:
                break
        return out

    def coherence(self, members):
        vecs = [self.wvec(m) for m in members if self.wvec(m) is not None]
        if len(vecs) < 2:
            return 0.0
        pairs = [float(np.dot(vecs[i], vecs[j]))
                 for i in range(len(vecs))
                 for j in range(i + 1, len(vecs))]
        return float(np.mean(pairs))

    def centroid(self, members):
        vecs = [self.wvec(m) for m in members if self.wvec(m) is not None]
        if not vecs: return None
        c = np.mean(vecs, axis=0).astype(np.float32)
        n = np.linalg.norm(c)
        return c / n if n > 0 else c


# ── Word quality helpers ──────────────────────────────────────────────────────
_MORPH_SUFFIXES = [
    "ING","ED","ER","EST","LY","FUL","LESS","NESS","MENT","ION",
    "TION","ATION","IZE","ISE","IFY","EN","S","ES","D","N","Y",
]

def _root_forms(word: str) -> set:
    w = word.upper()
    forms = {w}
    for suf in _MORPH_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[:-len(suf)]
            forms.add(base)
            for s2 in _MORPH_SUFFIXES:
                forms.add(base + s2)
        forms.add(w + suf)
    return forms

def has_morph_overlap(members: list) -> bool:
    for i in range(len(members)):
        fi = _root_forms(members[i])
        for j in range(i + 1, len(members)):
            if members[j].upper() in fi:
                return True
    return False

def label_contaminates(label: str, members: list) -> bool:
    label_words = [
        w for w in re.sub(r"_{2,}", " ", label).upper().split()
        if w not in _STOP and len(w) > 2
    ]
    mem_set = {m.upper() for m in members}
    for lw in label_words:
        if lw in mem_set:
            return True
        if any(m in _root_forms(lw) for m in mem_set):
            return True
    return False

def is_real_word(word: str, vocab: set) -> bool:
    w = word.upper().strip()
    if len(w) < MIN_WORD_LEN:       return False
    if w in _BLOCKLIST:             return False
    if re.match(r'^[A-Z]{1,2}$', w) and w not in {
        "GO","DO","BE","UP","OK","NO","SO","US","WE","HE","IT",
    }:                              return False
    return w in vocab

def member_fits_group(word: str, members: list, emb: _EmbLookup) -> bool:
    others = [m for m in members if m.upper() != word.upper()]
    if len(others) < 2: return True
    c = emb.centroid(others)
    if c is None: return True
    wv = emb.wvec(word)
    if wv is None: return True
    return float(np.dot(wv, c)) >= MIN_MEMBER_SIM


# ── Candidate group ───────────────────────────────────────────────────────────
class CandidateGroup:
    __slots__ = ("theme", "members", "theme_type", "coherence", "source", "domain")

    def __init__(self, theme, members, theme_type, coherence,
                 source="generated", domain=None):
        self.theme      = theme.upper().strip()
        self.members    = [m.upper().strip() for m in members]
        self.theme_type = theme_type
        self.coherence  = coherence
        self.source     = source
        self.domain     = domain or classify_domain(self.theme)

    def __repr__(self):
        return (f"CandidateGroup({self.theme!r}, {self.members}, "
                f"{self.theme_type}/{self.domain}, coh={self.coherence:.3f})")

    def effective_type(self):
        return T_FILL_SUFFIX if self.theme_type in FILL_FAMILY else self.theme_type

    def to_dict(self):
        return {
            "theme":      self.theme,
            "members":    self.members,
            "theme_type": self.theme_type,
            "coherence":  round(self.coherence, 4),
            "source":     self.source,
            "domain":     self.domain,
        }


# ── Group Generator ───────────────────────────────────────────────────────────
class GroupGenerator:

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        _SessionStore._load()

        print("  Loading patterns ...")
        with open(PATTERNS_PATH) as f:
            self._pat = json.load(f)

        print("  Loading embeddings ...")
        self._emb = _EmbLookup()

        print("  Building historical index ...")
        with open(DATA_PATH) as f:
            hist = json.load(f)

        self._hist_sets: set = {
            frozenset(g["members"])
            for p in hist for g in p["answers"]
        }
        self._vocab: set = {
            m.upper()
            for p in hist for g in p["answers"] for m in g["members"]
        }

        # ALL 12 types pre-initialised (prevents KeyError for any type)
        self._buckets: dict = {t: [] for t in STRATEGY_WEIGHTS}

        # ── Partition historical groups into buckets ───────────────────────
        # Regex matchers in priority order (more specific first)
        matchers = [
            (T_FILL_SUFFIX,  None),          # handled by ___ detection below
            (T_FILL_PREFIX,  None),           # handled by ___ detection below
            (T_HIDDEN_WORD,  re.compile(
                r'\b(HIDDEN|CONTAINS|CONCEALS|INSIDE|EACH HIDES)\b')),
            (T_INITIALISM,   re.compile(
                r'\b(ABBREVIAT|INITIALS|ACRONYM|INITIALISM|ABBREV)\b')),
            (T_HOMOPHONE,    re.compile(
                r'\b(HOMOPHONE|SOUNDS? LIKE|PRONOUNCED LIKE)\b')),
            (T_RHYME,        re.compile(
                r'\bRHYMES?\b')),
            (T_PLUS_MINUS,   re.compile(
                r'\b(ADD A LETTER|REMOVE A LETTER|PLUS ONE LETTER'
                r'|MINUS ONE LETTER|WITHOUT THE LETTER|INSERT A LETTER'
                r'|ADD ONE|REMOVE ONE|TAKE AWAY)\b')),
            (T_PERSON_NAME,  re.compile(
                r'\b(ALSO A (FIRST |LAST |GIVEN )?NAME|(FIRST|LAST|GIVEN|FULL) NAME'
                r'|SURNAME|PERSON\'S NAME|FAMOUS [A-Z]+S?)\b')),
            (T_BRAND,        re.compile(
                # Match brand / company / league names
                # Deliberately conservative — avoid sports team buckets
                # being over-selected
                r'\b(BRAND|LOGO|SLOGAN|ADVERTISING|COMMERCIAL)\b')),
            (T_DOUBLE,       re.compile(
                r'\b(ALSO (USED|MEANS|A)|ANOTHER WORD|DOUBLE MEANING'
                r'|TWO MEANINGS|SLANG FOR|CAN ALSO)\b')),
            (T_DESCRIPTOR,   re.compile(
                r'^(THINGS|WORDS|WAYS|TYPES|KINDS|PARTS|FORMS|ITEMS'
                r'|PHRASES|EXPRESSIONS|SOMETHING)\b')),
        ]

        for p in hist:
            for g in p["answers"]:
                lbl     = g["group"].upper()
                members = [m.upper() for m in g["members"]]
                e = {"group": lbl, "members": members, "level": g["level"],
                     "domain": classify_domain(lbl)}

                # Fill-blank detection first
                if '___' in lbl:
                    if re.match(r'^_{2,}', lbl.strip()):
                        self._buckets[T_FILL_PREFIX].append(e)
                    else:
                        self._buckets[T_FILL_SUFFIX].append(e)
                    continue

                # Try each matcher in order
                matched = False
                for btype, pattern in matchers[2:]:  # skip fill entries
                    if pattern and pattern.search(lbl):
                        self._buckets[btype].append(e)
                        matched = True
                        break

                if not matched:
                    # Classify into category vs brand based on domain
                    domain = classify_domain(lbl)
                    if domain == "general":
                        self._buckets[T_CATEGORY].append(e)
                    elif domain in ("sports", "geography", "animals",
                                    "music", "film_tv", "literature",
                                    "history", "science", "tech",
                                    "food", "nature", "names", "fashion"):
                        # Rich domain-tagged categories go into T_CATEGORY
                        self._buckets[T_CATEGORY].append(e)
                    else:
                        self._buckets[T_CATEGORY].append(e)

        self._compound_roots = self._pat.get("compound_roots", {})
        counts = {t: len(v) for t, v in self._buckets.items()}
        print(f"  Buckets: {counts}")

    # ── Core validity ─────────────────────────────────────────────────────────

    def _valid(self, theme: str, members: list, used: set,
               is_structural: bool = False) -> bool:
        m = [x.upper() for x in members]
        u = {x.upper() for x in used}

        if len(m) != MEMBERS_PER_GROUP:      return False
        if len(set(m)) != MEMBERS_PER_GROUP: return False
        if any(w in u for w in m):           return False
        if any(not is_real_word(w, self._vocab) for w in m):
            return False
        if has_morph_overlap(m):             return False
        if label_contaminates(theme, m):     return False

        if not is_structural:
            for word in m:
                if not member_fits_group(word, m, self._emb):
                    return False

        coh = self._emb.coherence(m)
        if not (MIN_COHERENCE <= coh <= MAX_COHERENCE): return False
        if frozenset(m) in self._hist_sets:  return False
        if _SessionStore.is_used(theme, m):  return False
        return True

    # ── Normalise members ─────────────────────────────────────────────────────

    def _norm(self, members: list, used: set,
              preserve_form: bool = False,
              theme_label: str = "") -> list:
        out = []
        u   = {x.upper() for x in used}
        for m in members:
            w = m.upper()
            if preserve_form:
                if is_real_word(w, self._vocab) and w not in u and w not in _BLOCKLIST:
                    out.append(w)
            else:
                n = smart_normalise(w, self._vocab, theme_label)
                if is_real_word(n, self._vocab) and n not in u and n not in _BLOCKLIST:
                    out.append(n)
                elif is_real_word(w, self._vocab) and w not in u and w not in _BLOCKLIST:
                    out.append(w)
        return out

    # ── Mutation (semantic/category types only) ───────────────────────────────

    def _mutate(self, label: str, members: list, used: set,
                n: int = 1) -> list | None:
        m = [x.upper() for x in members]
        swapped = 0
        indices = list(range(len(m)))
        random.shuffle(indices)

        for idx in indices:
            if swapped >= n:
                break
            wv = self._emb.wvec(m[idx])
            if wv is None:
                continue
            block = set(m) | used | {m[idx]}
            for new_w, _ in self._emb.nearest_words(wv, top_k=100, exclude=block):
                new_w_norm = smart_normalise(new_w, self._vocab, label)
                if not is_real_word(new_w_norm, self._vocab): continue
                if new_w_norm in _BLOCKLIST:                  continue
                trial = m[:]
                trial[idx] = new_w_norm
                if frozenset(trial) in self._hist_sets:       continue
                if _SessionStore.is_used(label, trial):       continue
                if has_morph_overlap(trial):                   continue
                if label_contaminates(label, trial):           continue
                if not member_fits_group(new_w_norm, trial, self._emb): continue
                m = trial
                swapped += 1
                break

        return m if swapped > 0 else None

    # ── Generic bucket pull ───────────────────────────────────────────────────

    def _from_bucket(self, bkey: str, used: set,
                     domain_counts: dict | None = None) -> 'CandidateGroup | None':
        """
        Pull one group from a bucket.
        - Structural types: no mutation, preserve original word forms
        - Semantic types: mutate 1 member, apply smart_normalise
        - domain_counts: if provided, skip groups whose domain would exceed
          MAX_DOMAIN_PER_PUZZLE
        """
        is_struct = bkey in STRUCTURAL_TYPES
        preserve  = is_struct or bkey in FILL_FAMILY
        dc        = domain_counts or {}

        cands = [
            g for g in self._buckets[bkey]
            if not any(m in used for m in g["members"])
            and canonical_theme(g["group"]) not in _SessionStore._used_canonical
        ]
        random.shuffle(cands)

        for tmpl in cands[:80]:
            label  = tmpl["group"]
            domain = tmpl.get("domain", classify_domain(label))

            # Domain throttle
            if dc.get(domain, 0) >= MAX_DOMAIN_PER_PUZZLE:
                continue

            members = self._norm(tmpl["members"], used,
                                 preserve_form=preserve,
                                 theme_label=label)
            if len(members) < MEMBERS_PER_GROUP:
                continue
            members = members[:MEMBERS_PER_GROUP]

            if is_struct:
                # Structural: never mutate — use as verified
                if frozenset(members) in self._hist_sets:
                    continue
            else:
                # Semantic: mutate 1 to ensure novelty
                mutated = self._mutate(label, members, used, n=1)
                if mutated is None:
                    continue
                members = mutated

            if self._valid(label, members, used, is_structural=is_struct):
                coh = self._emb.coherence(members)
                grp = CandidateGroup(label, members, bkey, coh, bkey, domain)
                _SessionStore.mark_used(label, members)
                return grp

        return None

    # ── Fill strategies ───────────────────────────────────────────────────────

    def _gen_fill(self, fill_type: str, used: set,
                  domain_counts: dict | None = None) -> 'CandidateGroup | None':
        """
        Shared implementation for fill_suffix and fill_prefix.
        Members are taken verbatim from compound_roots (no normalisation).
        Falls back to bucket if compound_roots exhausted.
        """
        want_suffix = (fill_type == T_FILL_SUFFIX)
        ctype       = "suffix" if want_suffix else "prefix"
        dc          = domain_counts or {}

        roots = [
            r for r, info in self._compound_roots.items()
            if info.get("type") == ctype
            and sum(1 for w in info["completions"]
                    if w.upper() not in used
                    and is_real_word(w.upper(), self._vocab)
                    and w.upper() not in _BLOCKLIST
                    ) >= MEMBERS_PER_GROUP
        ]
        random.shuffle(roots)

        for root in roots[:30]:
            label  = f"{root} ___" if want_suffix else f"___ {root}"
            canon  = canonical_theme(label)
            if canon in _SessionStore._used_canonical:
                continue
            domain = classify_domain(label)
            if dc.get(domain, 0) >= MAX_DOMAIN_PER_PUZZLE:
                continue

            info  = self._compound_roots[root]
            comps = list({
                w.upper()
                for w in info["completions"]
                if w.upper() not in used
                and is_real_word(w.upper(), self._vocab)
                and len(w) >= MIN_WORD_LEN
                and w.upper() not in _BLOCKLIST
            })
            for _ in range(40):
                if len(comps) < MEMBERS_PER_GROUP:
                    break
                subset = random.sample(comps, MEMBERS_PER_GROUP)
                if (frozenset(subset) not in self._hist_sets
                        and not _SessionStore.is_used(label, subset)
                        and self._valid(label, subset, used, is_structural=True)):
                    coh = self._emb.coherence(subset)
                    grp = CandidateGroup(label, subset, fill_type, coh,
                                         "compound", domain)
                    _SessionStore.mark_used(label, subset)
                    return grp

        return self._from_bucket(fill_type, used, domain_counts)

    def _gen_fill_suffix(self, used, dc=None):
        return self._gen_fill(T_FILL_SUFFIX, used, dc)

    def _gen_fill_prefix(self, used, dc=None):
        return self._gen_fill(T_FILL_PREFIX, used, dc)

    # ── Per-type dispatch ─────────────────────────────────────────────────────

    def _gen_category(self, used, dc=None):
        return self._from_bucket(T_CATEGORY, used, dc)

    def _gen_descriptor(self, used, dc=None):
        return self._from_bucket(T_DESCRIPTOR, used, dc)

    def _gen_double(self, used, dc=None):
        return self._from_bucket(T_DOUBLE, used, dc)

    def _gen_homophone(self, used, dc=None):
        return self._from_bucket(T_HOMOPHONE, used, dc)

    def _gen_plus_minus(self, used, dc=None):
        return self._from_bucket(T_PLUS_MINUS, used, dc)

    def _gen_rhyme(self, used, dc=None):
        return self._from_bucket(T_RHYME, used, dc)

    def _gen_person_name(self, used, dc=None):
        return self._from_bucket(T_PERSON_NAME, used, dc)

    def _gen_brand(self, used, dc=None):
        return self._from_bucket(T_BRAND, used, dc)

    def _gen_hidden_word(self, used, dc=None):
        return self._from_bucket(T_HIDDEN_WORD, used, dc)

    def _gen_initialism(self, used, dc=None):
        return self._from_bucket(T_INITIALISM, used, dc)

    # ── Master generate ───────────────────────────────────────────────────────

    def generate_group(self, strategy=None, used=None,
                       type_counts=None, max_per_type=2,
                       domain_counts=None) -> 'CandidateGroup | None':
        used        = {w.upper() for w in (used or set())}
        type_counts = type_counts or {}
        dc          = domain_counts or {}

        def eff(t): return T_FILL_SUFFIX if t in FILL_FAMILY else t

        eligible = {t: w for t, w in STRATEGY_WEIGHTS.items()
                    if type_counts.get(eff(t), 0) < max_per_type}
        if not eligible:
            eligible = dict(STRATEGY_WEIGHTS)

        dispatch = {
            T_FILL_SUFFIX:  self._gen_fill_suffix,
            T_FILL_PREFIX:  self._gen_fill_prefix,
            T_CATEGORY:     self._gen_category,
            T_DESCRIPTOR:   self._gen_descriptor,
            T_DOUBLE:       self._gen_double,
            T_HOMOPHONE:    self._gen_homophone,
            T_PLUS_MINUS:   self._gen_plus_minus,
            T_RHYME:        self._gen_rhyme,
            T_PERSON_NAME:  self._gen_person_name,
            T_BRAND:        self._gen_brand,
            T_HIDDEN_WORD:  self._gen_hidden_word,
            T_INITIALISM:   self._gen_initialism,
        }

        if strategy and strategy in dispatch and \
                type_counts.get(eff(strategy), 0) < max_per_type:
            order = [strategy] + random.choices(
                list(eligible.keys()), weights=list(eligible.values()), k=8)
        else:
            order = random.choices(
                list(eligible.keys()), weights=list(eligible.values()), k=10)

        for s in order:
            fn = dispatch.get(s)
            if fn:
                result = fn(used, dc)
                if result is not None:
                    return result
        return None

    # ── Pool generation ───────────────────────────────────────────────────────

    def generate_candidate_pool(self, pool_size=32, used=None,
                                max_per_type=2) -> list:
        used_up      = {w.upper() for w in (used or set())}
        pool         = []
        type_counts  = {}
        domain_counts= {}

        def eff(g): return g.effective_type()

        # First pass: try to get one of each major type (up to max_per_type)
        priority = [
            T_FILL_SUFFIX, T_FILL_PREFIX, T_CATEGORY, T_DESCRIPTOR,
            T_DOUBLE, T_HOMOPHONE, T_PLUS_MINUS, T_RHYME,
            T_PERSON_NAME, T_BRAND,
        ]
        for t in priority:
            if len(pool) >= pool_size:
                break
            g = self.generate_group(
                strategy=t, used=used_up,
                type_counts=type_counts, max_per_type=max_per_type,
                domain_counts=domain_counts,
            )
            if g:
                pool.append(g)
                used_up.update(g.members)
                type_counts[eff(g)]     = type_counts.get(eff(g), 0) + 1
                domain_counts[g.domain] = domain_counts.get(g.domain, 0) + 1

        # Second pass: fill remaining slots
        for _ in range(pool_size * 8):
            if len(pool) >= pool_size:
                break
            g = self.generate_group(
                used=used_up,
                type_counts=type_counts, max_per_type=max_per_type,
                domain_counts=domain_counts,
            )
            if g is None:
                continue
            if any(w in used_up for w in g.members):
                continue
            pool.append(g)
            used_up.update(g.members)
            type_counts[eff(g)]     = type_counts.get(eff(g), 0) + 1
            domain_counts[g.domain] = domain_counts.get(g.domain, 0) + 1

        return pool