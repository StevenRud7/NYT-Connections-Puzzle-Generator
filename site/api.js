/**
 * api.js
 * Fetch layer for the Connections Puzzle Generator.
 * Tries the live API first; falls back to static puzzle JSON
 * so the site works even when the backend is offline.
 */

const API_BASE = 'http://localhost:8000/api';

// ── Fallback puzzle ───────────────────────────────────────────────────────────
// A single hardcoded puzzle used when the API is unreachable.
// Replace members/themes as desired for offline demos.
const FALLBACK_PUZZLE = {
  id: 'fallback-001',
  groups: [
    {
      level: 0, color: 'yellow', hex: '#F9DF6D',
      theme: 'PAPER ___',
      members: ['CLIP', 'TRAIL', 'TOWEL', 'TIGER'],
      theme_type: 'fill_in_blank', coherence: 0.55,
    },
    {
      level: 1, color: 'green', hex: '#A0C35A',
      theme: 'EXHIBIT NERVOUSNESS',
      members: ['BLUSH', 'FIDGET', 'PACE', 'SWEAT'],
      theme_type: 'category_member', coherence: 0.45,
    },
    {
      level: 2, color: 'blue', hex: '#B0C4EF',
      theme: 'THINGS THAT CAN RUN, ANNOYINGLY',
      members: ['DYE', 'MASCARA', 'NOSE', 'STOCKINGS'],
      theme_type: 'associated_with', coherence: 0.38,
    },
    {
      level: 3, color: 'purple', hex: '#BA81C5',
      theme: 'EVALUATE',
      members: ['GRADE', 'RANK', 'RATE', 'SCORE'],
      theme_type: 'double_meaning', coherence: 0.30,
    },
  ],
  words_shuffled: ['CLIP','BLUSH','DYE','GRADE','TRAIL','FIDGET',
                   'MASCARA','RANK','TOWEL','PACE','NOSE','RATE',
                   'TIGER','SWEAT','STOCKINGS','SCORE'],
  red_herrings: ['SCORE', 'PACE'],
};

// ── API calls ─────────────────────────────────────────────────────────────────

/**
 * Fetch a fresh generated puzzle from the API.
 * Returns puzzle data object or null on failure.
 */
export async function fetchNewPuzzle() {
  try {
    const res = await fetch(`${API_BASE}/generate`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({}),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return normalisePuzzle(data.puzzle);
  } catch (err) {
    console.warn('[api] fetchNewPuzzle failed — using fallback.', err);
    return null;
  }
}

/**
 * Fetch a random stored puzzle from the API.
 */
export async function fetchRandomPuzzle() {
  try {
    const res = await fetch(`${API_BASE}/puzzle/random`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return normalisePuzzle(data);
  } catch (err) {
    console.warn('[api] fetchRandomPuzzle failed — using fallback.', err);
    return null;
  }
}

/**
 * Primary loader: tries random stored puzzle, then generates new,
 * then falls back to the hardcoded puzzle.
 */
export async function loadPuzzle() {
  // 1. Try a stored puzzle (fast — no generation needed)
  const stored = await fetchRandomPuzzle();
  if (stored) return { puzzle: stored, source: 'stored' };

  // 2. Try generating a fresh one
  const generated = await fetchNewPuzzle();
  if (generated) return { puzzle: generated, source: 'generated' };

  // 3. Offline fallback
  console.warn('[api] Using offline fallback puzzle.');
  return { puzzle: normalisePuzzle(FALLBACK_PUZZLE), source: 'fallback' };
}

/**
 * Validate a guess against the API.
 * Returns { correct, one_away, group, message } or null.
 */
export async function submitGuess(puzzleId, words) {
  try {
    const res = await fetch(`${API_BASE}/puzzle/${puzzleId}/guess`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ puzzle_id: puzzleId, words }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.warn('[api] submitGuess failed — falling back to local check.', err);
    return null;  // caller falls back to local validation
  }
}

// ── Normalisation ─────────────────────────────────────────────────────────────
// Ensures the puzzle object always has a shuffled words array
// regardless of which endpoint returned it.

function normalisePuzzle(p) {
  if (!p.words_shuffled || p.words_shuffled.length === 0) {
    const words = p.groups.flatMap(g => g.members);
    p.words_shuffled = shuffle([...words]);
  }
  return p;
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}