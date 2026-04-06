/**
 * app.js
 * Master game controller.
 * Wires Board, Lives, result bars, toast, and modal together.
 * Handles all game state transitions.
 */

import { loadPuzzle, submitGuess } from './api.js';
import { Board }         from './components/board.js';
import { Lives }         from './components/lives.js';
import { renderResultBar } from './components/result-bar.js';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $loading      = document.getElementById('loading');
const $loadingMsg   = document.getElementById('loading-msg');
const $boardEl      = document.getElementById('board');
const $solvedGroups = document.getElementById('solved-groups');
const $livesDots    = document.getElementById('lives-dots');
const $toastArea    = document.getElementById('toast-area');
const $btnShuffle   = document.getElementById('btn-shuffle');
const $btnDeselect  = document.getElementById('btn-deselect');
const $btnSubmit    = document.getElementById('btn-submit');
const $btnNew       = document.getElementById('btn-new');
const $modal        = document.getElementById('modal');
const $modalEmoji   = document.getElementById('modal-emoji');
const $modalTitle   = document.getElementById('modal-title');
const $modalSub     = document.getElementById('modal-subtitle');
const $shareGrid    = document.getElementById('share-grid');
const $btnShare     = document.getElementById('btn-share');
const $btnPlayAgain = document.getElementById('btn-play-again');

// ── Game state ────────────────────────────────────────────────────────────────
let _puzzle       = null;
let _board        = null;
let _lives        = null;
let _solvedLevels = [];     // levels solved in order (for share grid)
let _guessHistory = [];     // [{words, correct, color}] for share grid
let _gameOver     = false;

const LOADING_MESSAGES = [
  'Picking words carefully…',
  'Hiding the red herrings…',
  'Arranging the tiles…',
  'Almost there…',
];

// ── Boot ──────────────────────────────────────────────────────────────────────

async function boot() {
  showLoading(true);
  cycleLoadingMessage();

  const { puzzle, source } = await loadPuzzle();
  _puzzle = puzzle;

  if (source === 'fallback') {
    console.info('[game] Running on fallback puzzle (API unreachable).');
  }

  initGame();
  showLoading(false);
}

function initGame() {
  // Reset state
  _solvedLevels = [];
  _guessHistory = [];
  _gameOver     = false;

  // Clear DOM
  $solvedGroups.innerHTML = '';
  $boardEl.innerHTML      = '';
  $modal.classList.add('hidden');

  // Board
  _board = new Board($boardEl, _puzzle, {
    onSelectionChange: handleSelectionChange,
  });

  // Lives
  _lives = new Lives($livesDots, 4);

  // Controls initial state
  updateControls(0);
}

// ── Selection ─────────────────────────────────────────────────────────────────

function handleSelectionChange(selected) {
  updateControls(selected.length);
}

function updateControls(selCount) {
  $btnDeselect.disabled = selCount === 0 || _gameOver;
  $btnSubmit.disabled   = selCount !== 4  || _gameOver;
  $btnShuffle.disabled  = _gameOver;
}

// ── Submit guess ──────────────────────────────────────────────────────────────

async function handleSubmit() {
  if (_board.selectionCount !== 4 || _gameOver) return;

  const words = _board.selectedWords;

  // Disable board while processing
  _board.disable();
  $btnSubmit.disabled = true;

  // Try API validation first, fall back to local
  let result = await submitGuess(_puzzle.id, words);
  if (!result) result = localValidate(words);

  await processResult(result, words);

  if (!_gameOver) {
    _board.enable();
    updateControls(_board.selectionCount);
  }
}

// ── Local validation (API fallback) ──────────────────────────────────────────

function localValidate(words) {
  const guessed = new Set(words.map(w => w.toUpperCase()));

  let bestOverlap = 0;
  let matchedGroup = null;

  for (const g of _puzzle.groups) {
    const groupWords = new Set(g.members.map(m => m.toUpperCase()));
    const overlap    = [...guessed].filter(w => groupWords.has(w)).length;
    if (overlap > bestOverlap) {
      bestOverlap  = overlap;
      matchedGroup = g;
    }
  }

  if (bestOverlap === 4) {
    return { correct: true, one_away: false, group: matchedGroup,
             message: `Correct! ${matchedGroup.theme}` };
  }
  if (bestOverlap === 3) {
    return { correct: false, one_away: true, group: null,
             message: 'One away!' };
  }
  return { correct: false, one_away: false, group: null,
           message: 'Incorrect. Try again.' };
}

// ── Process result ────────────────────────────────────────────────────────────

async function processResult(result, words) {
  _guessHistory.push({
    words,
    correct: result.correct,
    color:   result.correct ? result.group.color : 'miss',
  });

  if (result.correct) {
    const group = result.group;
    // Mark tiles as solved (animated)
    await _board.markSolved(group.members);
    // Render result bar
    renderResultBar($solvedGroups, group);
    _solvedLevels.push(group.level);

    showToast(`✓ ${group.theme}`, 1800);

    // Check win
    if (_solvedLevels.length === 4) {
      _gameOver = true;
      await delay(500);
      showEndModal(true);
    }
  } else {
    _board.shakeSelected();

    if (result.one_away) {
      showToast('One away…', 2000);
    } else {
      showToast('Not quite.', 1500);
    }

    _board.clearSelection();
    const remaining = _lives.loseOne();

    if (remaining === 0) {
      _gameOver = true;
      _board.disable();
      await delay(600);
      revealAllGroups();
      await delay(800);
      showEndModal(false);
    }
  }
}

// ── Reveal all groups on loss ─────────────────────────────────────────────────

function revealAllGroups() {
  // Show all unsolved groups in order
  const unsolvedGroups = _puzzle.groups
    .filter(g => !_solvedLevels.includes(g.level))
    .sort((a, b) => a.level - b.level);

  unsolvedGroups.forEach(g => renderResultBar($solvedGroups, g));

  // Hide all remaining tiles
  $boardEl.querySelectorAll('.tile').forEach(t => {
    t.style.transition = 'opacity 0.4s ease';
    t.style.opacity    = '0';
    setTimeout(() => t.style.display = 'none', 400);
  });
}

// ── End modal ─────────────────────────────────────────────────────────────────

function showEndModal(won) {
  $modalEmoji.textContent  = won ? '🎉' : '😮';
  $modalTitle.textContent  = won ? 'Solved!' : 'Better luck next time';
  $modalSub.textContent    = won
    ? `You solved all four groups!`
    : `You ran out of guesses. Better luck next time!`;

  buildShareGrid();
  $modal.classList.remove('hidden');
}

function buildShareGrid() {
  $shareGrid.innerHTML = '';

  // One row per guess
  _guessHistory.forEach(guess => {
    const row = document.createElement('div');
    row.className = 'share-row';
    for (let i = 0; i < 4; i++) {
      const sq = document.createElement('div');
      sq.className = 'share-sq';
      // Color the square by which group that word belongs to
      const word    = guess.words[i];
      const group   = _puzzle.groups.find(g =>
        g.members.map(m => m.toUpperCase()).includes(word.toUpperCase())
      );
      sq.dataset.color = group ? group.color : 'miss';
      row.appendChild(sq);
    }
    $shareGrid.appendChild(row);
  });
}

function buildShareText() {
  const rows = [...$shareGrid.querySelectorAll('.share-row')].map(row => {
    return [...row.querySelectorAll('.share-sq')].map(sq => {
      const map = { yellow: '🟨', green: '🟩', blue: '🟦', purple: '🟪', miss: '⬜' };
      return map[sq.dataset.color] || '⬜';
    }).join('');
  }).join('\n');

  const won   = _solvedLevels.length === 4;
  const score = won
    ? `Solved in ${_guessHistory.length} guess${_guessHistory.length === 1 ? '' : 'es'}!`
    : 'Did not solve.';

  return `Connections Puzzle Generator\n${score}\n\n${rows}`;
}

// ── Toast ─────────────────────────────────────────────────────────────────────

let _toastTimer = null;

function showToast(message, duration = 1800) {
  clearTimeout(_toastTimer);
  $toastArea.innerHTML = '';

  const toast = document.createElement('div');
  toast.className   = 'toast';
  toast.textContent = message;
  $toastArea.appendChild(toast);

  _toastTimer = setTimeout(() => {
    toast.classList.add('toast--fade');
    toast.addEventListener('animationend', () => toast.remove(), { once: true });
  }, duration);
}

// ── Loading ───────────────────────────────────────────────────────────────────

let _msgInterval = null;

function showLoading(show) {
  $loading.classList.toggle('hidden', !show);
  if (!show) clearInterval(_msgInterval);
}

function cycleLoadingMessage() {
  let i = 0;
  $loadingMsg.textContent = LOADING_MESSAGES[0];
  _msgInterval = setInterval(() => {
    i = (i + 1) % LOADING_MESSAGES.length;
    $loadingMsg.textContent = LOADING_MESSAGES[i];
  }, 900);
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── Event listeners ───────────────────────────────────────────────────────────

$btnShuffle.addEventListener('click', () => _board?.shuffle());

$btnDeselect.addEventListener('click', () => {
  _board?.clearSelection();
  updateControls(0);
});

$btnSubmit.addEventListener('click', handleSubmit);

$btnNew.addEventListener('click', async () => {
  if ($btnNew.disabled) return;
  $btnNew.disabled = true;
  $modal.classList.add('hidden');

  showLoading(true);
  cycleLoadingMessage();

  const { puzzle } = await loadPuzzle();
  _puzzle = puzzle;
  initGame();
  showLoading(false);
  $btnNew.disabled = false;
});

$btnShare.addEventListener('click', async () => {
  const text = buildShareText();
  try {
    await navigator.clipboard.writeText(text);
    showToast('Copied to clipboard!', 2000);
  } catch {
    // Fallback: show text in a prompt
    prompt('Copy your result:', text);
  }
});

$btnPlayAgain.addEventListener('click', () => {
  $modal.classList.add('hidden');
  $btnNew.click();
});

// ── Start ─────────────────────────────────────────────────────────────────────
boot();