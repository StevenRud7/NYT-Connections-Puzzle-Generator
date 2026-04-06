/**
 * components/board.js
 * Renders and manages the 4×4 word tile grid.
 * Owns tile selection state and delegates interactions via callbacks.
 */

import { createTile } from './tile.js';

export class Board {
  /**
   * @param {HTMLElement} container  — the .board element
   * @param {object}      puzzle     — full puzzle data object
   * @param {object}      callbacks
   *   onSelectionChange(selected: string[])
   */
  constructor(container, puzzle, callbacks) {
    this._container = container;
    this._puzzle    = puzzle;
    this._callbacks = callbacks;
    this._selected  = new Set();      // currently selected word strings
    this._solved    = new Set();      // words that belong to solved groups
    this._tiles     = new Map();      // word → tile DOM element
    this._disabled  = false;

    this._render();
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  _render() {
    this._container.innerHTML = '';
    this._tiles.clear();

    this._puzzle.words_shuffled.forEach((word, i) => {
      const tile = createTile(word, () => this._onTileClick(word));
      // Staggered entrance animation
      tile.style.animationDelay = `${i * 35}ms`;
      tile.classList.add('tile--enter');
      this._container.appendChild(tile);
      this._tiles.set(word, tile);
    });
  }

  // ── Interaction ─────────────────────────────────────────────────────────────

  _onTileClick(word) {
    if (this._disabled || this._solved.has(word)) return;

    if (this._selected.has(word)) {
      this._selected.delete(word);
      this._setTileSelected(word, false);
    } else {
      if (this._selected.size >= 4) return;   // max 4 selections
      this._selected.add(word);
      this._setTileSelected(word, true);
    }

    this._callbacks.onSelectionChange([...this._selected]);
  }

  _setTileSelected(word, isSelected) {
    const tile = this._tiles.get(word);
    if (!tile) return;
    tile.classList.toggle('tile--selected', isSelected);
    if (isSelected) {
      tile.classList.remove('tile--bounce');
      void tile.offsetWidth;                  // reflow to restart animation
      tile.classList.add('tile--bounce');
    }
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /** Deselect all tiles */
  clearSelection() {
    this._selected.forEach(w => this._setTileSelected(w, false));
    this._selected.clear();
    this._callbacks.onSelectionChange([]);
  }

  /** Shake all currently-selected tiles (wrong guess feedback) */
  shakeSelected() {
    const words = [...this._selected];
    words.forEach(w => {
      const tile = this._tiles.get(w);
      if (!tile) return;
      tile.classList.remove('tile--shake');
      void tile.offsetWidth;
      tile.classList.add('tile--shake');
      tile.addEventListener('animationend', () => {
        tile.classList.remove('tile--shake');
      }, { once: true });
    });
  }

  /**
   * Mark a group of words as solved.
   * Triggers flip animation then hides tiles (the solved bar takes their place).
   * Returns a Promise that resolves when animations complete.
   */
  async markSolved(words) {
    this._disabled = true;

    // Flip tiles one by one with a slight stagger
    await Promise.all(words.map((w, i) => new Promise(resolve => {
      const tile = this._tiles.get(w);
      if (!tile) { resolve(); return; }
      setTimeout(() => {
        tile.classList.add('tile--flip', 'tile--solved');
        tile.addEventListener('animationend', resolve, { once: true });
      }, i * 80);
    })));

    // Hide flipped tiles
    words.forEach(w => {
      const tile = this._tiles.get(w);
      if (tile) tile.style.display = 'none';
    });
    words.forEach(w => this._solved.add(w));

    // Remove solved words from selection
    words.forEach(w => this._selected.delete(w));
    this._callbacks.onSelectionChange([...this._selected]);

    this._disabled = false;
  }

  /** Shuffle the remaining (unsolved) tiles in place */
  shuffle() {
    if (this._disabled) return;

    const remaining = [...this._tiles.entries()]
      .filter(([w]) => !this._solved.has(w))
      .map(([, el]) => el);

    // Fisher-Yates shuffle of positions
    for (let i = remaining.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [remaining[i], remaining[j]] = [remaining[j], remaining[i]];
    }

    // Re-append in new order (grid order = DOM order)
    remaining.forEach(tile => {
      tile.classList.remove('tile--bounce');
      void tile.offsetWidth;
      this._container.appendChild(tile);
    });
  }

  /** Disable all tile interaction */
  disable() { this._disabled = true; }

  /** Enable all tile interaction */
  enable()  { this._disabled = false; }

  get selectedWords() { return [...this._selected]; }
  get selectionCount() { return this._selected.size; }
}