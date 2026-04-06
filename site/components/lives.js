/**
 * components/lives.js
 * Manages the 4-dot "mistakes remaining" indicator.
 */

export class Lives {
  /**
   * @param {HTMLElement} container — the .lives__dots element
   * @param {number}      total     — total lives (default 4)
   */
  constructor(container, total = 4) {
    this._container = container;
    this._total     = total;
    this._remaining = total;
    this._render();
  }

  _render() {
    this._container.innerHTML = '';
    for (let i = 0; i < this._total; i++) {
      const dot = document.createElement('div');
      dot.className = 'lives__dot';
      dot.dataset.index = i;
      this._container.appendChild(dot);
    }
  }

  /** Remove one life and animate the dot. Returns remaining lives. */
  loseOne() {
    if (this._remaining <= 0) return 0;
    this._remaining--;

    // Mark the rightmost active dot as lost
    const dots    = [...this._container.querySelectorAll('.lives__dot')];
    const toMark  = dots.filter(d => !d.classList.contains('lost'));
    const last    = toMark[toMark.length - 1];
    if (last) last.classList.add('lost');

    return this._remaining;
  }

  get remaining() { return this._remaining; }
  get isGameOver() { return this._remaining <= 0; }

  reset(total = this._total) {
    this._total     = total;
    this._remaining = total;
    this._render();
  }
}