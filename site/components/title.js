/**
 * components/tile.js
 * Creates individual tile DOM elements.
 */

/**
 * Create a single tile button element.
 * @param {string}   word       — display text
 * @param {Function} onClick    — click handler
 * @returns {HTMLButtonElement}
 */
export function createTile(word, onClick) {
  const btn = document.createElement('button');
  btn.className    = 'tile';
  btn.textContent  = word;
  btn.dataset.word = word;
  btn.setAttribute('aria-label', word);
  btn.addEventListener('click', onClick);
  return btn;
}