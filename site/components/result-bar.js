/**
 * components/result-bar.js
 * Renders a solved group's colour bar above the board.
 */

/**
 * @param {HTMLElement} container  — the .solved-groups element
 * @param {object}      group      — group data { color, theme, members }
 */
export function renderResultBar(container, group) {
  const bar = document.createElement('div');
  bar.className      = 'solved-group';
  bar.dataset.color  = group.color;
  bar.dataset.level  = group.level;
  bar.innerHTML = `
    <div class="solved-group__theme">${group.theme}</div>
    <div class="solved-group__members">${group.members.join(', ')}</div>
  `;
  container.appendChild(bar);
}