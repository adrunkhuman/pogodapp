"use strict";

function formatScore(score) {
  return `${Math.round(score * 100)}%`;
}

function formatCoordinate(value) {
  return Number(value).toString();
}

window.renderScores = function renderScores(scores) {
  const map = document.getElementById("map");

  if (!map) {
    return;
  }

  if (!Array.isArray(scores) || scores.length === 0) {
    map.innerHTML = "<p>No scored locations available.</p>";
    return;
  }

  const items = scores
    .map(
      ({ lat, lon, score }) => `
        <li>
          <strong>${formatScore(score)}</strong>
          <span>${formatCoordinate(lat)}, ${formatCoordinate(lon)}</span>
        </li>`,
    )
    .join("");

  map.innerHTML = `
    <ol class="map-results">
      ${items}
    </ol>`;
};
