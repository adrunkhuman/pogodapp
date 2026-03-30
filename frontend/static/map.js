"use strict";

// Render parsed `/score` payload items and fall back to an empty state for invalid results.
window.renderScores = function renderScores(scores) {
  const map = document.getElementById("map");

  if (!map) {
    return;
  }

  map.replaceChildren();

  if (!Array.isArray(scores) || scores.length === 0) {
    const emptyState = document.createElement("p");
    emptyState.textContent = "No scored locations available.";
    map.append(emptyState);
    return;
  }

  const list = document.createElement("ol");
  list.className = "map-results";

  for (const { lat, lon, score } of scores) {
    const item = document.createElement("li");
    const scoreLabel = document.createElement("strong");
    const coordinateLabel = document.createElement("span");

    scoreLabel.textContent = `${Math.round(score * 100)}%`;
    coordinateLabel.textContent = `${lat}, ${lon}`;
    item.append(scoreLabel, coordinateLabel);
    list.append(item);
  }

  map.append(list);
};
