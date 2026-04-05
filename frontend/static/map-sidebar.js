"use strict";

function renderScoreList(scores) {
  const results = document.getElementById("score-results-list");
  if (!results) return;

  results.replaceChildren();

  if (scores.length === 0) {
    const item = document.createElement("li");
    item.className = "score-results__empty";
    item.textContent = "No scored locations available yet.";
    results.append(item);
    return;
  }

  const groups = new Map(CONTINENT_ORDER.map(continent => [continent, []]));
  for (const point of scores) {
    groups.get(point.continent)?.push(point);
  }

  for (const [continent, cities] of groups) {
    if (cities.length === 0) continue;
    const visibleCount = visibleCountForContinent(continent);
    const visibleCities = cities.slice(0, visibleCount);

    const header = document.createElement("li");
    header.className = "score-results__continent";
    header.textContent = continent;
    results.append(header);

    for (const point of visibleCities) {
      results.append(renderScoreItem(point));
    }

    if (visibleCount < cities.length) {
      const moreItem = document.createElement("li");
      const moreButton = document.createElement("button");
      moreItem.className = "score-results__more";
      moreButton.className = "score-results__more-button";
      moreButton.type = "button";
      moreButton.textContent = "show more";
      moreButton.addEventListener("click", () => {
        continentVisibleCounts.set(continent, nextVisibleCount(visibleCount));
        renderScoreList(currentScores);
        if (mapLoaded) applyMarkers(visibleMarkers());
      });
      moreItem.append(moreButton);
      results.append(moreItem);
    }
  }
}

function renderScoreItem(point) {
  const item = document.createElement("li");
  const score = document.createElement("span");
  const name = document.createElement("span");
  const flag = document.createElement("span");

  item.className = "score-results__item";
  item.tabIndex = 0;
  score.className = "score-results__score";
  name.className = "score-results__name";
  flag.className = "score-results__flag";

  score.textContent = `${Math.round(point.score * 100)}%`.padStart(4, " ");
  name.textContent = point.name;
  flag.textContent = point.flag;
  flag.title = countryNames?.of(point.country_code) ?? point.country_code;

  item.append(score, name, flag);
  item.addEventListener("click", () => focusCityFromList(point));
  item.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    focusCityFromList(point);
  });

  return item;
}
