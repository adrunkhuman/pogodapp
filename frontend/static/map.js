"use strict";

const PROTOMAPS_PM_TILES_URL = "pmtiles://https://build.protomaps.com/20260330.pmtiles";
const PROTOMAPS_ATTRIBUTION =
  '<a href="https://protomaps.com">Protomaps</a> © <a href="https://openstreetmap.org/copyright">OpenStreetMap</a>';
const SCORE_SOURCE_ID = "scores";
const SCORE_LAYER_ID = "scores-circles";
const EMPTY_SCORE_COLLECTION = { type: "FeatureCollection", features: [] };

let map;
let pendingScores = EMPTY_SCORE_COLLECTION;
let pmtilesProtocol;

function setMapStatus(message) {
  const status = document.getElementById("map-status");

  if (status) {
    status.textContent = message;
  }
}

function renderScoreList(collection) {
  const results = document.getElementById("score-results-list");

  if (!results) {
    return;
  }

  results.replaceChildren();

  if (collection.features.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No scored locations available yet.";
    results.append(item);
    return;
  }

  for (const feature of collection.features) {
    const item = document.createElement("li");
    const [lon, lat] = feature.geometry.coordinates;
    const score = feature.properties.score;

    item.textContent = `${Math.round(score * 100)}% match at ${lat}, ${lon}`;
    results.append(item);
  }
}

function toScoreFeatureCollection(scores) {
  if (!Array.isArray(scores)) {
    return EMPTY_SCORE_COLLECTION;
  }

  const features = [];

  for (const point of scores) {
    if (
      typeof point?.lat !== "number" ||
      typeof point?.lon !== "number" ||
      typeof point?.score !== "number"
    ) {
      continue;
    }

    features.push({
      type: "Feature",
      geometry: {
        type: "Point",
        coordinates: [point.lon, point.lat],
      },
      properties: {
        score: point.score,
        label: `${Math.round(point.score * 100)}%`,
      },
    });
  }

  return { type: "FeatureCollection", features };
}

function applyScores(collection) {
  pendingScores = collection;
  renderScoreList(collection);

  if (collection.features.length === 0) {
    setMapStatus("No scored locations available.");
  } else {
    setMapStatus(`${collection.features.length} scored locations rendered on the map.`);
  }

  if (!map || !map.isStyleLoaded()) {
    return;
  }

  const source = map.getSource(SCORE_SOURCE_ID);

  if (source) {
    source.setData(collection);
  }
}

function initializeMap() {
  const mapRoot = document.getElementById("map");

  if (!mapRoot || map) {
    return;
  }

  if (!window.maplibregl || !window.pmtiles || !window.basemaps) {
    mapRoot.textContent = "Map libraries failed to load.";
    setMapStatus("Map libraries failed to load.");
    return;
  }

  if (!pmtilesProtocol) {
    pmtilesProtocol = new window.pmtiles.Protocol();
    window.maplibregl.addProtocol("pmtiles", pmtilesProtocol.tile);
  }

  map = new window.maplibregl.Map({
    container: mapRoot,
    style: {
      version: 8,
      glyphs: "https://protomaps.github.io/basemaps-assets/fonts/{fontstack}/{range}.pbf",
      sprite: "https://protomaps.github.io/basemaps-assets/sprites/v4/light",
      sources: {
        protomaps: {
          type: "vector",
          url: PROTOMAPS_PM_TILES_URL,
          attribution: PROTOMAPS_ATTRIBUTION,
        },
      },
      layers: window.basemaps.layers("protomaps", window.basemaps.namedFlavor("light"), {
        lang: "en",
      }),
    },
    center: [12, 20],
    zoom: 1.4,
  });

  map.addControl(new window.maplibregl.NavigationControl({ showCompass: false }), "top-right");

  map.on("load", () => {
    map.addSource(SCORE_SOURCE_ID, {
      type: "geojson",
      data: pendingScores,
    });

    map.addLayer({
      id: SCORE_LAYER_ID,
      type: "circle",
      source: SCORE_SOURCE_ID,
      paint: {
        "circle-color": [
          "interpolate",
          ["linear"],
          ["get", "score"],
          0,
          "#355c7d",
          0.5,
          "#f8b65a",
          1,
          "#ea5f89",
        ],
        "circle-opacity": 0.82,
        "circle-radius": [
          "interpolate",
          ["linear"],
          ["get", "score"],
          0,
          4,
          1,
          13,
        ],
        "circle-stroke-color": "#fffaf3",
        "circle-stroke-width": 1.5,
      },
    });

    applyScores(pendingScores);
  });
}

window.renderScores = function renderScores(scores) {
  applyScores(toScoreFeatureCollection(scores));
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeMap, { once: true });
} else {
  initializeMap();
}

renderScoreList(EMPTY_SCORE_COLLECTION);
