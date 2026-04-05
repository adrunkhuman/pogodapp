"use strict";

const WORLD_BACKDROP_URL = "/static/data/world.geojson";
const BACKDROP_SOURCE_ID = "world-backdrop";
const OCEAN_LAYER_ID = "world-ocean";
const LAND_LAYER_ID = "world-land";
const BORDER_LAYER_ID = "world-borders";
const SCORE_SOURCE_ID = "scores";
const SCORE_HEATMAP_LAYER_ID = "scores-heatmap";
const EMPTY_FEATURE_COLLECTION = { type: "FeatureCollection", features: [] };

let map;
let pendingScores = EMPTY_FEATURE_COLLECTION;

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

  const top = collection.features
    .slice()
    .sort((a, b) => b.properties.score - a.properties.score)
    .slice(0, 20);

  for (const feature of top) {
    const item = document.createElement("li");
    const [lon, lat] = feature.geometry.coordinates;
    const score = feature.properties.score;

    item.textContent = `${Math.round(score * 100)}% match at ${lat}, ${lon}`;
    results.append(item);
  }
}

function toScoreCollection(scores) {
  if (!Array.isArray(scores)) {
    return EMPTY_FEATURE_COLLECTION;
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

  if (!window.maplibregl) {
    mapRoot.textContent = "Map library failed to load.";
    setMapStatus("Map library failed to load.");
    return;
  }

  map = new window.maplibregl.Map({
    container: mapRoot,
    style: {
      version: 8,
      sources: {},
      layers: [
        {
          id: OCEAN_LAYER_ID,
          type: "background",
          paint: {
            "background-color": "#dfeaf0",
          },
        },
      ],
    },
    center: [12, 20],
    zoom: 1.4,
  });

  map.addControl(new window.maplibregl.NavigationControl({ showCompass: false }), "top-right");
  map.on("dataabort", (event) => {
    if (event.sourceId === BACKDROP_SOURCE_ID) {
      setMapStatus("Map backdrop failed to load.");
    }
  });
  map.on("error", (event) => {
    if (event.sourceId === BACKDROP_SOURCE_ID) {
      setMapStatus("Map backdrop failed to load.");
    }
  });

  map.on("load", () => {
    map.addSource(BACKDROP_SOURCE_ID, {
      type: "geojson",
      data: WORLD_BACKDROP_URL,
    });

    map.addLayer({
      id: LAND_LAYER_ID,
      type: "fill",
      source: BACKDROP_SOURCE_ID,
      paint: {
        "fill-color": "#f7f0df",
        "fill-opacity": 0.92,
      },
    });

    map.addLayer({
      id: BORDER_LAYER_ID,
      type: "line",
      source: BACKDROP_SOURCE_ID,
      paint: {
        "line-color": "rgba(70, 96, 109, 0.38)",
        "line-width": 0.6,
      },
    });

    map.addSource(SCORE_SOURCE_ID, {
      type: "geojson",
      data: pendingScores,
    });

    map.addLayer({
      id: SCORE_HEATMAP_LAYER_ID,
      type: "heatmap",
      source: SCORE_SOURCE_ID,
      paint: {
        "heatmap-weight": [
          "interpolate",
          ["linear"],
          ["get", "score"],
          0, 0,
          1, 1,
        ],
        "heatmap-intensity": [
          "interpolate",
          ["linear"],
          ["zoom"],
          0, 0.6,
          3, 1.0,
          6, 1.4,
        ],
        "heatmap-radius": [
          "interpolate",
          ["linear"],
          ["zoom"],
          0, 6,
          2, 10,
          4, 16,
          6, 24,
        ],
        "heatmap-color": [
          "interpolate",
          ["linear"],
          ["heatmap-density"],
          0,    "rgba(53, 92, 125, 0)",
          0.2,  "rgba(53, 92, 125, 0.35)",
          0.45, "rgba(127, 179, 213, 0.55)",
          0.65, "rgba(248, 182, 90, 0.78)",
          0.82, "rgba(234, 95, 137, 0.88)",
          1,    "rgba(121, 40, 202, 0.92)",
        ],
        "heatmap-opacity": [
          "interpolate",
          ["linear"],
          ["zoom"],
          0, 0.85,
          4, 0.78,
          6, 0.70,
        ],
      },
    });

    setMapStatus("Map backdrop ready.");
    applyScores(pendingScores);
  });
}

window.renderScores = function renderScores(scores) {
  applyScores(toScoreCollection(scores));
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeMap, { once: true });
} else {
  initializeMap();
}

renderScoreList(EMPTY_FEATURE_COLLECTION);
