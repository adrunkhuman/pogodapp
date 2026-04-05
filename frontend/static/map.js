"use strict";

const WORLD_BACKDROP_URL = "/static/data/world.geojson";
const BACKDROP_SOURCE_ID = "world-backdrop";
const OCEAN_LAYER_ID = "world-ocean";
const LAND_LAYER_ID = "world-land";
const BORDER_LAYER_ID = "world-borders";
const HEATMAP_SOURCE_ID = "score-heatmap";
const HEATMAP_LAYER_ID = "score-heatmap";

// 1x1 transparent PNG — placeholder until the first score response arrives
const EMPTY_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

// World extent corners for the image source: [lon, lat] for TL, TR, BR, BL
const WORLD_CORNERS = [[-180, 90], [180, 90], [180, -90], [-180, -90]];

let map;

function setMapStatus(message) {
  const status = document.getElementById("map-status");

  if (status) {
    status.textContent = message;
  }
}

function renderScoreList(scores) {
  const results = document.getElementById("score-results-list");

  if (!results) {
    return;
  }

  results.replaceChildren();

  if (scores.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No scored locations available yet.";
    results.append(item);
    return;
  }

  for (const point of scores) {
    const item = document.createElement("li");
    item.textContent = `${Math.round(point.score * 100)}% match at ${point.lat}, ${point.lon}`;
    results.append(item);
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

    map.addSource(HEATMAP_SOURCE_ID, {
      type: "image",
      url: EMPTY_IMAGE,
      coordinates: WORLD_CORNERS,
    });

    map.addLayer({
      id: HEATMAP_LAYER_ID,
      type: "raster",
      source: HEATMAP_SOURCE_ID,
      paint: {
        "raster-opacity": 0.85,
        "raster-fade-duration": 200,
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

    setMapStatus("Map backdrop ready.");
  });
}

window.renderScores = function renderScores(response) {
  const { scores, heatmap } = response;

  renderScoreList(scores ?? []);

  if (heatmap && map && map.isStyleLoaded()) {
    map.getSource(HEATMAP_SOURCE_ID)?.updateImage({ url: heatmap });
    setMapStatus(`${scores.length} top matches shown.`);
  }
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeMap, { once: true });
} else {
  initializeMap();
}

renderScoreList([]);
