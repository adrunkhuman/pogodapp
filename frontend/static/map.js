"use strict";

const WORLD_BACKDROP_URL = "/static/data/world.geojson";
const BACKDROP_SOURCE_ID = "world-backdrop";
const OCEAN_LAYER_ID = "world-ocean";
const LAND_LAYER_ID = "world-land";
const BORDER_LAYER_ID = "world-borders";
const HEATMAP_SOURCE_ID = "score-heatmap";
const HEATMAP_LAYER_ID = "score-heatmap";
const MAP_CONFIG = window.POGODAPP_MAP_CONFIG ?? {
  projection: "mercator",
  imageCorners: [[-180, 85.051129], [180, 85.051129], [180, -85.051129], [-180, -85.051129]],
};

const WORLD_CORNERS = MAP_CONFIG.imageCorners;

// 1×1 transparent PNG — used to clear the heatmap when a response has no results.
const EMPTY_IMAGE =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

let map;
const countryNames = typeof Intl.DisplayNames === "function"
  ? new Intl.DisplayNames(["en"], { type: "region" })
  : null;
// Set to true once map.on("load") has fired and all layers are registered.
// isStyleLoaded() is unreliable after addSource() — it stays false while the
// GeoJSON worker tiles the backdrop, so the load event is the correct signal.
let mapLoaded = false;
// Holds a response that arrived before mapLoaded was set.
let pendingResponse = null;

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
    item.className = "score-results__empty";
    item.textContent = "No scored locations available yet.";
    results.append(item);
    return;
  }

  for (const point of scores) {
    const item = document.createElement("li");
    const score = document.createElement("span");
    const name = document.createElement("span");
    const flag = document.createElement("span");

    item.className = "score-results__item";
    score.className = "score-results__score";
    name.className = "score-results__name";
    flag.className = "score-results__flag";

    score.textContent = `${Math.round(point.score * 100)}%`.padStart(4, " ");
    name.textContent = point.name;
    flag.textContent = point.flag;
    flag.title = countryNames?.of(point.country_code) ?? point.country_code;

    item.append(score, name, flag);
    results.append(item);
  }
}

// Adds the heatmap source+layer on the first call; updates the image URL on subsequent calls.
function applyHeatmap(heatmap) {
  const source = map.getSource(HEATMAP_SOURCE_ID);

  if (source) {
    source.updateImage({ url: heatmap });
  } else {
    map.addSource(HEATMAP_SOURCE_ID, {
      type: "image",
      url: heatmap,
      coordinates: WORLD_CORNERS,
    });

    // Insert before BORDER_LAYER_ID so borders render on top of the heatmap.
    map.addLayer(
      {
        id: HEATMAP_LAYER_ID,
        type: "raster",
        source: HEATMAP_SOURCE_ID,
        paint: {
          "raster-opacity": 0.85,
          "raster-fade-duration": 200,
        },
      },
      BORDER_LAYER_ID,
    );
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
    projection: { type: MAP_CONFIG.projection },
    style: {
      version: 8,
      sources: {},
      layers: [
        {
          id: OCEAN_LAYER_ID,
          type: "background",
          paint: {
            "background-color": "#101010",
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
        "fill-color": "#202020",
        "fill-opacity": 1,
      },
    });

    map.addLayer({
      id: BORDER_LAYER_ID,
      type: "line",
      source: BACKDROP_SOURCE_ID,
      paint: {
        "line-color": "#333333",
        "line-width": 0.6,
      },
    });

    mapLoaded = true;
    setMapStatus("Map backdrop ready.");

    if (pendingResponse) {
      const { scores, heatmap } = pendingResponse;
      applyHeatmap(heatmap);
      setMapStatus(heatmap !== EMPTY_IMAGE ? `${scores.length} top matches shown.` : "No matches found.");
      pendingResponse = null;
    }
  });
}

window.renderScores = function renderScores(response) {
  const { scores, heatmap } = response;

  renderScoreList(scores ?? []);

  if (!map) {
    return;
  }

  // Clear the previous heatmap when results are empty so the layers stay in sync.
  const imageUrl = heatmap || EMPTY_IMAGE;

  if (mapLoaded) {
    applyHeatmap(imageUrl);
    setMapStatus(heatmap ? `${scores.length} top matches shown.` : "No matches found.");
  } else {
    // Response arrived before map.on("load") fired — defer until layers exist.
    pendingResponse = { ...response, heatmap: imageUrl };
  }
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeMap, { once: true });
} else {
  initializeMap();
}

renderScoreList([]);
