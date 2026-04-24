"use strict";

function applyScoreResponse(scores, heatmapUrl) {
  applyHeatmap(heatmapUrl || EMPTY_IMAGE);
  const markers = visibleScoresForList(scores);
  if (markers.length > 0) {
    applyMarkers(markers);
  } else {
    clearMarkers();
  }
  setMapStatus(heatmapUrl ? `${scores.length} top matches shown.` : "No matches found.");
}

window.renderHeatmap = function renderHeatmap(heatmapUrl) {
  pendingHeatmapUrl = heatmapUrl;
  if (!map || !mapLoaded) return;
  applyHeatmap(heatmapUrl || EMPTY_IMAGE);
};

function initializeMap() {
  const mapRoot = document.getElementById("map");

  if (!mapRoot || map) return;

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
          paint: { "background-color": "#101010" },
        },
      ],
    },
    center: [14.78, 34.252],
    zoom: 1.35,
  });

  map.addControl(new window.maplibregl.NavigationControl({ showCompass: false }), "top-right");

  map.on("dataabort", (event) => {
    if (event.sourceId === BACKDROP_SOURCE_ID) setMapStatus("Map backdrop failed to load.");
  });
  map.on("error", (event) => {
    if (event.sourceId === BACKDROP_SOURCE_ID) setMapStatus("Map backdrop failed to load.");
  });

  // Probe on hover — debounced; skipped when a layer-specific handler takes over.
  map.on("mousemove", scheduleHoverProbe);

  map.on("movestart", resetTransientMapUi);

  map.on("mouseleave", resetTransientMapUi);

  map.on("load", () => {
    map.addSource(BACKDROP_SOURCE_ID, { type: "geojson", data: WORLD_BACKDROP_URL });
    map.addSource(OCEAN_MASK_SOURCE_ID, { type: "geojson", data: WORLD_OCEAN_URL });

    map.addLayer({
      id: LAND_LAYER_ID,
      type: "fill",
      source: BACKDROP_SOURCE_ID,
      paint: { "fill-color": "#202020", "fill-opacity": 1 },
    });

    // Sits above the heatmap to clip bilinear texture bleed at coastlines.
    map.addLayer({
      id: OCEAN_MASK_LAYER_ID,
      type: "fill",
      source: OCEAN_MASK_SOURCE_ID,
      paint: { "fill-color": "#101010", "fill-opacity": 1 },
    });

    map.addLayer({
      id: BORDER_LAYER_ID,
      type: "line",
      source: BACKDROP_SOURCE_ID,
      paint: { "line-color": "#333333", "line-width": 0.6 },
    });

    loadLandmarkCities();

    mapLoaded = true;
    setMapStatus("Map backdrop ready.");

    if (pendingHeatmapUrl !== null) {
      applyHeatmap(pendingHeatmapUrl || EMPTY_IMAGE);
    }

    if (!pendingResponse) return;

    const { scores, heatmap_url } = pendingResponse;
    applyScoreResponse(scores ?? [], heatmap_url);
    pendingResponse = null;
  });
}

// Public handoff used by `app.js` after a successful `/score` HTMX response.
// Expects the raw backend payload and lets the map load the heatmap separately.
window.renderScores = function renderScores(response) {
  const { scores, heatmap_url } = response;
  continentVisibleCounts.clear();
  currentScores = scores ?? [];

  renderScoreList(currentScores);

  if (!map) return;

  if (mapLoaded) {
    applyScoreResponse(currentScores, heatmap_url);
  } else {
    pendingResponse = response;
  }
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeMap, { once: true });
} else {
  initializeMap();
}

renderScoreList([]);
