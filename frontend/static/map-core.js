"use strict";

const WORLD_BACKDROP_URL = "/static/data/world.geojson";
const WORLD_OCEAN_URL = "/static/data/world_ocean.geojson";
const LANDMARK_CITIES_URL = "/static/data/landmark_cities.json";
const BACKDROP_SOURCE_ID = "world-backdrop";
const OCEAN_MASK_SOURCE_ID = "world-ocean-mask";
const OCEAN_LAYER_ID = "world-ocean";
const LAND_LAYER_ID = "world-land";
const BORDER_LAYER_ID = "world-borders";
const OCEAN_MASK_LAYER_ID = "world-ocean-mask";
const HEATMAP_SOURCE_ID = "score-heatmap";
const HEATMAP_LAYER_ID = "score-heatmap";
const LANDMARK_SOURCE_ID = "landmark-cities";
const LANDMARK_LAYER_ID = "landmark-cities";
const MARKER_SOURCE_ID = "scored-cities";
const MARKER_LAYER_ID = "scored-cities";
const FOCUS_SOURCE_ID = "focused-city";
const FOCUS_RING_LAYER_ID = "focused-city-ring";
const MAP_CONFIG = window.POGODAPP_MAP_CONFIG ?? {
  projection: "mercator",
  imageCorners: [[-180, 85.051129], [180, 85.051129], [180, -85.051129], [-180, -85.051129]],
};

const WORLD_CORNERS = MAP_CONFIG.imageCorners;
const EMPTY_IMAGE =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
const SCORE_LOW_COLOR  = "#555555";
const SCORE_MID_COLOR  = "#ff9f43";
const SCORE_HIGH_COLOR = "#d946ef";
const TOOLTIP_PADDING_PX = 12;
const TOOLTIP_HIDE_DELAY_MS = 1400;
const TOOLTIP_FOCUS_HIDE_DELAY_MS = 2200;
const FOCUS_PING_MS = 1800;
const FOCUS_ANIMATION_MS = 900;
const FOCUS_VISIBILITY_PADDING_PX = 72;
const CITY_SNAP_RADIUS_PX = 14;
const PROBE_HOVER_DELAY_MS = 80;
const PROBE_HOVER_COOLDOWN_MS = 250;
const PROBE_TIMEOUT_MS = 5000;
const SCORE_TIMEOUT_MS = 30000;
const CONTINENT_ORDER = ["Europe", "Asia", "Africa", "North America", "South America", "Oceania"];
const SCORE_COLOR_EXPR = [
  "interpolate", ["linear"], ["get", "score"],
  0.00, "#355c7d",
  0.55, "#5d8cff",
  0.75, "#ff9f43",
  0.88, "#ff5f87",
  0.95, "#d946ef",
  1.00, "#7928ca",
];

let map;
const countryNames = typeof Intl.DisplayNames === "function"
  ? new Intl.DisplayNames(["en"], { type: "region" })
  : null;
let mapLoaded = false;
let pendingResponse = null;
let probeTimer = null;
let probeCooldownTimer = null;
let probeController = null;
let probeTimeoutId = null;
let tooltipHideTimer = null;
let focusClearTimer = null;
let focusAnimationFrame = null;
let focusRequestToken = 0;
let hoveringLayer = false;
let currentScores = [];
const continentVisibleCounts = new Map();

const tooltip = document.getElementById("map-probe-tooltip");

function setMapStatus(message) {
  const status = document.getElementById("map-status");
  if (status) status.textContent = message;
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function countryFlag(code) {
  return [...code.toUpperCase()].map(c => String.fromCodePoint(c.charCodeAt(0) + 127397)).join("");
}

function scoreColor(v) {
  if (v >= 0.75) return SCORE_HIGH_COLOR;
  if (v >= 0.5) return SCORE_MID_COLOR;
  return SCORE_LOW_COLOR;
}

function scoreSpan(v) {
  const pct = `${String(Math.round(v * 100)).padStart(3)}%`;
  return `<span style="color:${scoreColor(v)}">${pct}</span>`;
}

function visibleCountForContinent(continent) {
  return continentVisibleCounts.get(continent) ?? 5;
}

function nextVisibleCount(currentVisibleCount) {
  return Math.ceil((currentVisibleCount + 1) / 5) * 5;
}

function visibleScoresForList(scores) {
  const visibleScores = [];
  const groupedCounts = new Map();

  for (const point of scores) {
    const continent = point.continent;
    const usedCount = groupedCounts.get(continent) ?? 0;
    if (usedCount >= visibleCountForContinent(continent)) continue;
    groupedCounts.set(continent, usedCount + 1);
    visibleScores.push(point);
  }

  return visibleScores;
}

function visibleMarkers() {
  return visibleScoresForList(currentScores);
}

function cancelTooltipHideTimer() {
  clearTimeout(tooltipHideTimer);
}

function abortActiveProbe() {
  clearTimeout(probeTimeoutId);
  if (probeController) probeController.abort();
}

function cancelProbeCooldown() {
  clearTimeout(probeCooldownTimer);
}

function armTooltipHideTimer(delayMs = TOOLTIP_HIDE_DELAY_MS) {
  cancelTooltipHideTimer();
  tooltipHideTimer = setTimeout(() => hideTooltip(), delayMs);
}

function persistTooltip() {
  cancelTooltipHideTimer();
}

function clearFocusedCity() {
  clearTimeout(focusClearTimer);
  cancelAnimationFrame(focusAnimationFrame);
  if (!map) return;
  const source = map.getSource(FOCUS_SOURCE_ID);
  if (!source) return;
  source.setData({ type: "FeatureCollection", features: [] });
}

function focusSourceGeoJSON(point, phase = 0) {
  return {
    type: "FeatureCollection",
    features: point ? [{
      type: "Feature",
      geometry: { type: "Point", coordinates: [point.lon, point.lat] },
      properties: { score: point.score, phase },
    }] : [],
  };
}

function ensureFocusLayers() {
  if (map.getSource(FOCUS_SOURCE_ID)) return;

  map.addSource(FOCUS_SOURCE_ID, { type: "geojson", data: focusSourceGeoJSON(null) });
  map.addLayer({
    id: FOCUS_RING_LAYER_ID,
    type: "circle",
    source: FOCUS_SOURCE_ID,
    paint: {
      "circle-radius": [
        "+",
        ["interpolate", ["linear"], ["zoom"], 1, 6, 6, 8, 10, 10],
        ["*", ["get", "phase"], ["interpolate", ["linear"], ["zoom"], 1, 46, 6, 62, 10, 82]],
      ],
      "circle-color": "rgba(0,0,0,0)",
      "circle-stroke-width": [
        "+",
        ["interpolate", ["linear"], ["zoom"], 1, 1.5, 6, 2.5, 10, 3.5],
        ["*", ["get", "phase"], 2.5],
      ],
      "circle-stroke-color": "#f5f5f5",
      "circle-stroke-opacity": ["+", 0.2, ["*", ["get", "phase"], 0.8]],
    },
  });
}

function isPointVisible(point) {
  if (!map) return true;
  const pixel = map.project([point.lon, point.lat]);
  const width = map.getContainer().clientWidth;
  const height = map.getContainer().clientHeight;
  return (
    pixel.x >= FOCUS_VISIBILITY_PADDING_PX &&
    pixel.y >= FOCUS_VISIBILITY_PADDING_PX &&
    pixel.x <= width - FOCUS_VISIBILITY_PADDING_PX &&
    pixel.y <= height - FOCUS_VISIBILITY_PADDING_PX
  );
}

function ensurePointVisible(point, callback) {
  if (!map || isPointVisible(point)) {
    callback();
    return;
  }

  const token = ++focusRequestToken;
  const onMoveEnd = () => {
    map.off("moveend", onMoveEnd);
    if (token !== focusRequestToken) return;
    callback();
  };

  map.on("moveend", onMoveEnd);
  map.easeTo({
    center: [point.lon, point.lat],
    zoom: Math.max(map.getZoom(), 2.4),
    duration: 450,
    essential: true,
  });
}

function animateFocusedCity(point) {
  ensureFocusLayers();
  cancelAnimationFrame(focusAnimationFrame);
  const source = map.getSource(FOCUS_SOURCE_ID);
  const startedAt = performance.now();

  const tick = (now) => {
    const elapsed = now - startedAt;
    const phase = Math.max(0, 1 - elapsed / FOCUS_ANIMATION_MS);
    source.setData(focusSourceGeoJSON(point, phase));
    if (phase > 0) {
      focusAnimationFrame = requestAnimationFrame(tick);
      return;
    }
    source.setData(focusSourceGeoJSON(point, 0));
    focusAnimationFrame = null;
  };

  focusAnimationFrame = requestAnimationFrame(tick);
}

function focusCity(point, { withTooltip = false, hideDelayMs = TOOLTIP_HIDE_DELAY_MS, moveIfNeeded = false } = {}) {
  if (!map || !mapLoaded || !point) return;
  focusRequestToken += 1;

  const finalizeFocus = () => {
    animateFocusedCity(point);
    clearTimeout(focusClearTimer);
    focusClearTimer = setTimeout(() => clearFocusedCity(), FOCUS_PING_MS);

    if (!withTooltip) return;

    const pixel = map.project([point.lon, point.lat]);
    const mapBox = map.getContainer().getBoundingClientRect();
    fetchProbe(
      point.probe_lat ?? point.lat,
      point.probe_lon ?? point.lon,
      mapBox.left + pixel.x,
      mapBox.top + pixel.y,
      `${point.flag} ${point.name}`,
      { hideDelayMs },
    );
  };

  if (moveIfNeeded) {
    ensurePointVisible(point, finalizeFocus);
    return;
  }

  finalizeFocus();
}

function focusCityFromList(point) {
  focusCity(point, {
    withTooltip: true,
    hideDelayMs: TOOLTIP_FOCUS_HIDE_DELAY_MS,
    moveIfNeeded: true,
  });
}
