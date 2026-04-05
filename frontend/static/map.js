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

// 1×1 transparent PNG — used to clear the heatmap when a response has no results.
const EMPTY_IMAGE =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

// Score-band coloring used in the probe tooltip and marker strokes.
// Matches the heatmap palette tiers: dim < 50%, warm 50-75%, bright ≥ 75%.
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

// Color stops matching the server-side heatmap palette (heatmap.py _COLOR_STOPS).
// Used for scored city circle stroke colors so markers read consistently with the heatmap.
const SCORE_COLOR_EXPR = [
  "interpolate", ["linear"], ["get", "score"],
  0.00, "#355c7d",
  0.55, "#5d8cff",
  0.75, "#ff9f43",
  0.88, "#ff5f87",
  0.95, "#d946ef",
  1.00, "#7928ca",
];

// Continent → country code sets for sidebar grouping (mirrors backend/cities.py).
const _EUROPE = new Set("AD AL AT BA BE BG BY CH CY CZ DE DK EE ES FI FR GB GI GR HR HU IE IS IT LI LT LU LV MC MD ME MK MT NL NO PL PT RO RS RU SE SI SK SM UA VA XK".split(" "));
const _ASIA   = new Set("AE AF AM AZ BD BH BN BT CN GE ID IL IN IQ IR JO JP KG KH KP KR KW KZ LA LB LK MM MN MO MV MY NP OM PH PK PS QA SA SG SY TH TJ TL TM TR TW UZ VN YE".split(" "));
const _AFRICA = new Set("AO BF BI BJ BW CD CF CG CI CM CV DJ DZ EG EH ER ET GA GH GM GN GQ GW KE KM LR LS LY MA MG ML MR MU MW MZ NA NE NG RW SC SD SL SN SO SS ST SZ TD TG TN TZ UG ZA ZM ZW".split(" "));
const _NORTH_AMERICA  = new Set("AG BB BS BZ CA CR CU DM DO GD GT HN HT JM KN LC MX NI PA TT US VC".split(" "));
const _SOUTH_AMERICA  = new Set("AR BO BR CL CO EC GF GY PE PY SR UY VE".split(" "));
const _OCEANIA = new Set("AU FJ FM KI MH NR NZ PG PW SB TO TV VU WS".split(" "));
const EUROPE_ASIA_LONGITUDE_SPLIT = 60;

const CONTINENT_ORDER = ["Europe", "Asia", "Africa", "North America", "South America", "Oceania"];

function continentOf(code, lon = null) {
  if (code === "RU" && lon != null) return lon >= EUROPE_ASIA_LONGITUDE_SPLIT ? "Asia" : "Europe";
  if (_EUROPE.has(code))        return "Europe";
  if (_ASIA.has(code))          return "Asia";
  if (_AFRICA.has(code))        return "Africa";
  if (_NORTH_AMERICA.has(code)) return "North America";
  if (_SOUTH_AMERICA.has(code)) return "South America";
  if (_OCEANIA.has(code))       return "Oceania";
  return "Other";
}

let map;
const countryNames = typeof Intl.DisplayNames === "function"
  ? new Intl.DisplayNames(["en"], { type: "region" })
  : null;
// Set to true once map.on("load") has fired and all layers are registered.
let mapLoaded = false;
// Holds a response that arrived before mapLoaded was set.
let pendingResponse = null;
// Probe debounce + abort controller.
let probeTimer = null;
let probeController = null;
let tooltipHideTimer = null;
let focusClearTimer = null;
let focusAnimationFrame = null;
let focusRequestToken = 0;
// True while cursor is inside a city marker or landmark layer — suppresses the
// general mousemove probe so the layer-specific handler stays in control.
let hoveringLayer = false;
let currentScores = [];
const continentVisibleCounts = new Map();

const tooltip = document.getElementById("map-probe-tooltip");

function setMapStatus(message) {
  const status = document.getElementById("map-status");
  if (status) status.textContent = message;
}

// ─── Utilities ──────────────────────────────────────────────────────────────

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Regional indicator letters: A=🇦, Z=🇿 — pair two letters to get a flag emoji.
function countryFlag(code) {
  return [...code.toUpperCase()].map(c => String.fromCodePoint(c.charCodeAt(0) + 127397)).join("");
}

function scoreColor(v) {
  if (v >= 0.75) return SCORE_HIGH_COLOR;
  if (v >= 0.5)  return SCORE_MID_COLOR;
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
    const continent = continentOf(point.country_code, point.lon);
    const usedCount = groupedCounts.get(continent) ?? 0;
    if (usedCount >= visibleCountForContinent(continent)) {
      continue;
    }
    groupedCounts.set(continent, usedCount + 1);
    visibleScores.push(point);
  }

  return visibleScores;
}

function cancelTooltipHideTimer() {
  clearTimeout(tooltipHideTimer);
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
    fetchProbe(point.probe_lat ?? point.lat, point.probe_lon ?? point.lon, mapBox.left + pixel.x, mapBox.top + pixel.y, `${point.flag} ${point.name}`, {
      hideDelayMs,
    });
  };

  if (moveIfNeeded) {
    ensurePointVisible(point, finalizeFocus);
    return;
  }

  finalizeFocus();
}

// ─── Sidebar ────────────────────────────────────────────────────────────────

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

  // Group by continent, preserving score order within each group.
  const groups = new Map(CONTINENT_ORDER.map(c => [c, []]));
  for (const point of scores) {
    const continent = continentOf(point.country_code, point.lon);
    groups.get(continent)?.push(point);
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
      const item  = document.createElement("li");
      const score = document.createElement("span");
      const name  = document.createElement("span");
      const flag  = document.createElement("span");

      item.className  = "score-results__item";
      item.tabIndex = 0;
      score.className = "score-results__score";
      name.className  = "score-results__name";
      flag.className  = "score-results__flag";

      score.textContent = `${Math.round(point.score * 100)}%`.padStart(4, " ");
      name.textContent  = point.name;
      flag.textContent  = point.flag;
      flag.title = countryNames?.of(point.country_code) ?? point.country_code;

      item.append(score, name, flag);
      item.addEventListener("click", () => focusCity(point, {
        withTooltip: true,
        hideDelayMs: TOOLTIP_FOCUS_HIDE_DELAY_MS,
        moveIfNeeded: true,
      }));
      item.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        focusCity(point, {
          withTooltip: true,
          hideDelayMs: TOOLTIP_FOCUS_HIDE_DELAY_MS,
          moveIfNeeded: true,
        });
      });
      results.append(item);
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
        if (mapLoaded) {
          applyMarkers(visibleScoresForList(currentScores));
        }
      });
      moreItem.append(moreButton);
      results.append(moreItem);
    }
  }
}

// ─── Heatmap ─────────────────────────────────────────────────────────────────

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

    // Insert before the ocean mask so the mask clips bilinear bleed at coastlines.
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
      OCEAN_MASK_LAYER_ID,
    );
  }
}

// ─── City markers ────────────────────────────────────────────────────────────

function markersToGeoJSON(markers) {
  return {
    type: "FeatureCollection",
    features: markers.map(m => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [m.lon, m.lat] },
      properties: {
        name: m.name,
        score: m.score,
        flag: m.flag,
        country_code: m.country_code,
        probe_lat: m.probe_lat,
        probe_lon: m.probe_lon,
      },
    })),
  };
}

function applyMarkers(markers) {
  const geojson = markersToGeoJSON(markers);
  const source = map.getSource(MARKER_SOURCE_ID);

  if (source) {
    source.setData(geojson);
    return;
  }

  map.addSource(MARKER_SOURCE_ID, { type: "geojson", data: geojson });

  // Score-colored fill + white stroke — color conveys rank, white outline pops on any heatmap.
  map.addLayer({
    id: MARKER_LAYER_ID,
    type: "circle",
    source: MARKER_SOURCE_ID,
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 1, 5, 6, 7, 10, 10],
      "circle-color": SCORE_COLOR_EXPR,
      "circle-opacity": 1.0,
      "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 1, 1.5, 6, 2, 10, 2.5],
      "circle-stroke-color": "#ffffff",
      "circle-stroke-opacity": 0.9,
      "circle-blur": 0.05,
    },
  });

  // Unified probe tooltip — no separate MapLibre popup.
  map.on("mouseenter", MARKER_LAYER_ID, (e) => {
    hoveringLayer = true;
    map.getCanvas().style.cursor = "pointer";
    clearTimeout(probeTimer);
    if (probeController) probeController.abort();
    const props = e.features[0].properties;
    fetchProbe(
      Number(props.probe_lat ?? e.lngLat.lat), Number(props.probe_lon ?? e.lngLat.lng),
      e.originalEvent.clientX, e.originalEvent.clientY,
      `${props.flag} ${props.name}`,
      { hideDelayMs: null },
    );
  });

  map.on("mouseleave", MARKER_LAYER_ID, () => {
    hoveringLayer = false;
    map.getCanvas().style.cursor = "";
    hideTooltip();
  });
}

function clearMarkers() {
  const source = map.getSource(MARKER_SOURCE_ID);
  if (source) source.setData({ type: "FeatureCollection", features: [] });
}

// ─── Landmark cities ─────────────────────────────────────────────────────────

function loadLandmarkCities() {
  fetch(LANDMARK_CITIES_URL)
    .then(r => r.json())
    .then(cities => {
      const geojson = {
        type: "FeatureCollection",
        features: cities.map(c => ({
          type: "Feature",
          geometry: { type: "Point", coordinates: [c.lon, c.lat] },
          properties: { name: c.name, country_code: c.country_code ?? null },
        })),
      };

      map.addSource(LANDMARK_SOURCE_ID, { type: "geojson", data: geojson });

      // Brighter neutral dots — always visible for geographic orientation.
      map.addLayer(
        {
          id: LANDMARK_LAYER_ID,
          type: "circle",
          source: LANDMARK_SOURCE_ID,
          paint: {
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 1, 2.75, 6, 4.5, 10, 6],
            "circle-color": "#c9c9c9",
            "circle-opacity": 0.92,
            "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 1, 0.8, 6, 1.2, 10, 1.6],
            "circle-stroke-color": "#141414",
            "circle-stroke-opacity": 0.95,
          },
        },
      );

      map.on("mouseenter", LANDMARK_LAYER_ID, (e) => {
        hoveringLayer = true;
        map.getCanvas().style.cursor = "default";
        clearTimeout(probeTimer);
        if (probeController) probeController.abort();
        const lp = e.features[0].properties;
        const landmarkHeader = lp.country_code
          ? `${countryFlag(lp.country_code)} ${lp.name}`
          : lp.name;
        fetchProbe(e.lngLat.lat, e.lngLat.lng, e.originalEvent.clientX, e.originalEvent.clientY, landmarkHeader, {
          hideDelayMs: null,
        });
      });

      map.on("mouseleave", LANDMARK_LAYER_ID, () => {
        hoveringLayer = false;
        hideTooltip();
      });
    })
    .catch(() => {
      // Landmark layer is non-critical — ignore failures silently.
    });
}

// ─── Probe tooltip ───────────────────────────────────────────────────────────

function getCurrentPreferences() {
  const form = document.getElementById("preferences");
  if (!form) return null;
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}

function positionTooltip(x, y) {
  if (!tooltip || !map) return;

  const mapBox = map.getContainer().getBoundingClientRect();
  const tooltipBox = tooltip.getBoundingClientRect();
  const left = Math.min(
    Math.max(x + TOOLTIP_PADDING_PX, mapBox.left + TOOLTIP_PADDING_PX),
    mapBox.right - tooltipBox.width - TOOLTIP_PADDING_PX,
  );
  const top = Math.min(
    Math.max(y - tooltipBox.height / 2, mapBox.top + TOOLTIP_PADDING_PX),
    mapBox.bottom - tooltipBox.height - TOOLTIP_PADDING_PX,
  );

  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function nearestFeatureAtPoint(point) {
  if (!mapLoaded) return null;

  const bounds = [
    [point.x - CITY_SNAP_RADIUS_PX, point.y - CITY_SNAP_RADIUS_PX],
    [point.x + CITY_SNAP_RADIUS_PX, point.y + CITY_SNAP_RADIUS_PX],
  ];
  const candidates = map.queryRenderedFeatures(bounds, { layers: [MARKER_LAYER_ID, LANDMARK_LAYER_ID] });
  if (candidates.length === 0) return null;

  let nearest = null;
  let nearestDistance = Infinity;
  for (const feature of candidates) {
    if (!feature.geometry || feature.geometry.type !== "Point") continue;
    const [lon, lat] = feature.geometry.coordinates;
    const featurePoint = map.project([lon, lat]);
    const dx = featurePoint.x - point.x;
    const dy = featurePoint.y - point.y;
    const distance = Math.hypot(dx, dy);
    if (distance > CITY_SNAP_RADIUS_PX || distance >= nearestDistance) continue;
    nearest = {
      lat,
      lon,
      header: feature.layer.id === MARKER_LAYER_ID
        ? `${feature.properties.flag} ${feature.properties.name}`
        : (feature.properties.country_code
          ? `${countryFlag(feature.properties.country_code)} ${feature.properties.name}`
          : feature.properties.name),
      cursor: feature.layer.id === MARKER_LAYER_ID ? "pointer" : "default",
    };
    nearestDistance = distance;
  }

  return nearest;
}

function showTooltip(data, x, y, cityHeader = null, { hideDelayMs = null } = {}) {
  if (!tooltip || !data.found) {
    if (tooltip) tooltip.hidden = true;
    return;
  }

  const overall = data.overall_score ?? ((data.temp_score + data.rain_score + data.cloud_score) / 3);
  const temp = `${data.avg_temp_c > 0 ? "+" : ""}${data.avg_temp_c.toFixed(1)}°C`.padStart(8);
  const rain = `${Math.round(data.avg_precip_mm)}mm/mo`.padStart(8);
  const sun  = `${Math.round(100 - data.avg_cloud_pct)}% sun`.padStart(8);
  const line = (label, value, score) => (
    `<div class="probe-tooltip__row">` +
    `<span class="probe-tooltip__label">${label}</span>` +
    `<span class="probe-tooltip__value">${value}</span>` +
    `<span class="probe-tooltip__metric">${scoreSpan(score)}</span>` +
    `</div>`
  );
  const cityName = cityHeader ? cityHeader.replace(/^\S+\s+/, "") : "";
  const cityFlag = cityHeader ? cityHeader.match(/^\S+/)?.[0] ?? "" : "";

  // Score always left so it doesn't jump when entering/leaving a city marker.
  const header = cityHeader
    ? `<div class="probe-tooltip__header"><span class="probe-tooltip__header-score">${scoreSpan(overall)}</span><span class="probe-tooltip__header-city"><span class="probe-tooltip__header-name">${escapeHtml(cityName)}</span><span class="probe-tooltip__header-flag">${escapeHtml(cityFlag)}</span></span></div>`
    : `<div class="probe-tooltip__header"><span class="probe-tooltip__header-score">${scoreSpan(overall)}</span></div>`;

  tooltip.innerHTML =
    header +
    line("temp", temp, data.temp_score) +
    line("rain", rain, data.rain_score) +
    line("sun", sun, data.cloud_score);

  tooltip.hidden = false;
  positionTooltip(x, y);
  if (hideDelayMs == null) {
    persistTooltip();
  } else {
    armTooltipHideTimer(hideDelayMs);
  }
}

function hideTooltip() {
  cancelTooltipHideTimer();
  if (tooltip) tooltip.hidden = true;
}

function fetchProbe(lat, lon, clientX, clientY, cityHeader = null, { hideDelayMs = null } = {}) {
  const prefs = getCurrentPreferences();
  if (!prefs) return;

  if (probeController) probeController.abort();
  probeController = new AbortController();

  const params = new URLSearchParams({ lat, lon, ...prefs });

  fetch(`/probe?${params}`, { signal: probeController.signal })
    .then(r => r.json())
    .then(data => showTooltip(data, clientX, clientY, cityHeader, { hideDelayMs }))
    .catch(() => {});
}

// ─── Map init ─────────────────────────────────────────────────────────────────

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
  map.on("mousemove", (e) => {
    if (hoveringLayer) return;
    hideTooltip();
    clearTimeout(probeTimer);
    probeTimer = setTimeout(() => {
      const snapped = nearestFeatureAtPoint(e.point);
      if (snapped) {
        map.getCanvas().style.cursor = snapped.cursor;
        fetchProbe(snapped.lat, snapped.lon, e.originalEvent.clientX, e.originalEvent.clientY, snapped.header, {
          hideDelayMs: null,
        });
        return;
      }

      map.getCanvas().style.cursor = "";
      fetchProbe(e.lngLat.lat, e.lngLat.lng, e.originalEvent.clientX, e.originalEvent.clientY, null, {
        hideDelayMs: null,
      });
    }, 80);
  });

  map.on("movestart", () => {
    clearTimeout(probeTimer);
    if (probeController) probeController.abort();
    hideTooltip();
    clearFocusedCity();
  });

  map.on("mouseleave", () => {
    clearTimeout(probeTimer);
    if (probeController) probeController.abort();
    hideTooltip();
    clearFocusedCity();
  });

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

    if (pendingResponse) {
      const { scores, markers, heatmap } = pendingResponse;
      applyHeatmap(heatmap);
      applyMarkers(markers ?? []);
      setMapStatus(heatmap !== EMPTY_IMAGE ? `${scores.length} top matches shown.` : "No matches found.");
      pendingResponse = null;
    }
  });
}

// ─── Public API ──────────────────────────────────────────────────────────────

window.renderScores = function renderScores(response) {
  const { scores, heatmap } = response;
  continentVisibleCounts.clear();
  currentScores = scores ?? [];

  renderScoreList(currentScores);

  if (!map) return;

  const imageUrl = heatmap || EMPTY_IMAGE;
  const markerList = visibleScoresForList(currentScores);

  if (mapLoaded) {
    applyHeatmap(imageUrl);
    if (markerList.length > 0) {
      applyMarkers(markerList);
    } else {
      clearMarkers();
    }
    setMapStatus(heatmap ? `${scores.length} top matches shown.` : "No matches found.");
  } else {
    pendingResponse = { ...response, heatmap: imageUrl };
  }
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeMap, { once: true });
} else {
  initializeMap();
}

renderScoreList([]);
