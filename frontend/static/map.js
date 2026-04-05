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
const MAP_CONFIG = window.POGODAPP_MAP_CONFIG ?? {
  projection: "mercator",
  imageCorners: [[-180, 85.051129], [180, 85.051129], [180, -85.051129], [-180, -85.051129]],
};

const WORLD_CORNERS = MAP_CONFIG.imageCorners;

// 1×1 transparent PNG — used to clear the heatmap when a response has no results.
const EMPTY_IMAGE =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

// Color stops matching the server-side heatmap palette (heatmap.py _COLOR_STOPS).
// Used for scored city circle colors so markers read consistently with the heatmap.
const SCORE_COLOR_EXPR = [
  "interpolate", ["linear"], ["get", "score"],
  0.00, "#355c7d",
  0.45, "#7fb3d5",
  0.65, "#f8b65a",
  0.82, "#ea5f89",
  1.00, "#7928ca",
];

// Continent → country code sets for sidebar grouping (mirrors backend/cities.py).
const _EUROPE = new Set("AD AL AT BA BE BG BY CH CY CZ DE DK EE ES FI FR GB GI GR HR HU IE IS IT LI LT LU LV MC MD ME MK MT NL NO PL PT RO RS RU SE SI SK SM UA VA XK".split(" "));
const _ASIA   = new Set("AE AF AM AZ BD BH BN BT CN GE ID IL IN IQ IR JO JP KG KH KP KR KW KZ LA LB LK MM MN MO MV MY NP OM PH PK PS QA SA SG SY TH TJ TL TM TR TW UZ VN YE".split(" "));
const _AFRICA = new Set("AO BF BI BJ BW CD CF CG CI CM CV DJ DZ EG EH ER ET GA GH GM GN GQ GW KE KM LR LS LY MA MG ML MR MU MW MZ NA NE NG RW SC SD SL SN SO SS ST SZ TD TG TN TZ UG ZA ZM ZW".split(" "));
const _NORTH_AMERICA  = new Set("AG BB BS BZ CA CR CU DM DO GD GT HN HT JM KN LC MX NI PA TT US VC".split(" "));
const _SOUTH_AMERICA  = new Set("AR BO BR CL CO EC GF GY PE PY SR UY VE".split(" "));
const _OCEANIA = new Set("AU FJ FM KI MH NR NZ PG PW SB TO TV VU WS".split(" "));

const CONTINENT_ORDER = ["Europe", "Asia", "Africa", "North America", "South America", "Oceania", "Other"];

function continentOf(code) {
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

const tooltip = document.getElementById("map-probe-tooltip");

function setMapStatus(message) {
  const status = document.getElementById("map-status");
  if (status) status.textContent = message;
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
    const continent = continentOf(point.country_code);
    groups.get(continent)?.push(point);
  }

  for (const [continent, cities] of groups) {
    if (cities.length === 0) continue;

    const header = document.createElement("li");
    header.className = "score-results__continent";
    header.textContent = continent;
    results.append(header);

    for (const point of cities) {
      const item  = document.createElement("li");
      const score = document.createElement("span");
      const name  = document.createElement("span");
      const flag  = document.createElement("span");

      item.className  = "score-results__item";
      score.className = "score-results__score";
      name.className  = "score-results__name";
      flag.className  = "score-results__flag";

      score.textContent = `${Math.round(point.score * 100)}%`.padStart(4, " ");
      name.textContent  = point.name;
      flag.textContent  = point.flag;
      flag.title = countryNames?.of(point.country_code) ?? point.country_code;

      item.append(score, name, flag);
      results.append(item);
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
      properties: { name: m.name, score: m.score, flag: m.flag, country_code: m.country_code },
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

  map.addLayer({
    id: MARKER_LAYER_ID,
    type: "circle",
    source: MARKER_SOURCE_ID,
    paint: {
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 1, 3, 6, 5, 10, 7],
      "circle-color": SCORE_COLOR_EXPR,
      "circle-opacity": ["interpolate", ["linear"], ["get", "score"], 0, 0.25, 0.5, 0.7, 1, 1],
      "circle-stroke-width": 1,
      "circle-stroke-color": "#111111",
      "circle-stroke-opacity": 0.6,
    },
  });

  // Popup on hover.
  const popup = new window.maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 10 });

  map.on("mouseenter", MARKER_LAYER_ID, (e) => {
    map.getCanvas().style.cursor = "pointer";
    const props = e.features[0].properties;
    popup
      .setLngLat(e.lngLat)
      .setHTML(`<span class="city-popup">${props.flag} ${props.name} &nbsp; ${Math.round(props.score * 100)}%</span>`)
      .addTo(map);
  });

  map.on("mouseleave", MARKER_LAYER_ID, () => {
    map.getCanvas().style.cursor = "";
    popup.remove();
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
          properties: { name: c.name },
        })),
      };

      map.addSource(LANDMARK_SOURCE_ID, { type: "geojson", data: geojson });

      // Dim gray dots — always visible for geographic orientation.
      map.addLayer(
        {
          id: LANDMARK_LAYER_ID,
          type: "circle",
          source: LANDMARK_SOURCE_ID,
          paint: {
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 1, 1.5, 6, 3, 10, 4],
            "circle-color": "#555555",
            "circle-opacity": 0.6,
            "circle-stroke-width": 0,
          },
        },
        // Insert below scored markers so scored cities visually dominate.
        BORDER_LAYER_ID,
      );

      const landmarkPopup = new window.maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 8 });

      map.on("mouseenter", LANDMARK_LAYER_ID, (e) => {
        map.getCanvas().style.cursor = "default";
        landmarkPopup
          .setLngLat(e.lngLat)
          .setHTML(`<span class="city-popup">${e.features[0].properties.name}</span>`)
          .addTo(map);
      });

      map.on("mouseleave", LANDMARK_LAYER_ID, () => {
        landmarkPopup.remove();
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

function showTooltip(data, x, y) {
  if (!tooltip || !data.found) {
    if (tooltip) tooltip.hidden = true;
    return;
  }

  const pct = v => `${Math.round(v * 100)}%`.padStart(4);
  const temp = `${data.avg_temp_c > 0 ? "+" : ""}${data.avg_temp_c.toFixed(1)}°C`.padStart(8);
  const rain = `${Math.round(data.avg_precip_mm)}mm/mo`.padStart(8);
  const sun  = `${Math.round(100 - data.avg_cloud_pct)}% sun`.padStart(8);

  tooltip.textContent =
    `temp ${temp}  ${pct(data.temp_score)}\n` +
    `rain ${rain}  ${pct(data.rain_score)}\n` +
    `sun  ${sun}  ${pct(data.cloud_score)}`;

  tooltip.style.left = `${x}px`;
  tooltip.style.top  = `${y}px`;
  tooltip.hidden = false;
}

function hideTooltip() {
  if (tooltip) tooltip.hidden = true;
}

function fetchProbe(lat, lon, clientX, clientY) {
  const prefs = getCurrentPreferences();
  if (!prefs) return;

  if (probeController) probeController.abort();
  probeController = new AbortController();

  const params = new URLSearchParams({ lat, lon, ...prefs });

  fetch(`/probe?${params}`, { signal: probeController.signal })
    .then(r => r.json())
    .then(data => showTooltip(data, clientX, clientY))
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

  // Probe on hover — debounced to avoid hammering the server.
  map.on("mousemove", (e) => {
    clearTimeout(probeTimer);
    probeTimer = setTimeout(() => {
      fetchProbe(e.lngLat.lat, e.lngLat.lng, e.originalEvent.clientX, e.originalEvent.clientY);
    }, 80);
  });

  map.on("mouseleave", () => {
    clearTimeout(probeTimer);
    if (probeController) probeController.abort();
    hideTooltip();
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
  const { scores, markers, heatmap } = response;

  renderScoreList(scores ?? []);

  if (!map) return;

  const imageUrl = heatmap || EMPTY_IMAGE;
  const markerList = markers ?? [];

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
