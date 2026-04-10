"use strict";

const probeCache = new Map();
const PROBE_GRID_DEGREES = window.POGODAPP_MAP_CONFIG?.probeGridDegrees ?? (5 / 60);
const PROBE_GRID_HALF_DEGREES = PROBE_GRID_DEGREES / 2;
const PROBE_MAX_LATITUDE_INDEX = Math.round(180 / PROBE_GRID_DEGREES) - 1;
const PROBE_MAX_LONGITUDE_INDEX = Math.round(360 / PROBE_GRID_DEGREES) - 1;

document.addEventListener("input", (event) => {
  if (event.target.closest("#preferences")) probeCache.clear();
}, { passive: true });

function getCurrentPreferences() {
  const form = document.getElementById("preferences");
  if (!form) return null;
  return Object.fromEntries(new FormData(form).entries());
}

function roundProbeCoordinate(value) {
  return Math.round(value * 10000) / 10000;
}

function snapProbeCoordinate(lat, lon) {
  const latitudeIndex = Math.min(
    Math.max(Math.round((90 - PROBE_GRID_HALF_DEGREES - lat) / PROBE_GRID_DEGREES), 0),
    PROBE_MAX_LATITUDE_INDEX,
  );
  const longitudeIndex = Math.min(
    Math.max(Math.round((lon - (-180 + PROBE_GRID_HALF_DEGREES)) / PROBE_GRID_DEGREES), 0),
    PROBE_MAX_LONGITUDE_INDEX,
  );

  return {
    lat: roundProbeCoordinate(90 - PROBE_GRID_HALF_DEGREES - latitudeIndex * PROBE_GRID_DEGREES),
    lon: roundProbeCoordinate(-180 + PROBE_GRID_HALF_DEGREES + longitudeIndex * PROBE_GRID_DEGREES),
  };
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

function registerLayerProbeHandlers(layerId, { cursor, header, coordinates = null }) {
  map.on("mouseenter", layerId, (event) => {
    hoveringLayer = true;
    map.getCanvas().style.cursor = cursor;
    clearTimeout(probeTimer);
    abortActiveProbe();
    const [lat, lon] = coordinates ? coordinates(event) : [event.lngLat.lat, event.lngLat.lng];
    fetchProbe(lat, lon, event.originalEvent.clientX, event.originalEvent.clientY, header(event), { hideDelayMs: null });
  });

  map.on("mouseleave", layerId, () => {
    hoveringLayer = false;
    map.getCanvas().style.cursor = "";
    hideTooltip();
  });
}

function probeTooltipRow(label, value, score) {
  return (
    `<div class="probe-tooltip__row">` +
    `<span class="probe-tooltip__label">${label}</span>` +
    `<span class="probe-tooltip__value">${value}</span>` +
    `<span class="probe-tooltip__metric">${scoreSpan(score)}</span>` +
    `</div>`
  );
}

function probeTooltipHeader(cityHeader, overallScore) {
  if (!cityHeader) {
    return `<div class="probe-tooltip__header"><span class="probe-tooltip__header-score">${scoreSpan(overallScore)}</span></div>`;
  }

  const cityName = cityHeader.replace(/^\S+\s+/, "");
  const cityFlag = cityHeader.match(/^\S+/)?.[0] ?? "";
  return (
    `<div class="probe-tooltip__header">` +
    `<span class="probe-tooltip__header-score">${scoreSpan(overallScore)}</span>` +
    `<span class="probe-tooltip__header-city">` +
    `<span class="probe-tooltip__header-name">${escapeHtml(cityName)}</span>` +
    `<span class="probe-tooltip__header-flag">${escapeHtml(cityFlag)}</span>` +
    `</span>` +
    `</div>`
  );
}

function showTooltip(data, x, y, cityHeader = null, { hideDelayMs = null } = {}) {
  if (!tooltip || !data.found) {
    if (tooltip) tooltip.hidden = true;
    return;
  }

  tooltip.innerHTML = probeTooltipHeader(cityHeader, data.overall_score) + data.metrics
    .map((metric) => probeTooltipRow(metric.label, metric.display_value, metric.score))
    .join("");

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

function runProbeRequest(cacheKey, params, clientX, clientY, cityHeader, { hideDelayMs = null } = {}) {
  abortActiveProbe();
  probeController = new AbortController();
  probeTimeoutId = setTimeout(() => probeController.abort(), PROBE_TIMEOUT_MS);

  fetch(`/probe?${params}`, { signal: probeController.signal })
    .then((response) => {
      if (!response.ok) throw new Error(`Probe request failed with ${response.status}`);
      return response.json();
    })
    .then((data) => {
      probeCache.set(cacheKey, data);
      showTooltip(data, clientX, clientY, cityHeader, { hideDelayMs });
    })
    .catch(() => {})
    .finally(() => clearTimeout(probeTimeoutId));
}

function queueProbeRequest(cacheKey, params, clientX, clientY, cityHeader, { hideDelayMs = null, cooldownMs = 0 } = {}) {
  cancelProbeCooldown();
  if (cooldownMs <= 0) {
    runProbeRequest(cacheKey, params, clientX, clientY, cityHeader, { hideDelayMs });
    return;
  }

  probeCooldownTimer = setTimeout(
    () => runProbeRequest(cacheKey, params, clientX, clientY, cityHeader, { hideDelayMs }),
    cooldownMs,
  );
}

function fetchProbe(lat, lon, clientX, clientY, cityHeader = null, { hideDelayMs = null, cooldownMs = 0 } = {}) {
  const prefs = getCurrentPreferences();
  if (!prefs) return;

  const snapped = snapProbeCoordinate(lat, lon);

  const cacheKey = `${snapped.lat},${snapped.lon},${new URLSearchParams(prefs)}`;
  const cached = probeCache.get(cacheKey);
  if (cached) {
    cancelProbeCooldown();
    abortActiveProbe();
    showTooltip(cached, clientX, clientY, cityHeader, { hideDelayMs });
    return;
  }

  const params = new URLSearchParams({ lat: snapped.lat, lon: snapped.lon, ...prefs });
  queueProbeRequest(cacheKey, params, clientX, clientY, cityHeader, { hideDelayMs, cooldownMs });
}

function scheduleHoverProbe(event) {
  if (hoveringLayer) return;

  clearTimeout(probeTimer);
  probeTimer = setTimeout(() => {
    const snapped = nearestFeatureAtPoint(event.point);
    if (snapped) {
      map.getCanvas().style.cursor = snapped.cursor;
      fetchProbe(snapped.lat, snapped.lon, event.originalEvent.clientX, event.originalEvent.clientY, snapped.header, {
        hideDelayMs: null,
        cooldownMs: PROBE_HOVER_COOLDOWN_MS,
      });
      return;
    }

    map.getCanvas().style.cursor = "";
    fetchProbe(event.lngLat.lat, event.lngLat.lng, event.originalEvent.clientX, event.originalEvent.clientY, null, {
      hideDelayMs: null,
      cooldownMs: PROBE_HOVER_COOLDOWN_MS,
    });
  }, PROBE_HOVER_DELAY_MS);
}

function resetTransientMapUi() {
  clearTimeout(probeTimer);
  cancelProbeCooldown();
  abortActiveProbe();
  hideTooltip();
  clearFocusedCity();
}
