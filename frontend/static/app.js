"use strict";

const CONTROL_FORMATTERS = {
  preferred_day_temperature: (value) => `${value}C`,
  summer_heat_limit: (value) => `too hot above ${value}C`,
  winter_cold_limit: (value) => `too cold below ${value}C`,
  dryness_preference: (value) => {
    if (value >= 75) return "prefer dry";
    if (value >= 40) return "some rain is fine";
    return "rain is okay";
  },
  sunshine_preference: (value) => {
    if (value >= 75) return "need sun";
    if (value >= 40) return "mixed skies";
    return "clouds are fine";
  },
};

function formatControlValue(input) {
  const formatter = CONTROL_FORMATTERS[input.dataset.field];
  const value = Number(input.value);
  return formatter ? formatter(value) : input.value;
}

function constrainTemperatureControls(form) {
  const preferredDayInput = form.elements.namedItem("preferred_day_temperature");
  const summerHeatInput = form.elements.namedItem("summer_heat_limit");
  const winterColdInput = form.elements.namedItem("winter_cold_limit");

  if (!(preferredDayInput instanceof HTMLInputElement)) return;
  if (!(summerHeatInput instanceof HTMLInputElement)) return;
  if (!(winterColdInput instanceof HTMLInputElement)) return;

  const preferredDayValue = Number(preferredDayInput.value);
  const summerHeatMinimum = Number(summerHeatInput.dataset.minimum || summerHeatInput.min);
  const winterColdMaximum = Number(winterColdInput.dataset.maximum || winterColdInput.max);

  summerHeatInput.dataset.minimum = String(summerHeatMinimum);
  winterColdInput.dataset.maximum = String(winterColdMaximum);

  summerHeatInput.min = String(Math.max(summerHeatMinimum, preferredDayValue));
  winterColdInput.max = String(Math.min(winterColdMaximum, preferredDayValue));

  if (Number(summerHeatInput.value) < preferredDayValue) {
    summerHeatInput.value = preferredDayInput.value;
  }

  if (Number(winterColdInput.value) > preferredDayValue) {
    winterColdInput.value = preferredDayInput.value;
  }
}

function syncRangeControl(input, output) {
  const minimum = Number(input.min);
  const maximum = Number(input.max);
  const value = Number(input.value);
  const progress = ((value - minimum) / (maximum - minimum)) * 100;

  input.style.setProperty("--range-progress", `${progress}%`);
  if (output) output.value = formatControlValue(input);
}

function extractScoreErrorMessage(xhr) {
  if (xhr.status === 422) {
    try {
      const payload = JSON.parse(xhr.responseText);
      const detail = Array.isArray(payload.detail) ? payload.detail : [];
      const firstMessage = detail.find((item) => typeof item?.msg === "string")?.msg;
      if (firstMessage) return firstMessage;
    } catch {
      return "Invalid preferences.";
    }

    return "Invalid preferences.";
  }

  return "Could not calculate scores.";
}

function bindPreferenceControls(form) {
  const syncAllControls = () => {
    constrainTemperatureControls(form);
    for (const input of form.querySelectorAll("input[type='range']")) {
      const output = form.querySelector(`output[for='${input.id}']`);
      syncRangeControl(input, output);
    }
  };

  for (const input of form.querySelectorAll("input[type='range']")) {
    input.addEventListener("input", syncAllControls);
  }

  syncAllControls();
  return syncAllControls;
}

function buildHeatmapUrl(form) {
  const params = new URLSearchParams();
  for (const input of form.querySelectorAll("input[type='range']")) {
    params.set(input.name, input.value);
  }
  return `/heatmap?${params.toString()}`;
}

function bindScoreHandoff(form, syncControls) {
  const loadingIndicator = document.getElementById("score-loading-indicator");
  const errorIndicator = document.getElementById("score-error-indicator");

  const setLoading = (isLoading) => {
    if (!loadingIndicator) return;
    loadingIndicator.hidden = !isLoading;
  };

  const setError = (message) => {
    if (!(errorIndicator instanceof HTMLElement)) return;
    errorIndicator.hidden = message.length === 0;
    errorIndicator.textContent = message;
  };

  document.body.addEventListener("htmx:beforeRequest", (event) => {
    if (event.detail.elt !== form) return;
    syncControls();
    if (typeof window.renderHeatmap === "function") {
      window.renderHeatmap(buildHeatmapUrl(form));
    }
    setError("");
    setLoading(true);
  });

  // HTMX returns raw /score JSON here; hand it straight to the map renderer.
  document.body.addEventListener("htmx:afterRequest", (event) => {
    if (event.detail.elt !== form) return;
    setLoading(false);
    if (event.detail.xhr.status !== 200) {
      setError(extractScoreErrorMessage(event.detail.xhr));
      return;
    }

    setError("");
    window.renderScores(JSON.parse(event.detail.xhr.responseText));
  });
}

function initializeAppShell() {
  const form = document.getElementById("preferences");
  if (!form) return;
  const syncControls = bindPreferenceControls(form);
  bindScoreHandoff(form, syncControls);
  if (window.POGODAPP_INITIAL_SCORES) {
    window.renderScores(window.POGODAPP_INITIAL_SCORES);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeAppShell, { once: true });
} else {
  initializeAppShell();
}
