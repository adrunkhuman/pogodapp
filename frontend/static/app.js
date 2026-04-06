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

function syncRangeControl(input, output) {
  const minimum = Number(input.min);
  const maximum = Number(input.max);
  const value = Number(input.value);
  const progress = ((value - minimum) / (maximum - minimum)) * 100;

  input.style.setProperty("--range-progress", `${progress}%`);
  if (output) output.value = formatControlValue(input);
}

function bindPreferenceControls(form) {
  for (const input of form.querySelectorAll("input[type='range']")) {
    const output = form.querySelector(`output[for='${input.id}']`);
    const syncControl = () => syncRangeControl(input, output);
    syncControl();
    input.addEventListener("input", syncControl);
  }
}

function bindScoreHandoff(form) {
  // Successful `#preferences` submissions return the raw `/score` JSON payload,
  // which the split map scripts consume through `window.renderScores`.
  document.body.addEventListener("htmx:afterRequest", (event) => {
    if (event.detail.elt !== form || event.detail.xhr.status !== 200) return;
    window.renderScores(JSON.parse(event.detail.xhr.responseText));
  });
}

function initializeAppShell() {
  const form = document.getElementById("preferences");
  if (!form) return;
  bindPreferenceControls(form);
  bindScoreHandoff(form);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeAppShell, { once: true });
} else {
  initializeAppShell();
}
