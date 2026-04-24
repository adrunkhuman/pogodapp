from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path


def _run_app_runtime_scenario(scenario: str) -> None:
    root = Path(__file__).resolve().parents[1]
    app_script = (root / "frontend" / "static" / "app.js").read_text(encoding="utf-8")
    script = textwrap.dedent(
        rf"""
        const vm = require("node:vm");

        class HTMLElementStub {{
          constructor(id = "") {{
            this.id = id;
            this.hidden = false;
            this.textContent = "";
            this.value = "";
            this.dataset = {{}};
            this.style = {{
              values: new Map(),
              setProperty(name, value) {{
                this.values.set(name, value);
              }},
            }};
          }}
        }}

        class HTMLInputElementStub extends HTMLElementStub {{
          constructor(id, name, min, max, value, field) {{
            super(id);
            this.name = name;
            this.min = String(min);
            this.max = String(max);
            this.step = "1";
            this.value = String(value);
            this.type = "range";
            this.dataset = {{ field }};
            this.listeners = new Map();
          }}

          addEventListener(name, handler) {{
            this.listeners.set(name, handler);
          }}
        }}

        class HTMLOutputElementStub extends HTMLElementStub {{
          constructor(id, htmlFor) {{
            super(id);
            this.htmlFor = htmlFor;
          }}
        }}

        const bodyHandlers = new Map();
        const documentHandlers = new Map();
        const renderCalls = [];

        const preferredDayInput = new HTMLInputElementStub("preferred_day_temperature", "preferred_day_temperature", -5, 35, 22, "preferred_day_temperature");
        const summerHeatInput = new HTMLInputElementStub("summer_heat_limit", "summer_heat_limit", -5, 42, 10, "summer_heat_limit");
        const winterColdInput = new HTMLInputElementStub("winter_cold_limit", "winter_cold_limit", -15, 35, 30, "winter_cold_limit");
        const drynessInput = new HTMLInputElementStub("dryness_preference", "dryness_preference", 0, 100, 60, "dryness_preference");
        const sunshineInput = new HTMLInputElementStub("sunshine_preference", "sunshine_preference", 0, 100, 60, "sunshine_preference");
        const inputs = [preferredDayInput, summerHeatInput, winterColdInput, drynessInput, sunshineInput];

        const outputs = new Map(inputs.map((input) => [input.id, new HTMLOutputElementStub(`${{input.id}}-output`, input.id)]));
        const loadingIndicator = new HTMLElementStub("score-loading-indicator");
        loadingIndicator.hidden = true;
        const errorIndicator = new HTMLElementStub("score-error-indicator");
        errorIndicator.hidden = true;

        const form = new HTMLElementStub("preferences");
        form.elements = {{
          namedItem(name) {{
            return inputs.find((input) => input.name === name) ?? null;
          }},
        }};
        form.querySelectorAll = (selector) => selector === "input[type='range']" ? inputs : [];
        form.querySelector = (selector) => {{
          const match = selector.match(/^output\[for='(.+)'\]$/);
          return match ? outputs.get(match[1]) ?? null : null;
        }};

        globalThis.window = globalThis;
        globalThis.HTMLElement = HTMLElementStub;
        globalThis.HTMLInputElement = HTMLInputElementStub;
        globalThis.document = {{
          readyState: "complete",
          body: {{
            addEventListener(name, handler) {{
              bodyHandlers.set(name, handler);
            }},
          }},
          addEventListener(name, handler) {{
            documentHandlers.set(name, handler);
          }},
          getElementById(id) {{
            if (id === "preferences") return form;
            if (id === "score-loading-indicator") return loadingIndicator;
            if (id === "score-error-indicator") return errorIndicator;
            return null;
          }},
        }};
        globalThis.renderScores = (payload) => {{
          renderCalls.push(payload);
        }};

        vm.runInThisContext({app_script!r});

        const triggerBody = (name, detail) => {{
          const handler = bodyHandlers.get(name);
          if (!handler) throw new Error(`missing body handler ${{name}}`);
          handler({{ detail }});
        }};

        {scenario}
        """
    )

    result = subprocess.run(["node", "-e", script], cwd=root, capture_output=True, text=True, check=False)  # noqa: S603,S607
    assert result.returncode == 0, result.stderr or result.stdout


def _run_sidebar_runtime_scenario(scenario: str) -> None:
    root = Path(__file__).resolve().parents[1]
    sidebar_script = (root / "frontend" / "static" / "map-sidebar.js").read_text(encoding="utf-8")
    script = textwrap.dedent(
        rf"""
        const vm = require("node:vm");

        class ElementStub {{
          constructor(tagName) {{
            this.tagName = tagName;
            this.children = [];
            this.className = "";
            this.tabIndex = null;
            this.textContent = "";
            this.title = "";
            this.type = "";
            this.listeners = new Map();
          }}

          append(...children) {{
            this.children.push(...children);
          }}

          replaceChildren(...children) {{
            this.children = [...children];
          }}

          addEventListener(name, handler) {{
            this.listeners.set(name, handler);
          }}
        }}

        const results = new ElementStub("ul");
        const focusCalls = [];

        globalThis.document = {{
          createElement(tagName) {{ return new ElementStub(tagName); }},
          getElementById(id) {{ return id === "score-results-list" ? results : null; }},
        }};
        globalThis.CONTINENT_ORDER = ["Europe", "Asia", "Africa", "North America", "South America", "Oceania"];
        globalThis.continentVisibleCounts = new Map();
        globalThis.visibleCountForContinent = (continent) => globalThis.continentVisibleCounts.get(continent) ?? 5;
        globalThis.nextVisibleCount = (currentVisibleCount) => Math.ceil((currentVisibleCount + 1) / 5) * 5;
        globalThis.currentScores = [];
        globalThis.mapLoaded = false;
        globalThis.countryNames = {{ of(code) {{ return code === "CO" ? "Colombia" : code; }} }};
        globalThis.focusCityFromList = (point) => focusCalls.push(point);
        globalThis.applyMarkers = () => {{}};
        globalThis.visibleMarkers = () => [];

        vm.runInThisContext({sidebar_script!r});

        {scenario}
        """
    )

    result = subprocess.run(["node", "-e", script], cwd=root, capture_output=True, text=True, check=False)  # noqa: S603,S607
    assert result.returncode == 0, result.stderr or result.stdout


def test_app_runtime_preserves_original_slider_bounds_when_typical_day_changes() -> None:
    _run_app_runtime_scenario(
        textwrap.dedent(
            """
            if (summerHeatInput.min !== "22") throw new Error(`expected initial summer min 22, got ${summerHeatInput.min}`);
            if (winterColdInput.max !== "22") throw new Error(`expected initial winter max 22, got ${winterColdInput.max}`);
            if (summerHeatInput.value !== "22") throw new Error(`expected summer value clamped to 22, got ${summerHeatInput.value}`);
            if (winterColdInput.value !== "22") throw new Error(`expected winter value clamped to 22, got ${winterColdInput.value}`);

            preferredDayInput.value = "35";
            preferredDayInput.listeners.get("input")();

            if (summerHeatInput.min !== "35") throw new Error(`expected summer min 35, got ${summerHeatInput.min}`);
            if (winterColdInput.max !== "35") throw new Error(`expected winter max restored to 35, got ${winterColdInput.max}`);

            preferredDayInput.value = "-5";
            preferredDayInput.listeners.get("input")();

            if (summerHeatInput.min !== "-5") throw new Error(`expected summer min restored to -5, got ${summerHeatInput.min}`);
            if (winterColdInput.max !== "-5") throw new Error(`expected winter max lowered to -5, got ${winterColdInput.max}`);
            if (winterColdInput.value !== "-5") throw new Error(`expected winter value clamped to -5, got ${winterColdInput.value}`);
            """
        )
    )


def test_app_runtime_shows_generic_error_and_clears_it_after_success() -> None:
    _run_app_runtime_scenario(
        textwrap.dedent(
            """
            triggerBody("htmx:beforeRequest", { elt: form });
            if (loadingIndicator.hidden) throw new Error("loading indicator should show before request");
            if (!errorIndicator.hidden) throw new Error("error indicator should be hidden before request");

            triggerBody("htmx:afterRequest", { elt: form, xhr: { status: 503, responseText: '{"detail":"boom"}' } });

            if (!loadingIndicator.hidden) throw new Error("loading indicator should hide after failed request");
            if (errorIndicator.hidden) throw new Error("error indicator should show after failed request");
            if (errorIndicator.textContent !== "Could not calculate scores.") throw new Error(`unexpected error text ${errorIndicator.textContent}`);
            if (renderCalls.length !== 0) throw new Error("failed request should not render scores");

            triggerBody("htmx:beforeRequest", { elt: form });
            triggerBody("htmx:afterRequest", { elt: form, xhr: { status: 200, responseText: '{"scores":[],"heatmap_url":""}' } });

            if (!errorIndicator.hidden) throw new Error("successful request should clear error indicator");
            if (!loadingIndicator.hidden) throw new Error("loading indicator should stay hidden after success");
            if (renderCalls.length !== 1) throw new Error(`expected one render call, got ${renderCalls.length}`);
            """
        )
    )


def test_app_runtime_resyncs_temperature_controls_before_htmx_submit() -> None:
    _run_app_runtime_scenario(
        textwrap.dedent(
            """
            preferredDayInput.value = "18";
            summerHeatInput.min = "-5";
            summerHeatInput.value = "10";
            winterColdInput.max = "35";
            winterColdInput.value = "20";

            triggerBody("htmx:beforeRequest", { elt: form });

            if (summerHeatInput.min !== "18") throw new Error(`expected summer min 18, got ${summerHeatInput.min}`);
            if (summerHeatInput.value !== "18") throw new Error(`expected summer value clamped to 18, got ${summerHeatInput.value}`);
            if (winterColdInput.max !== "18") throw new Error(`expected winter max 18, got ${winterColdInput.max}`);
            if (winterColdInput.value !== "18") throw new Error(`expected winter value clamped to 18, got ${winterColdInput.value}`);
            """
        )
    )


def test_app_runtime_surfaces_validation_error_message_for_invalid_preferences() -> None:
    _run_app_runtime_scenario(
        textwrap.dedent(
            """
            triggerBody("htmx:beforeRequest", { elt: form });
            triggerBody("htmx:afterRequest", {
              elt: form,
              xhr: {
                status: 422,
                responseText: '{"detail":[{"msg":"preferred_day_temperature must be greater than or equal to winter_cold_limit"}]}'
              }
            });

            if (errorIndicator.hidden) throw new Error("validation error should show");
            if (errorIndicator.textContent !== "preferred_day_temperature must be greater than or equal to winter_cold_limit") {
              throw new Error(`unexpected validation text ${errorIndicator.textContent}`);
            }
            """
        )
    )


def test_sidebar_runtime_renders_city_labels_and_focus_handlers() -> None:
    _run_sidebar_runtime_scenario(
        textwrap.dedent(
            """
            const scores = [{
              name: "Bogota",
              continent: "South America",
              country_code: "CO",
              flag: "🇨🇴",
              score: 0.91,
              lat: 4.711,
              lon: -74.0721,
              probe_lat: 4.7083,
              probe_lon: -74.0417,
            }];

            renderScoreList(scores);

            const cityItem = results.children.find((item) => item.className === "score-results__item");
            if (!cityItem) throw new Error("missing rendered city item");
            const renderedText = cityItem.children.map((child) => child.textContent).join(" ");
            if (!renderedText.includes("Bogota")) throw new Error(`missing city name in ${renderedText}`);
            if (!renderedText.includes("🇨🇴")) throw new Error(`missing flag in ${renderedText}`);
            if (renderedText.includes("4.711")) throw new Error(`rendered coordinates instead of label: ${renderedText}`);

            cityItem.listeners.get("click")();

            if (focusCalls.length !== 1) throw new Error(`expected one focus call, got ${focusCalls.length}`);
            if (focusCalls[0].name !== "Bogota") throw new Error("focused wrong city");
            """
        )
    )
