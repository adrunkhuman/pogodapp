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
            triggerBody("htmx:afterRequest", { elt: form, xhr: { status: 200, responseText: '{"scores":[],"heatmap":""}' } });

            if (!errorIndicator.hidden) throw new Error("successful request should clear error indicator");
            if (!loadingIndicator.hidden) throw new Error("loading indicator should stay hidden after success");
            if (renderCalls.length !== 1) throw new Error(`expected one render call, got ${renderCalls.length}`);
            """
        )
    )
