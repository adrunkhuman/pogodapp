from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path


def _run_probe_runtime_scenario(scenario: str) -> None:
    root = Path(__file__).resolve().parents[1]
    map_core = (root / "frontend" / "static" / "map-core.js").read_text(encoding="utf-8")
    map_probe = (root / "frontend" / "static" / "map-probe.js").read_text(encoding="utf-8")
    script = textwrap.dedent(
        f"""
        const vm = require("node:vm");

        const eventHandlers = new Map();
        const timers = new Map();
        let nextTimerId = 1;
        const pendingFetches = [];
        const tooltipEl = {{ hidden: true, innerHTML: "", style: {{}}, getBoundingClientRect: () => ({{ width: 120, height: 40 }}) }};
        const form = {{ id: "preferences" }};
        const mapHandlers = new Map();
        const mapStub = {{
          on(eventName, layerOrHandler, handler) {{
            const key = handler ? `${{eventName}}:${{layerOrHandler}}` : eventName;
            mapHandlers.set(key, handler ?? layerOrHandler);
          }},
          getCanvas() {{
            return {{ style: {{ cursor: "" }} }};
          }},
          getContainer() {{
            return {{ getBoundingClientRect: () => ({{ left: 0, top: 0, right: 800, bottom: 600 }}) }};
          }},
          queryRenderedFeatures() {{
            return [];
          }},
          getSource() {{
            return null;
          }},
          project([lon, lat]) {{
            return {{ x: lon, y: lat }};
          }},
        }};

        globalThis.window = globalThis;
        globalThis.setTimeout = (fn, ms) => {{
          const id = nextTimerId++;
          timers.set(id, {{ fn, ms }});
          return id;
        }};
        globalThis.clearTimeout = (id) => {{
          timers.delete(id);
        }};
        globalThis.requestAnimationFrame = () => 0;
        globalThis.cancelAnimationFrame = () => {{}};
        globalThis.fetch = (url, options = {{}}) => new Promise((resolve, reject) => {{
          pendingFetches.push({{ url, options, resolve, reject }});
        }});
        globalThis.document = {{
          readyState: "complete",
          addEventListener(name, handler) {{
            eventHandlers.set(name, handler);
          }},
          getElementById(id) {{
            if (id === "map-probe-tooltip") return tooltipEl;
            if (id === "preferences") return form;
            return null;
          }},
        }};
        globalThis.FormData = class FormData {{
          constructor(_form) {{}}
          entries() {{
            return [
              ["preferred_day_temperature", "22"],
              ["summer_heat_limit", "30"],
              ["winter_cold_limit", "5"],
              ["dryness_preference", "60"],
              ["sunshine_preference", "60"],
            ][Symbol.iterator]();
          }}
        }};

        vm.runInThisContext({map_core!r});
        vm.runInThisContext({map_probe!r});
        vm.runInThisContext(`
          globalThis.__probeTest = {{
            mapHandlers,
            eventHandlers,
            pendingFetches,
            tooltip,
            setMap(value) {{ map = value; }},
            setMapLoaded(value) {{ mapLoaded = value; }},
            pendingTimerCount(delay) {{
              return [...timers.values()].filter((timer) => timer.ms === delay).length;
            }},
            runTimersByDelay(delay) {{
              const due = [...timers.entries()].filter(([, timer]) => timer.ms === delay);
              for (const [id, timer] of due) {{
                timers.delete(id);
                timer.fn();
              }}
            }},
          }};
        `);

        async function flushMicrotasks() {{
          await Promise.resolve();
          await Promise.resolve();
        }}

        async function resolveFetch(index, payload) {{
          const request = pendingFetches[index];
          request.resolve({{
            ok: true,
            json: async () => payload,
          }});
          await flushMicrotasks();
        }}

        async function main() {{
          const test = globalThis.__probeTest;
          test.setMap(mapStub);
          test.setMapLoaded(true);
          {scenario}
        }}

        main().catch((error) => {{
          console.error(error);
          process.exit(1);
        }});
        """
    )

    result = subprocess.run(["node", "-e", script], cwd=root, capture_output=True, text=True, check=False)  # noqa: S603,S607
    assert result.returncode == 0, result.stderr or result.stdout


def test_probe_runtime_ignores_inflight_layer_response_after_mouseleave() -> None:
    _run_probe_runtime_scenario(
        textwrap.dedent(
            """
            registerLayerProbeHandlers("marker", { cursor: "pointer", header: () => "Flag City" });
            mapHandlers.get("mouseenter:marker")({
              lngLat: { lat: 12, lng: 34 },
              originalEvent: { clientX: 10, clientY: 20 },
            });
            if (pendingFetches.length !== 1) throw new Error(`expected 1 pending fetch, got ${pendingFetches.length}`);

            mapHandlers.get("mouseleave:marker")();
            await resolveFetch(0, { found: true, overall_score: 0.7, metrics: [] });

            if (!tooltip.hidden) throw new Error("stale probe response re-showed the tooltip after mouseleave");
            if (tooltip.innerHTML !== "") throw new Error("stale probe response rewrote tooltip content after mouseleave");
            """
        )
    )


def test_probe_runtime_snaps_query_params_to_backend_grid() -> None:
    _run_probe_runtime_scenario(
        textwrap.dedent(
            """
            fetchProbe(37.51, -122.02, 10, 20, null, { cooldownMs: 0, requestToken: ++probeRequestToken });

            if (pendingFetches.length !== 1) throw new Error(`expected 1 probe request, got ${pendingFetches.length}`);
            const url = new URL(pendingFetches[0].url, "https://example.test");

            if (url.pathname !== "/probe") throw new Error(`unexpected probe path ${url.pathname}`);
            if (url.searchParams.get("lat") !== "37.5417") throw new Error(`unexpected snapped lat ${url.searchParams.get("lat")}`);
            if (url.searchParams.get("lon") !== "-122.0417") throw new Error(`unexpected snapped lon ${url.searchParams.get("lon")}`);
            if (url.searchParams.get("preferred_day_temperature") !== "22") throw new Error("missing preferences in probe query");
            """
        )
    )


def test_probe_runtime_cancels_queued_hover_probe_after_preference_change() -> None:
    _run_probe_runtime_scenario(
        textwrap.dedent(
            """
            scheduleHoverProbe({
              point: { x: 1, y: 2 },
              lngLat: { lat: 12, lng: 34 },
              originalEvent: { clientX: 10, clientY: 20 },
            });
            __probeTest.runTimersByDelay(80);

            eventHandlers.get("input")({
              target: {
                closest(selector) {
                  return selector === "#preferences" ? {} : null;
                },
              },
            });
            __probeTest.runTimersByDelay(250);

            if (pendingFetches.length !== 0) throw new Error(`expected queued hover probe to be cancelled, got ${pendingFetches.length} fetches`);
            """
        )
    )


def test_probe_runtime_keeps_new_timeout_when_older_request_finishes() -> None:
    _run_probe_runtime_scenario(
        textwrap.dedent(
            """
            fetchProbe(12, 34, 10, 20, null, { cooldownMs: 0, requestToken: ++probeRequestToken });
            if (__probeTest.pendingTimerCount(5000) !== 1) throw new Error("expected first probe timeout");

            fetchProbe(13, 35, 11, 21, null, { cooldownMs: 0, requestToken: ++probeRequestToken });
            if (__probeTest.pendingTimerCount(5000) !== 1) throw new Error("expected one active timeout after replacing probe");

            pendingFetches[0].reject(new Error("aborted"));
            await flushMicrotasks();

            if (__probeTest.pendingTimerCount(5000) !== 1) throw new Error("older probe cleanup cleared the newer timeout");
            """
        )
    )


def test_probe_runtime_hides_visible_tooltip_on_preference_change() -> None:
    _run_probe_runtime_scenario(
        textwrap.dedent(
            """
            tooltip.hidden = false;
            tooltip.innerHTML = "stale";

            eventHandlers.get("input")({
              target: {
                closest(selector) {
                  return selector === "#preferences" ? {} : null;
                },
              },
            });

            if (!tooltip.hidden) throw new Error("preference change should hide stale tooltip");
            """
        )
    )


def test_probe_runtime_ignores_cached_hover_probe_after_reset() -> None:
    _run_probe_runtime_scenario(
        textwrap.dedent(
            """
            fetchProbe(12, 34, 10, 20, null, { cooldownMs: 0, requestToken: ++probeRequestToken });
            await resolveFetch(0, { found: true, overall_score: 0.7, metrics: [] });

            scheduleHoverProbe({
              point: { x: 1, y: 2 },
              lngLat: { lat: 12, lng: 34 },
              originalEvent: { clientX: 10, clientY: 20 },
            });
            resetTransientMapUi();
            __probeTest.runTimersByDelay(80);

            if (!tooltip.hidden) throw new Error("reset should suppress cached hover tooltip reuse");
            """
        )
    )
