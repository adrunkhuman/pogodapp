"use strict";

function applyHeatmap(heatmap) {
  const source = map.getSource(HEATMAP_SOURCE_ID);

  if (source) {
    source.updateImage({ url: heatmap });
    return;
  }

  map.addSource(HEATMAP_SOURCE_ID, {
    type: "image",
    url: heatmap,
    coordinates: WORLD_CORNERS,
  });

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

function markersToGeoJSON(markers) {
  return {
    type: "FeatureCollection",
    features: markers.map((marker) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [marker.lon, marker.lat] },
      properties: {
        name: marker.name,
        score: marker.score,
        flag: marker.flag,
        country_code: marker.country_code,
        probe_lat: marker.probe_lat,
        probe_lon: marker.probe_lon,
      },
    })),
  };
}

function applyMarkers(markers) {
  const source = map.getSource(MARKER_SOURCE_ID);
  const geojson = markersToGeoJSON(markers);

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
      "circle-radius": ["interpolate", ["linear"], ["zoom"], 1, 5, 6, 7, 10, 10],
      "circle-color": SCORE_COLOR_EXPR,
      "circle-opacity": 1.0,
      "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 1, 1.5, 6, 2, 10, 2.5],
      "circle-stroke-color": "#ffffff",
      "circle-stroke-opacity": 0.9,
      "circle-blur": 0.05,
    },
  });

  registerLayerProbeHandlers(MARKER_LAYER_ID, {
    cursor: "pointer",
    header: (event) => `${event.features[0].properties.flag} ${event.features[0].properties.name}`,
    coordinates: (event) => {
      const props = event.features[0].properties;
      return [Number(props.probe_lat ?? event.lngLat.lat), Number(props.probe_lon ?? event.lngLat.lng)];
    },
  });
}

function clearMarkers() {
  const source = map.getSource(MARKER_SOURCE_ID);
  if (source) source.setData({ type: "FeatureCollection", features: [] });
}

function loadLandmarkCities() {
  fetch(LANDMARK_CITIES_URL)
    .then((response) => response.json())
    .then((cities) => {
      const geojson = {
        type: "FeatureCollection",
        features: cities.map((city) => ({
          type: "Feature",
          geometry: { type: "Point", coordinates: [city.lon, city.lat] },
          properties: { name: city.name, country_code: city.country_code ?? null },
        })),
      };

      map.addSource(LANDMARK_SOURCE_ID, { type: "geojson", data: geojson });
      map.addLayer({
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
      });

      registerLayerProbeHandlers(LANDMARK_LAYER_ID, {
        cursor: "default",
        header: (event) => {
          const landmark = event.features[0].properties;
          return landmark.country_code ? `${countryFlag(landmark.country_code)} ${landmark.name}` : landmark.name;
        },
      });
    })
    .catch(() => {});
}
