// src/HeatmapLayer.js
import { useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet.heat';
import { useEffect } from 'react';

function HeatmapLayer({ points = [], gradient, radius = 25, blur = 15 }) {
  const map = useMap();

  useEffect(() => {
    if (!points.length) return;

    const heatLayer = L.heatLayer(points, {
      radius,
      blur,
      gradient,
    }).addTo(map);

    return () => {
      map.removeLayer(heatLayer);
    };
  }, [map, points, radius, blur, gradient]);

  return null;
}

export default HeatmapLayer;