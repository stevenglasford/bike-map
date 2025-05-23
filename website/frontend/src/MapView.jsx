// frontend/src/MapView.jsx
import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Tooltip, Popup } from 'react-leaflet';
import { HeatmapLayer } from 'react-leaflet-heatmap-layer';

function MapView() {
  const [data, setData] = useState(null);
  const [personal, setPersonal] = useState(false);

  useEffect(() => {
    fetchData();
  }, [personal]);

  const fetchData = async () => {
    const url = 'http://localhost:5000/api/data' + (personal ? '?personal=1' : '');
    const res = await fetch(url);
    const json = await res.json();
    setData(json);
  };

  if (!data) return <div>Loading map data...</div>;

  // Determine one map center (e.g., first point)
  const center = data.speed_points.length > 0 
      ? [data.speed_points[0][0], data.speed_points[0][1]] 
      : [0, 0];

  // Example custom icons (in a real app use actual images)
  const icons = {
    pedestrian: new L.Icon({iconUrl: 'pedestrian.png', iconSize: [25, 25]}),
    car: new L.Icon({iconUrl: 'car.png', iconSize: [25, 25]}),
    bike: new L.Icon({iconUrl: 'bike.png', iconSize: [25, 25]}),
    bus: new L.Icon({iconUrl: 'bus.png', iconSize: [25, 25]}),
    light: new L.Icon({iconUrl: 'traffic_light.png', iconSize: [25, 25]})
  };

  return (
    <div>
      <h2 style={{ backgroundColor:'#000080', color:'#fff', padding:'5px' }}>
        Map View ({personal ? "My Data" : "All Data"})
      </h2>
      <button onClick={() => setPersonal(!personal)}>
        {personal ? "Show All Data" : "Show My Data"}
      </button>
      <MapContainer center={center} zoom={16} style={{ height: "600px", margin: '10px' }}>
        {/* OpenStreetMap tiles */}
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
        />
        {/* Speed heatmap (green=slow, red=fast gradient) */}
        <HeatmapLayer
          fitBoundsOnLoad
          fitBoundsOnUpdate
          points={data.speed_points.map(p => ({lat: p[0], lng: p[1], intensity: p[2]}))}
          longitudeExtractor={p => p.lng}
          latitudeExtractor={p => p.lat}
          intensityExtractor={p => p.intensity}
          gradient={{0.4: 'green', 0.65: 'yellow', 1: 'red'}}
          radius={25}
          blur={15}
        />
        {/* Sound heatmap (green=quiet, red=loud) */}
        <HeatmapLayer
          points={data.sound_points.map(p => ({lat: p[0], lng: p[1], intensity: p[2]}))}
          longitudeExtractor={p => p.lng}
          latitudeExtractor={p => p.lat}
          intensityExtractor={p => p.intensity}
          gradient={{0.2: 'blue', 0.5: 'lime', 1: 'red'}}
          radius={20}
          blur={15}
        />
        {/* Object icons */}
        {data.markers.map((obj, idx) => (
          <Marker 
            key={idx} 
            position={[obj.lat, obj.lon]} 
            icon={icons[obj.type] || icons['pedestrian']}
            // Optionally rotate icon by direction (advanced: use a rotated DivIcon)
          >
            <Tooltip>
              {obj.type.charAt(0).toUpperCase() + obj.type.slice(1)}<br/>
              Speed: {obj.speed} mph<br/>
              Direction: {obj.direction}°
            </Tooltip>
          </Marker>
        ))}
        {/* Traffic lights */}
        {data.traffic_lights.map((light, i) => (
          <Marker key={i} position={[light.lat, light.lon]} icon={icons['light']}>
            <Popup>
              Avg wait: {light.avg_wait}s<br/>
              Green success: {light.green_success}%
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}

export default MapView;