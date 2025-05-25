import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Tooltip, Popup } from 'react-leaflet';
import HeatmapLayer from './HeatmapLayer';
import UploadForm from './UploadForm';
import CsvLocalLoader from './CsvLocalLoader';
import L from 'leaflet';

function MapView() {
  const [data, setData] = useState(null);
  const [personal, setPersonal] = useState(false);
  const [localMarkers, setLocalMarkers] = useState([]);
  const [alignedOutput, setAlignedOutput] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const url = '/api/data' + (personal ? '?personal=1' : '');
      const res = await fetch(url);
      const json = await res.json();
      setData(json);
    } catch (err) {
      console.error(err);
      setError("Failed to load data from server.");
    } finally {
      setLoading(false);
      setLocalMarkers([]);
      setAlignedOutput([]);
    }
  };

  useEffect(() => {
    fetchData();
  }, [personal]);

  const handleUploadComplete = () => {
    setTimeout(() => fetchData(), 10000);
  };

  const handleCsvData = ({ type, data }) => {
    if (type === "aligned_output") setAlignedOutput(data);
    else if (type === "markers") setLocalMarkers(data);
  };

  const center = data?.speed_points?.[0]
    ? [data.speed_points[0][0], data.speed_points[0][1]]
    : alignedOutput[0]
      ? [alignedOutput[0].lat, alignedOutput[0].lon]
      : [44.9778, -93.2650];

  const icons = {
    pedestrian: new L.Icon({ iconUrl: 'pedestrian.png', iconSize: [25, 25] }),
    car: new L.Icon({ iconUrl: 'car.png', iconSize: [25, 25] }),
    bike: new L.Icon({ iconUrl: 'bike.png', iconSize: [25, 25] }),
    bus: new L.Icon({ iconUrl: 'bus.png', iconSize: [25, 25] }),
    light: new L.Icon({ iconUrl: 'traffic_light.png', iconSize: [25, 25] })
  };

  return (
    <div>
      <h2 style={{ backgroundColor: '#000080', color: '#fff', padding: '5px' }}>
        Map View ({personal ? "My Data (CSV)" : "All Data (Server)"})
      </h2>

      <button onClick={() => setPersonal(!personal)}>
        {personal ? "Switch to All Data" : "Switch to My CSV Data"}
      </button>

      {personal ? (
        <CsvLocalLoader onDataLoaded={handleCsvData} />
      ) : (
        <div style={{ margin: '10px', padding: '10px', backgroundColor: '#f0f0f0' }}>
          <UploadForm onUploadComplete={handleUploadComplete} />
        </div>
      )}

      {loading && <p style={{ color: '#888' }}>Loading data...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}

      <MapContainer center={center} zoom={13} style={{ height: "600px", margin: '10px' }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
        />

        {/* Server-based data */}
        {!personal && data && (
          <>
            <HeatmapLayer
              fitBoundsOnLoad
              fitBoundsOnUpdate
              points={data.speed_points.map(p => ({ lat: p[0], lng: p[1], intensity: p[2] }))}
              latitudeExtractor={p => p.lat}
              longitudeExtractor={p => p.lng}
              intensityExtractor={p => p.intensity}
              gradient={{ 0.4: 'green', 0.65: 'yellow', 1: 'red' }}
              radius={25}
              blur={15}
            />
            <HeatmapLayer
              points={data.sound_points.map(p => ({ lat: p[0], lng: p[1], intensity: p[2] }))}
              latitudeExtractor={p => p.lat}
              longitudeExtractor={p => p.lng}
              intensityExtractor={p => p.intensity}
              gradient={{ 0.2: 'blue', 0.5: 'lime', 1: 'red' }}
              radius={20}
              blur={15}
            />
            {data.markers.map((obj, idx) => (
              <Marker
                key={idx}
                position={[obj.lat, obj.lon]}
                icon={icons[obj.type] || icons['pedestrian']}
              >
                <Tooltip>
                  {obj.type}<br />
                  Speed: {obj.speed} mph<br />
                  Direction: {obj.direction}°
                </Tooltip>
              </Marker>
            ))}
            {data.traffic_lights.map((light, i) => (
              <Marker key={i} position={[light.lat, light.lon]} icon={icons['light']}>
                <Popup>
                  Avg wait: {light.avg_wait}s<br />
                  Green success: {light.green_success}%
                </Popup>
              </Marker>
            ))}
          </>
        )}

        {/* Local CSV: aligned_output */}
        {personal && alignedOutput.length > 0 && (
          <>
            <HeatmapLayer
              points={alignedOutput.map(p => ({ lat: p.lat, lng: p.lon, intensity: p.speed }))}
              latitudeExtractor={p => p.lat}
              longitudeExtractor={p => p.lng}
              intensityExtractor={p => p.intensity}
              gradient={{ 0.2: 'green', 0.5: 'yellow', 1: 'red' }}
              radius={25}
              blur={15}
            />
            {alignedOutput.map((point, idx) => (
              <Marker
                key={`aligned-${idx}`}
                position={[point.lat, point.lon]}
                icon={L.divIcon({ className: 'invisible-icon' })}
              >
                <Tooltip direction="top" offset={[0, -10]} opacity={0.9}>
                  <div style={{ fontSize: '0.8em' }}>
                    <b>Second:</b> {point.second}<br />
                    <b>Lat:</b> {point.lat.toFixed(5)}<br />
                    <b>Lon:</b> {point.lon.toFixed(5)}<br />
                    <b>Speed:</b> {point.speed.toFixed(2)} mph
                  </div>
                </Tooltip>
              </Marker>
            ))}
          </>
        )}

        {/* Local CSV: markers */}
        {personal && localMarkers.length > 0 && localMarkers.map((obj, idx) => (
          <Marker
            key={`local-${idx}`}
            position={[obj.lat, obj.lon]}
            icon={icons[obj.type] || icons['pedestrian']}
          >
            <Tooltip>
              {obj.type}<br />
              Speed: {obj.speed} mph<br />
              Direction: {obj.direction}°
            </Tooltip>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}

export default MapView;