import React, { useState, useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Tooltip, Popup, Polyline, CircleMarker } from 'react-leaflet';
import HeatmapLayer from './HeatmapLayer';
import UploadForm from './UploadForm';
import CsvLocalLoader from './CsvLocalLoader';
import L from 'leaflet';
import { useMap } from 'react-leaflet';

function FitMapToAligned({ points }) {
  const map = useMap();
  useEffect(() => {
    if (points.length > 0) {
      const bounds = points.map(p => [p.lat, p.lon]);
      map.fitBounds(bounds, { padding: [30, 30] });
    }
  }, [points, map]);
  return null;
}

function MapView() {
  const [data, setData] = useState(null);
  const [personal, setPersonal] = useState(false);
  const [localMarkers, setLocalMarkers] = useState([]);
  const [alignedOutput, setAlignedOutput] = useState([]);
  const [mergedOutput, setMergedOutput] = useState([]);
  const [soundPoints, setSoundPoints] = useState([]);
  const [objectDetections, setObjectDetections] = useState([]);
  const [objectCounts, setObjectCounts] = useState([]);
  const [stoplights, setStoplights] = useState([]);
  const [frameNoise, setFrameNoise] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedObject, setSelectedObject] = useState(null);
  const [showObjectPaths, setShowObjectPaths] = useState(true);
  const [showStoplights, setShowStoplights] = useState(true);
  const [timeFilter, setTimeFilter] = useState({ min: 0, max: Infinity });

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
      // Clear local data when switching modes
      setLocalMarkers([]);
      setAlignedOutput([]);
      setObjectDetections([]);
      setObjectCounts([]);
      setStoplights([]);
    }
  };

  useEffect(() => {
    fetchData();
  }, [personal]);

  const handleUploadComplete = () => {
    setTimeout(() => fetchData(), 10000);
  };

  const handleCsvData = (payload) => {
    if (payload.type === "aligned_output") {
      alert(`Loaded ${payload.data.length} aligned points`);
      setAlignedOutput(payload.data);
      setMergedOutput([]);
    } else if (payload.type === "merged_output") {
      alert(`Loaded ${payload.data.length} merged points`);
      setMergedOutput(payload.data);
      setAlignedOutput([]);
    } else if (payload.type === "markers") {
      alert(`Loaded ${payload.data.length} markers`);
      setLocalMarkers(payload.data);
    } else if (payload.type === "sound_heatmap") {
      alert(`Loaded ${payload.data.length} sound points`);
      setSoundPoints(payload.data);
    } else if (payload.type === "frame_noise") {
      alert(`Loaded ${payload.data.length} frame noise points`);
      setFrameNoise(payload.data);
    } else if (payload.type === "object_detection") {
      alert(`Loaded ${payload.data.length} detected objects`);
      setObjectDetections(payload.data);
      // Calculate time range
      const times = payload.data.map(d => d.seconds);
      setTimeFilter({ min: Math.min(...times), max: Math.max(...times) });
    } else if (payload.type === "object_counts") {
      alert(`Loaded object count summary for ${payload.data.length} objects`);
      setObjectCounts(payload.data);
    } else if (payload.type === "stoplights") {
      alert(`Loaded ${payload.data.length} stoplight detections`);
      setStoplights(payload.data);
    } else {
      alert("CSV format not recognized.");
    }
  };

  // Group objects by ID for path visualization
  const objectPaths = useMemo(() => {
    const paths = {};
    objectDetections
      .filter(d => d.seconds >= timeFilter.min && d.seconds <= timeFilter.max)
      .forEach(detection => {
        if (!paths[detection.id]) {
          paths[detection.id] = [];
        }
        paths[detection.id].push(detection);
      });
    return paths;
  }, [objectDetections, timeFilter]);

  // Group stoplights by location
  const stoplightGroups = useMemo(() => {
    const groups = {};
    stoplights.forEach(light => {
      const key = `${light.lat.toFixed(5)}_${light.lon.toFixed(5)}`;
      if (!groups[key]) {
        groups[key] = {
          lat: light.lat,
          lon: light.lon,
          detections: []
        };
      }
      groups[key].detections.push(light);
    });
    return Object.values(groups);
  }, [stoplights]);

  const center = data?.speed_points?.[0]
    ? [data.speed_points[0][0], data.speed_points[0][1]]
    : alignedOutput[0]
      ? [alignedOutput[0].lat, alignedOutput[0].lon]
      : objectDetections[0]
        ? [objectDetections[0].lat, objectDetections[0].lon]
        : [44.9778, -93.2650];

  const icons = {
    pedestrian: new L.Icon({ iconUrl: 'pedestrian.png', iconSize: [25, 25] }),
    car: new L.Icon({ iconUrl: 'car.png', iconSize: [25, 25] }),
    bike: new L.Icon({ iconUrl: 'bike.png', iconSize: [25, 25] }),
    bus: new L.Icon({ iconUrl: 'bus.png', iconSize: [25, 25] }),
    truck: new L.Icon({ iconUrl: 'truck.png', iconSize: [25, 25] }),
    motorcycle: new L.Icon({ iconUrl: 'bike.png', iconSize: [20, 20] }),
    light: new L.Icon({ iconUrl: 'traffic_light.png', iconSize: [25, 25] })
  };

  const getObjectColor = (type) => {
    const colors = {
      car: '#FF0000',
      truck: '#8B0000',
      bus: '#FF8C00',
      motorcycle: '#FF1493',
      bicycle: '#00CED1',
      person: '#00FF00',
      pedestrian: '#00FF00',
      boat: '#1E90FF',
      'fire hydrant': '#FFD700',
      'traffic light': '#FFFF00',
      skateboard: '#FF69B4',
      skis: '#87CEEB',
      default: '#808080'
    };
    return colors[type.toLowerCase()] || colors.default;
  };

  return (
    <div style={{ fontFamily: 'MS Sans Serif, sans-serif' }}>
      <h2 style={{ 
        backgroundColor: '#000080', 
        color: '#fff', 
        padding: '5px',
        margin: '0',
        fontSize: '18px'
      }}>
        🗺️ Map View ({personal ? "My Data (CSV)" : "All Data (Server)"})
      </h2>

      <div style={{ 
        display: 'flex', 
        gap: '10px', 
        margin: '10px',
        flexWrap: 'wrap' 
      }}>
        <button 
          onClick={() => setPersonal(!personal)}
          style={{
            padding: '4px 12px',
            backgroundColor: '#c0c0c0',
            border: '2px outset #ddd',
            cursor: 'pointer',
            fontFamily: 'MS Sans Serif, sans-serif'
          }}
        >
          {personal ? "📊 Switch to All Data" : "💾 Switch to My CSV Data"}
        </button>

        {personal && objectDetections.length > 0 && (
          <>
            <button 
              onClick={() => setShowObjectPaths(!showObjectPaths)}
              style={{
                padding: '4px 12px',
                backgroundColor: showObjectPaths ? '#90EE90' : '#c0c0c0',
                border: '2px outset #ddd',
                cursor: 'pointer',
                fontFamily: 'MS Sans Serif, sans-serif'
              }}
            >
              {showObjectPaths ? "🚗 Hide Paths" : "🚗 Show Paths"}
            </button>

            <button 
              onClick={() => setShowStoplights(!showStoplights)}
              style={{
                padding: '4px 12px',
                backgroundColor: showStoplights ? '#90EE90' : '#c0c0c0',
                border: '2px outset #ddd',
                cursor: 'pointer',
                fontFamily: 'MS Sans Serif, sans-serif'
              }}
            >
              {showStoplights ? "🚦 Hide Lights" : "🚦 Show Lights"}
            </button>
          </>
        )}
      </div>

      {personal ? (
        <CsvLocalLoader onDataLoaded={handleCsvData} />
      ) : (
        <div style={{ margin: '10px', padding: '10px', backgroundColor: '#f0f0f0' }}>
          <UploadForm onUploadComplete={handleUploadComplete} />
        </div>
      )}

      {/* Object Summary Panel */}
      {personal && (objectCounts.length > 0 || objectDetections.length > 0) && (
        <div style={{ 
          margin: '10px', 
          padding: '10px', 
          backgroundColor: '#fffacd',
          border: '2px ridge #888',
          fontFamily: 'Courier, monospace',
          fontSize: '0.9em'
        }}>
          <h4 style={{ margin: '0 0 10px 0' }}>📊 Object Detection Summary</h4>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
            gap: '10px'
          }}>
            {objectCounts.length > 0 ? (
              objectCounts
                .sort((a, b) => b.framesSeen - a.framesSeen)
                .slice(0, 10)
                .map((obj, idx) => (
                  <div 
                    key={idx}
                    style={{ 
                      padding: '5px',
                      backgroundColor: selectedObject === obj.id ? '#b0e0e6' : '#fff',
                      border: '1px solid #999',
                      cursor: 'pointer'
                    }}
                    onClick={() => setSelectedObject(obj.id === selectedObject ? null : obj.id)}
                  >
                    <strong>{obj.type}</strong> #{obj.id}<br/>
                    Motion: {obj.motion}<br/>
                    Frames: {obj.framesSeen}
                  </div>
                ))
            ) : (
              // Create summary from objectDetections
              Object.entries(
                objectDetections.reduce((acc, det) => {
                  if (!acc[det.id]) {
                    acc[det.id] = {
                      id: det.id,
                      type: det.type,
                      count: 0,
                      speeds: [],
                      bearings: []
                    };
                  }
                  acc[det.id].count++;
                  if (det.speed) acc[det.id].speeds.push(det.speed);
                  if (det.bearing) acc[det.id].bearings.push(det.bearing);
                  return acc;
                }, {})
              )
                .map(([id, data]) => ({
                  id,
                  type: data.type,
                  framesSeen: data.count,
                  avgSpeed: data.speeds.length > 0 ? 
                    (data.speeds.reduce((a, b) => a + b, 0) / data.speeds.length).toFixed(1) : 
                    'N/A',
                  maxSpeed: data.speeds.length > 0 ? 
                    Math.max(...data.speeds).toFixed(1) : 
                    'N/A'
                }))
                .sort((a, b) => b.framesSeen - a.framesSeen)
                .slice(0, 15)
                .map((obj, idx) => (
                  <div 
                    key={idx}
                    style={{ 
                      padding: '5px',
                      backgroundColor: selectedObject === obj.id ? '#b0e0e6' : '#fff',
                      border: '1px solid #999',
                      cursor: 'pointer'
                    }}
                    onClick={() => setSelectedObject(obj.id === selectedObject ? null : obj.id)}
                  >
                    <strong style={{ color: getObjectColor(obj.type) }}>
                      {obj.type}
                    </strong> #{obj.id}<br/>
                    Frames: {obj.framesSeen}<br/>
                    Avg Speed: {obj.avgSpeed} mph<br/>
                    Max Speed: {obj.maxSpeed} mph
                  </div>
                ))
            )}
          </div>
        </div>
      )}

      {/* Time Filter for Object Detection */}
      {personal && objectDetections.length > 0 && (
        <div style={{ 
          margin: '10px', 
          padding: '10px', 
          backgroundColor: '#e0e0e0',
          border: '1px solid #999'
        }}>
          <label>Time Range (seconds): </label>
          <input 
            type="range" 
            min={Math.min(...objectDetections.map(d => d.seconds))}
            max={Math.max(...objectDetections.map(d => d.seconds))}
            value={timeFilter.max}
            onChange={(e) => setTimeFilter({ ...timeFilter, max: parseInt(e.target.value) })}
            style={{ width: '300px' }}
          />
          <span> {timeFilter.min}s - {timeFilter.max}s</span>
          <div style={{ marginTop: '5px', fontSize: '0.8em', color: '#666' }}>
            Showing {Object.keys(objectPaths).length} objects with 
            {' ' + Object.values(objectPaths).reduce((sum, positions) => sum + positions.length, 0)} total detections
          </div>
        </div>
      )}

      {loading && <p style={{ color: '#888', margin: '10px' }}>⏳ Loading data...</p>}
      {error && <p style={{ color: 'red', margin: '10px' }}>❌ {error}</p>}

      <MapContainer center={center} zoom={13} style={{ height: "600px", margin: '10px', border: '2px inset #999' }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
        />

        {/* Auto-fit for various data types */}
        {personal && (alignedOutput.length > 0 || mergedOutput.length > 0 || objectDetections.length > 0) && (
          <FitMapToAligned points={
            alignedOutput.length > 0 ? alignedOutput : 
            mergedOutput.length > 0 ? mergedOutput : 
            objectDetections
          } />
        )}

        {/* Server-based data (All Data mode) */}
        {!personal && data && (
          <>
            <HeatmapLayer
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

        {/* Object Detection Paths */}
        {personal && showObjectPaths && Object.entries(objectPaths).map(([objectId, positions]) => {
          if (selectedObject && objectId !== selectedObject) return null;
          const color = getObjectColor(positions[0].type);
          const coords = positions.map(p => [p.lat, p.lon]);
          
          // Only show objects that have valid coordinates
          const validCoords = coords.filter(coord => 
            !isNaN(coord[0]) && !isNaN(coord[1]) && coord[0] !== 0 && coord[1] !== 0
          );
          
          if (validCoords.length === 0) return null;
          
          return (
            <React.Fragment key={objectId}>
              <Polyline 
                positions={validCoords} 
                color={color} 
                weight={3} 
                opacity={0.7}
              />
              {/* Show bearing arrows at intervals */}
              {positions
                .filter((p, idx) => idx % 10 === 0 && p.bearing && p.lat && p.lon)
                .map((position, idx) => {
                  const arrowIcon = L.divIcon({
                    html: `<div style="
                      transform: rotate(${position.bearing}deg);
                      color: ${color};
                      font-size: 20px;
                      font-weight: bold;
                      text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
                    ">↑</div>`,
                    className: 'bearing-arrow',
                    iconSize: [20, 20],
                    iconAnchor: [10, 10]
                  });
                  
                  return (
                    <Marker
                      key={`bearing-${objectId}-${idx}`}
                      position={[position.lat, position.lon]}
                      icon={arrowIcon}
                    >
                      <Tooltip>
                        {position.type} #{objectId}<br/>
                        Speed: {position.speed.toFixed(1)} mph<br/>
                        Bearing: {position.bearing.toFixed(0)}°<br/>
                        Distance: {position.distance ? position.distance.toFixed(1) + 'm' : 'N/A'}<br/>
                        Frame: {position.frame}<br/>
                        Time: {position.time}
                      </Tooltip>
                    </Marker>
                  );
                })}
              {/* Show start and end markers */}
              <CircleMarker
                center={validCoords[0]}
                radius={8}
                fillColor={color}
                fillOpacity={0.8}
                color="#000"
                weight={1}
              >
                <Tooltip>
                  <strong>Start</strong><br/>
                  {positions[0].type} #{objectId}<br/>
                  Speed: {positions[0].speed.toFixed(1)} mph<br/>
                  Bearing: {positions[0].bearing ? positions[0].bearing.toFixed(0) + '°' : 'N/A'}<br/>
                  Time: {positions[0].time}
                </Tooltip>
              </CircleMarker>
              <CircleMarker
                center={validCoords[validCoords.length - 1]}
                radius={8}
                fillColor={color}
                fillOpacity={0.8}
                color="#000"
                weight={2}
              >
                <Tooltip>
                  <strong>End</strong><br/>
                  {positions[0].type} #{objectId}<br/>
                  Speed: {positions[positions.length - 1].speed.toFixed(1)} mph<br/>
                  Bearing: {positions[positions.length - 1].bearing ? positions[positions.length - 1].bearing.toFixed(0) + '°' : 'N/A'}<br/>
                  Time: {positions[positions.length - 1].time}
                </Tooltip>
              </CircleMarker>
            </React.Fragment>
          );
        })}

        {/* Stoplight Visualizations */}
        {personal && showStoplights && stoplightGroups.map((group, idx) => {
          // Check what values we actually have in the data
          const statusCounts = {};
          group.detections.forEach(d => {
            const status = d.status.toLowerCase();
            statusCounts[status] = (statusCounts[status] || 0) + 1;
          });
          
          const total = group.detections.length;
          
          // Try to identify red/yellow/green from the status values
          const redCount = statusCounts['red'] || statusCounts['stoplight_red'] || 0;
          const greenCount = statusCounts['green'] || statusCounts['stoplight_green'] || 0;
          const yellowCount = statusCounts['yellow'] || statusCounts['stoplight_yellow'] || 0;
          const unknownCount = total - redCount - greenCount - yellowCount;
          
          return (
            <Marker
              key={`stoplight-${idx}`}
              position={[group.lat, group.lon]}
              icon={icons['light']}
            >
              <Popup>
                <div style={{ fontFamily: 'Courier, monospace', fontSize: '0.9em' }}>
                  <strong>🚦 Traffic Light Analysis</strong><br/>
                  <hr style={{ margin: '5px 0' }}/>
                  Total observations: {total}<br/>
                  <div style={{ marginTop: '5px' }}>
                    {redCount > 0 && (
                      <div style={{ 
                        display: 'flex', 
                        alignItems: 'center',
                        marginBottom: '2px'
                      }}>
                        <div style={{ 
                          width: '15px', 
                          height: '15px', 
                          backgroundColor: 'red',
                          marginRight: '5px',
                          border: '1px solid #000',
                          borderRadius: '50%'
                        }}></div>
                        Red: {redCount} ({(redCount/total*100).toFixed(1)}%)
                      </div>
                    )}
                    {yellowCount > 0 && (
                      <div style={{ 
                        display: 'flex', 
                        alignItems: 'center',
                        marginBottom: '2px'
                      }}>
                        <div style={{ 
                          width: '15px', 
                          height: '15px', 
                          backgroundColor: 'yellow',
                          marginRight: '5px',
                          border: '1px solid #000',
                          borderRadius: '50%'
                        }}></div>
                        Yellow: {yellowCount} ({(yellowCount/total*100).toFixed(1)}%)
                      </div>
                    )}
                    {greenCount > 0 && (
                      <div style={{ 
                        display: 'flex', 
                        alignItems: 'center',
                        marginBottom: '2px'
                      }}>
                        <div style={{ 
                          width: '15px', 
                          height: '15px', 
                          backgroundColor: 'green',
                          marginRight: '5px',
                          border: '1px solid #000',
                          borderRadius: '50%'
                        }}></div>
                        Green: {greenCount} ({(greenCount/total*100).toFixed(1)}%)
                      </div>
                    )}
                    {unknownCount > 0 && (
                      <div style={{ 
                        display: 'flex', 
                        alignItems: 'center'
                      }}>
                        <div style={{ 
                          width: '15px', 
                          height: '15px', 
                          backgroundColor: '#999',
                          marginRight: '5px',
                          border: '1px solid #000',
                          borderRadius: '50%'
                        }}></div>
                        Detected: {unknownCount} ({(unknownCount/total*100).toFixed(1)}%)
                      </div>
                    )}
                  </div>
                  <hr style={{ margin: '5px 0' }}/>
                  <small>
                    Location: {group.lat.toFixed(5)}, {group.lon.toFixed(5)}<br/>
                    {Object.keys(statusCounts).length > 0 && (
                      <>Status types: {Object.keys(statusCounts).join(', ')}</>
                    )}
                  </small>
                </div>
              </Popup>
            </Marker>
          );
        })}

        {/* Frame Noise Heatmap */}
        {personal && frameNoise.length > 0 && (
          <>
            <HeatmapLayer
              points={(() => {
                const minNoise = Math.min(...frameNoise.map(p => p.noise));
                const maxNoise = Math.max(...frameNoise.map(p => p.noise));
                return frameNoise.map(p => ({
                  lat: p.lat,
                  lng: p.lon,
                  intensity: maxNoise !== minNoise
                    ? (p.noise - minNoise) / (maxNoise - minNoise)
                    : 0.5
                }));
              })()}
              latitudeExtractor={p => p.lat}
              longitudeExtractor={p => p.lng}
              intensityExtractor={p => p.intensity}
              gradient={{ 0: 'blue', 0.5: 'yellow', 1: 'red' }}
              radius={20}
              blur={15}
            />
          </>
        )}

        {/* Sound Points Heatmap */}
        {personal && soundPoints.length > 0 && (
          <>
            <FitMapToAligned points={soundPoints} />
        
            <HeatmapLayer
              points={(() => {
                const minNoise = Math.min(...soundPoints.map(p => p.noise));
                const maxNoise = Math.max(...soundPoints.map(p => p.noise));
                return soundPoints.map(p => ({
                  lat: p.lat,
                  lng: p.lon,
                  intensity: maxNoise !== minNoise
                    ? 1 - (p.noise - minNoise) / (maxNoise - minNoise)  // inverse scale
                    : 0.5
                }));
              })()}
              latitudeExtractor={p => p.lat}
              longitudeExtractor={p => p.lng}
              intensityExtractor={p => p.intensity}
              gradient={{ 0: 'green', 0.5: 'yellow', 1: 'red' }}
              radius={15}
              blur={12}
            />
        
            {soundPoints.map((p, idx) => (
              <Marker
                key={`noise-${idx}`}
                position={[p.lat, p.lon]}
                icon={L.divIcon({ className: 'invisible-icon' })}
              >
                <Tooltip direction="top" offset={[0, -10]} opacity={0.9}>
                  <div style={{ fontSize: '0.8em' }}>
                    <b>{p.index >= 0 ? "Index" : "Second"}:</b> {p.index}<br />
                    <b>Time:</b> {p.time}<br />
                    <b>Lat:</b> {p.lat.toFixed(5)}<br />
                    <b>Lon:</b> {p.lon.toFixed(5)}<br />
                    <b>Noise:</b> {p.noise.toFixed(2)} dB
                  </div>
                </Tooltip>
              </Marker>
            ))}
          </>
        )}

        {/* Aligned Output (CSV) */}
        {personal && alignedOutput.length > 0 && (
          <>
            <HeatmapLayer
              points={(() => {
                const maxSpeed = Math.max(...alignedOutput.map(d => d.speed));
                return alignedOutput.map(p => ({
                  lat: p.lat,
                  lng: p.lon,
                  intensity: maxSpeed > 0 ? p.speed / maxSpeed : 0
                }));
              })()}
              latitudeExtractor={p => p.lat}
              longitudeExtractor={p => p.lng}
              intensityExtractor={p => p.intensity}
              gradient={{ 0: 'green', 0.5: 'yellow', 1: 'red' }}
              radius={10}
              blur={10}
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

        {/* Merged Output (CSV) */}
        {personal && mergedOutput.length > 0 && (
          <>
            <HeatmapLayer
              points={(() => {
                const maxSpeed = Math.max(...mergedOutput.map(d => d.speed));
                return mergedOutput.map(p => ({
                  lat: p.lat,
                  lng: p.lon,
                  intensity: maxSpeed > 0 ? p.speed / maxSpeed : 0
                }));
              })()}
              latitudeExtractor={p => p.lat}
              longitudeExtractor={p => p.lng}
              intensityExtractor={p => p.intensity}
              gradient={{ 0: 'green', 0.5: 'yellow', 1: 'red' }}
              radius={10}
              blur={10}
            />
            {mergedOutput.map((point, idx) => (
              <Marker
                key={`merged-${idx}`}
                position={[point.lat, point.lon]}
                icon={L.divIcon({ className: 'invisible-icon' })}
              >
                <Tooltip direction="top" offset={[0, -10]} opacity={0.9}>
                  <div style={{ fontSize: '0.8em' }}>
                    <b>Second:</b> {point.second}<br />
                    <b>Time:</b> {point.gpx_time}<br />
                    <b>Lat:</b> {point.lat.toFixed(5)}<br />
                    <b>Lon:</b> {point.lon.toFixed(5)}<br />
                    <b>Speed:</b> {point.speed.toFixed(2)} mph
                  </div>
                </Tooltip>
              </Marker>
            ))}
          </>
        )}

        {/* Local Markers (generic CSV) */}
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