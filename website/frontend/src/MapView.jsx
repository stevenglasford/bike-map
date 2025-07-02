import React, { useState, useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Tooltip, Popup, Polyline, CircleMarker } from 'react-leaflet';
import HeatmapLayer from './HeatmapLayer';
import UploadForm from './UploadForm';
import CsvLocalLoader from './CsvLocalLoader';
import L from 'leaflet';
import { useMap } from 'react-leaflet';

// Component to capture map reference
function MapRef({ setMapInstance }) {
  const map = useMap();
  
  useEffect(() => {
    console.log('🗺️ Map reference captured:', map);
    setMapInstance(map);
  }, [map, setMapInstance]);
  
  return null;
}

// Auto-fit component with enhanced debugging
function FitMapToAligned({ points, trigger }) {
  const map = useMap();
  
  useEffect(() => {
    if (!points || points.length === 0) {
      console.log('🗺️ FitMapToAligned: No points provided');
      return;
    }

    console.log('🗺️ FitMapToAligned: Processing', points.length, 'points');
    console.log('🗺️ First few points:', points.slice(0, 3));

    // Extract valid coordinates, handling different data structures
    const validBounds = [];
    
    points.forEach((point, index) => {
      let lat, lon;
      
      // Handle different data structures
      if (typeof point.lat === 'number' && typeof point.lon === 'number') {
        lat = point.lat;
        lon = point.lon;
      } else if (typeof point.latitude === 'number' && typeof point.longitude === 'number') {
        lat = point.latitude;
        lon = point.longitude;
      } else if (Array.isArray(point) && point.length >= 2) {
        lat = point[0];
        lon = point[1];
      } else {
        console.warn(`🗺️ Invalid point at index ${index}:`, point);
        return;
      }

      // Validate coordinates
      if (isNaN(lat) || isNaN(lon) || lat === 0 || lon === 0 || 
          lat < -90 || lat > 90 || lon < -180 || lon > 180) {
        console.warn(`🗺️ Invalid coordinates at index ${index}: lat=${lat}, lon=${lon}`);
        return;
      }

      validBounds.push([lat, lon]);
    });

    console.log('🗺️ Valid bounds found:', validBounds.length);

    if (validBounds.length > 0) {
      try {
        console.log('🗺️ Fitting map to bounds:', validBounds.slice(0, 3), '...');
        
        // Calculate bounds for debugging
        const lats = validBounds.map(b => b[0]);
        const lons = validBounds.map(b => b[1]);
        console.log('🗺️ Lat range:', Math.min(...lats), 'to', Math.max(...lats));
        console.log('🗺️ Lon range:', Math.min(...lons), 'to', Math.max(...lons));
        
        map.fitBounds(validBounds, { 
          padding: [30, 30],
          maxZoom: 16
        });
        console.log('🗺️ Map fitted successfully');
      } catch (error) {
        console.error('🗺️ Error fitting map bounds:', error);
      }
    } else {
      console.warn('🗺️ No valid coordinates found to fit map');
    }
  }, [points, map, trigger]);

  return null;
}

function MapView() {
  // State variables
  const [data, setData] = useState(null);
  const [personal, setPersonal] = useState(false);
  const [localMarkers, setLocalMarkers] = useState([]);
  const [alignedOutput, setAlignedOutput] = useState([]);
  const [mergedOutput, setMergedOutput] = useState([]);
  const [soundPoints, setSoundPoints] = useState([]);
  const [objectDetections, setObjectDetections] = useState([]);
  const [objectCounts, setObjectCounts] = useState([]);
  const [stoplights, setStoplights] = useState([]);
  const [validStoplights, setValidStoplights] = useState([]); // NEW: Filtered valid coordinates
  const [frameNoise, setFrameNoise] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedObject, setSelectedObject] = useState(null);
  const [showObjectPaths, setShowObjectPaths] = useState(true);
  const [showStoplights, setShowStoplights] = useState(true);
  const [timeFilter, setTimeFilter] = useState({ min: 0, max: Infinity });
  
  // Map instance for external controls
  const [mapInstance, setMapInstance] = useState(null);
  const [autoFitTrigger, setAutoFitTrigger] = useState(0);

  // NEW: Filter valid coordinates whenever stoplights change
  useEffect(() => {
    if (stoplights.length > 0) {
      console.log('🌍 Filtering coordinates for', stoplights.length, 'stoplights...');
      
      const valid = stoplights.filter(light => {
        const latValid = light.lat >= -90 && light.lat <= 90 && !isNaN(light.lat);
        const lonValid = light.lon >= -180 && light.lon <= 180 && !isNaN(light.lon);
        return latValid && lonValid;
      });
      
      const invalid = stoplights.filter(light => {
        const latValid = light.lat >= -90 && light.lat <= 90 && !isNaN(light.lat);
        const lonValid = light.lon >= -180 && light.lon <= 180 && !isNaN(light.lon);
        return !(latValid && lonValid);
      });
      
      console.log(`🌍 Coordinate filtering results:`);
      console.log(`🌍 Total: ${stoplights.length}`);
      console.log(`🌍 Valid: ${valid.length}`);
      console.log(`🌍 Invalid: ${invalid.length}`);
      
      if (valid.length > 0) {
        const lats = valid.map(s => s.lat);
        const lons = valid.map(s => s.lon);
        console.log(`🌍 Valid range - Lat: ${Math.min(...lats)} to ${Math.max(...lats)}`);
        console.log(`🌍 Valid range - Lon: ${Math.min(...lons)} to ${Math.max(...lons)}`);
      }
      
      if (invalid.length > 0) {
        console.log('🌍 Sample invalid coordinates:', invalid.slice(0, 5).map(s => ({lat: s.lat, lon: s.lon})));
      }
      
      setValidStoplights(valid);
      
      if (valid.length !== stoplights.length) {
        alert(`⚠️ Coordinate Filter Results:\n\nTotal stoplights: ${stoplights.length}\nValid coordinates: ${valid.length}\nInvalid coordinates: ${invalid.length}\n\nOnly showing valid coordinates on map.`);
      }
      
      // Trigger auto-fit for valid coordinates
      if (valid.length > 0) {
        setAutoFitTrigger(prev => prev + 1);
      }
    } else {
      setValidStoplights([]);
    }
  }, [stoplights]);

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
      if (personal) {
        setLocalMarkers([]);
        setAlignedOutput([]);
        setObjectDetections([]);
        setObjectCounts([]);
        setStoplights([]);
        setValidStoplights([]);
        setSoundPoints([]);
        setFrameNoise([]);
      }
    }
  };

  useEffect(() => {
    fetchData();
  }, [personal]);

  const handleUploadComplete = () => {
    setTimeout(() => fetchData(), 10000);
  };

  // Enhanced debug function
  const debugCsvData = (data, type) => {
    console.log('🔍 === CSV DEBUG START ===');
    console.log('🔍 Type:', type);
    console.log('🔍 Total rows:', data.length);
    console.log('🔍 First 3 rows:', data.slice(0, 3));
    
    if (data.length > 0) {
      const firstRow = data[0];
      console.log('🔍 Column names:', Object.keys(firstRow));
      
      // Check for coordinate columns
      const possibleLatCols = ['lat', 'latitude', 'lat_deg', 'y'];
      const possibleLonCols = ['lon', 'longitude', 'lng', 'lon_deg', 'x'];
      
      const latCol = possibleLatCols.find(col => col in firstRow);
      const lonCol = possibleLonCols.find(col => col in firstRow);
      
      console.log('🔍 Detected lat column:', latCol, '=', firstRow[latCol]);
      console.log('🔍 Detected lon column:', lonCol, '=', firstRow[lonCol]);
      
      if (latCol && lonCol) {
        const validCoords = data.filter(row => {
          const lat = parseFloat(row[latCol]);
          const lon = parseFloat(row[lonCol]);
          return !isNaN(lat) && !isNaN(lon) && lat !== 0 && lon !== 0;
        });
        console.log('🔍 Valid coordinate rows:', validCoords.length);
        
        if (validCoords.length > 0) {
          const lats = validCoords.map(row => parseFloat(row[latCol]));
          const lons = validCoords.map(row => parseFloat(row[lonCol]));
          console.log('🔍 Lat range:', Math.min(...lats), 'to', Math.max(...lats));
          console.log('🔍 Lon range:', Math.min(...lons), 'to', Math.max(...lons));
        }
      }
      
      // Special debugging for stoplights
      if (type === 'stoplights') {
        console.log('🚦 STOPLIGHT SPECIFIC DEBUG:');
        console.log('🚦 object_class values:', [...new Set(data.map(row => row.object_class))]);
        console.log('🚦 stoplight_color values:', [...new Set(data.map(row => row.stoplight_color))]);
        console.log('🚦 Sample coordinates:', data.slice(0, 5).map(row => ({lat: row.lat, lon: row.lon})));
      }
    }
    console.log('🔍 === CSV DEBUG END ===');
  };

  const handleCsvData = (payload) => {
    console.log('📥 CSV Data received:', payload.type, 'with', payload.data.length, 'items');
    
    // Add debugging
    debugCsvData(payload.data, payload.type);
    
    if (payload.type === "aligned_output") {
      alert(`Loaded ${payload.data.length} aligned points`);
      setAlignedOutput(payload.data);
      setMergedOutput([]);
      setAutoFitTrigger(prev => prev + 1);
    } else if (payload.type === "merged_output") {
      alert(`Loaded ${payload.data.length} merged points`);
      setMergedOutput(payload.data);
      setAlignedOutput([]);
      setAutoFitTrigger(prev => prev + 1);
    } else if (payload.type === "markers") {
      alert(`Loaded ${payload.data.length} markers`);
      setLocalMarkers(payload.data);
    } else if (payload.type === "sound_heatmap") {
      alert(`Loaded ${payload.data.length} sound points`);
      setSoundPoints(payload.data);
      setAutoFitTrigger(prev => prev + 1);
    } else if (payload.type === "frame_noise") {
      alert(`Loaded ${payload.data.length} frame noise points`);
      setFrameNoise(payload.data);
      setAutoFitTrigger(prev => prev + 1);
    } else if (payload.type === "object_detection") {
      alert(`Loaded ${payload.data.length} detected objects`);
      setObjectDetections(payload.data);
      // Calculate time range
      const times = payload.data.map(d => d.seconds);
      setTimeFilter({ min: Math.min(...times), max: Math.max(...times) });
      setAutoFitTrigger(prev => prev + 1);
    } else if (payload.type === "object_counts") {
      alert(`Loaded object count summary for ${payload.data.length} objects`);
      setObjectCounts(payload.data);
    } else if (payload.type === "stoplights") {
      alert(`🚦 Loaded ${payload.data.length} stoplight detections!`);
      setStoplights(payload.data);
      
      // Debug the stoplight data specifically
      console.log('🚦 STOPLIGHT DATA SET:', payload.data.slice(0, 5));
      console.log('🚦 Setting autoFitTrigger to trigger map fit');
      
    } else {
      alert(`❌ CSV format not recognized. Detected type: ${payload.type}`);
      console.log('❌ Unrecognized CSV type:', payload.type);
    }
  };

  const fitMapToData = (points, label) => {
  if (!mapInstance) {
    alert('❌ Map not ready yet');
    console.log('❌ mapInstance is null');
    return;
  }

  if (!points || points.length === 0) {
    alert(`❌ No ${label} data to fit map to`);
    console.log(`❌ points is:`, points);
    return;
  }

  console.log(`🎯 === FIT MAP DEBUG START ===`);
  console.log(`🎯 Fitting map to ${points.length} ${label} points`);
  console.log(`🎯 First 5 data points:`, points.slice(0, 5));
  console.log(`🎯 Sample point structure:`, points[0]);
  console.log(`🎯 Sample point keys:`, Object.keys(points[0] || {}));
  
  const validBounds = [];
  const invalidPoints = [];
  
  points.forEach((point, index) => {
    let lat, lon;
    
    // Debug the point structure
    if (index < 5) {
      console.log(`🎯 Point ${index} analysis:`);
      console.log(`  - Raw point:`, point);
      console.log(`  - point.lat:`, point.lat, `(type: ${typeof point.lat})`);
      console.log(`  - point.lon:`, point.lon, `(type: ${typeof point.lon})`);
      console.log(`  - point.latitude:`, point.latitude, `(type: ${typeof point.latitude})`);
      console.log(`  - point.longitude:`, point.longitude, `(type: ${typeof point.longitude})`);
    }
    
    // Extract coordinates with enhanced debugging
    if (typeof point.lat === 'number' && typeof point.lon === 'number') {
      lat = point.lat;
      lon = point.lon;
      if (index < 5) console.log(`  - Using point.lat/lon (numbers): ${lat}, ${lon}`);
    } else if (typeof point.lat === 'string' && typeof point.lon === 'string') {
      lat = parseFloat(point.lat);
      lon = parseFloat(point.lon);
      if (index < 5) console.log(`  - Using point.lat/lon (parsed strings): ${lat}, ${lon}`);
    } else if (typeof point.latitude === 'number' && typeof point.longitude === 'number') {
      lat = point.latitude;
      lon = point.longitude;
      if (index < 5) console.log(`  - Using point.latitude/longitude (numbers): ${lat}, ${lon}`);
    } else if (Array.isArray(point) && point.length >= 2) {
      lat = point[0];
      lon = point[1];
      if (index < 5) console.log(`  - Using array format: ${lat}, ${lon}`);
    } else {
      if (index < 5) console.warn(`  - ❌ Invalid point structure at ${index}:`, point);
      invalidPoints.push({ index, point, reason: 'Invalid structure' });
      return;
    }

    // Validate coordinates with detailed logging
    const latNum = parseFloat(lat);
    const lonNum = parseFloat(lon);
    
    if (index < 5) {
      console.log(`  - Parsed coordinates: lat=${latNum}, lon=${lonNum}`);
      console.log(`  - isNaN check: lat=${isNaN(latNum)}, lon=${isNaN(lonNum)}`);
      console.log(`  - Zero check: lat=${latNum === 0}, lon=${lonNum === 0}`);
      console.log(`  - Range check: lat=${latNum >= -90 && latNum <= 90}, lon=${lonNum >= -180 && lonNum <= 180}`);
    }

    if (isNaN(latNum) || isNaN(lonNum)) {
      if (index < 5) console.warn(`  - ❌ NaN coordinates at ${index}: lat=${latNum}, lon=${lonNum}`);
      invalidPoints.push({ index, point, reason: 'NaN coordinates', lat: latNum, lon: lonNum });
      return;
    }
    
    if (latNum === 0 || lonNum === 0) {
      if (index < 5) console.warn(`  - ❌ Zero coordinates at ${index}: lat=${latNum}, lon=${lonNum}`);
      invalidPoints.push({ index, point, reason: 'Zero coordinates', lat: latNum, lon: lonNum });
      return;
    }
    
    if (latNum < -90 || latNum > 90 || lonNum < -180 || lonNum > 180) {
      if (index < 5) console.warn(`  - ❌ Out of range coordinates at ${index}: lat=${latNum}, lon=${lonNum}`);
      invalidPoints.push({ index, point, reason: 'Out of range', lat: latNum, lon: lonNum });
      return;
    }

    // If we get here, coordinates are valid
    validBounds.push([latNum, lonNum]);
    if (index < 5) console.log(`  - ✅ Valid coordinates: [${latNum}, ${lonNum}]`);
  });

  console.log(`🎯 === VALIDATION RESULTS ===`);
  console.log(`🎯 Total points processed: ${points.length}`);
  console.log(`🎯 Valid coordinates found: ${validBounds.length}`);
  console.log(`🎯 Invalid points: ${invalidPoints.length}`);
  
  if (invalidPoints.length > 0) {
    console.log(`🎯 Sample invalid points:`, invalidPoints.slice(0, 5));
    console.log(`🎯 Invalid point reasons:`, invalidPoints.map(p => p.reason).slice(0, 10));
  }
  
  if (validBounds.length > 0) {
    console.log(`🎯 First 5 valid bounds:`, validBounds.slice(0, 5));
    
    try {
      // Log bounds for debugging
      const lats = validBounds.map(b => b[0]);
      const lons = validBounds.map(b => b[1]);
      console.log(`🎯 ${label} bounds - Lat: ${Math.min(...lats)} to ${Math.max(...lats)}, Lon: ${Math.min(...lons)} to ${Math.max(...lons)}`);
      
      mapInstance.fitBounds(validBounds, { 
        padding: [30, 30],
        maxZoom: 16
      });
      alert(`✅ Map fitted to ${validBounds.length} valid ${label} coordinates\n\nValid: ${validBounds.length}\nInvalid: ${invalidPoints.length}`);
    } catch (error) {
      console.error('🎯 Error fitting map:', error);
      alert(`❌ Error fitting map: ${error.message}`);
    }
  } else {
    console.log(`🎯 === FAILURE ANALYSIS ===`);
    console.log(`🎯 No valid coordinates found!`);
    console.log(`🎯 Data type analysis:`, {
      isArray: Array.isArray(points),
      length: points.length,
      firstItemType: typeof points[0],
      firstItemKeys: points[0] ? Object.keys(points[0]) : 'none'
    });
    
    alert(`❌ No valid coordinates found in ${label} data!
    
Debug Info:
- Total points: ${points.length}
- Valid bounds: ${validBounds.length}  
- Invalid points: ${invalidPoints.length}

Top invalid reasons:
${invalidPoints.slice(0, 3).map(p => `- ${p.reason}: lat=${p.lat}, lon=${p.lon}`).join('\n')}

Check console for detailed analysis.`);
  }
  
  console.log(`🎯 === FIT MAP DEBUG END ===`);
};

  // NEW: Coordinate inspection function
  const inspectCoordinates = () => {
    console.log('🔍 === COORDINATE INSPECTION START ===');
    console.log('🔍 Total stoplights:', stoplights.length);
    console.log('🔍 First 10 stoplights:', stoplights.slice(0, 10));
    
    if (stoplights.length > 0) {
      // Check coordinate ranges
      const lats = stoplights.map(s => s.lat);
      const lons = stoplights.map(s => s.lon);
      console.log('🔍 All lat range:', Math.min(...lats), 'to', Math.max(...lats));
      console.log('🔍 All lon range:', Math.min(...lons), 'to', Math.max(...lons));
      
      // Count invalid coordinates
      const invalidCoords = stoplights.filter(s => 
        s.lat < -90 || s.lat > 90 || s.lon < -180 || s.lon > 180 || isNaN(s.lat) || isNaN(s.lon)
      );
      console.log('🔍 Invalid coordinates count:', invalidCoords.length);
      console.log('🔍 Sample invalid coordinates:', invalidCoords.slice(0, 5));
      
      // Check for valid coordinates
      const validCoords = stoplights.filter(s => 
        s.lat >= -90 && s.lat <= 90 && s.lon >= -180 && s.lon <= 180 && !isNaN(s.lat) && !isNaN(s.lon)
      );
      console.log('🔍 Valid coordinates count:', validCoords.length);
      console.log('🔍 Sample valid coordinates:', validCoords.slice(0, 5));
      
      if (validCoords.length > 0) {
        const validLats = validCoords.map(s => s.lat);
        const validLons = validCoords.map(s => s.lon);
        console.log('🔍 Valid lat range:', Math.min(...validLats), 'to', Math.max(...validLats));
        console.log('🔍 Valid lon range:', Math.min(...validLons), 'to', Math.max(...validLons));
      }
      
      // Check for potential data column issues
      console.log('🔍 Data structure check:');
      console.log('🔍 Sample data keys:', Object.keys(stoplights[0] || {}));
      console.log('🔍 Sample lat values:', stoplights.slice(0, 5).map(s => s.lat));
      console.log('🔍 Sample lon values:', stoplights.slice(0, 5).map(s => s.lon));
      
      console.log('🔍 === COORDINATE INSPECTION END ===');
      
      alert(`🔍 Coordinate Inspection Results:
      
Total Stoplights: ${stoplights.length}
Valid Coordinates: ${validCoords.length}
Invalid Coordinates: ${invalidCoords.length}

${validCoords.length > 0 ? 
  `Valid Range:
  Lat: ${Math.min(...validCoords.map(s => s.lat)).toFixed(4)} to ${Math.max(...validCoords.map(s => s.lat)).toFixed(4)}
  Lon: ${Math.min(...validCoords.map(s => s.lon)).toFixed(4)} to ${Math.max(...validCoords.map(s => s.lon)).toFixed(4)}` : 
  'No valid coordinates found!'
}

Check console for detailed analysis!`);
    } else {
      alert('No stoplight data to inspect');
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

  // Group stoplights by location - use VALID stoplights only
  const stoplightGroups = useMemo(() => {
    const groups = {};
    validStoplights.forEach(light => {
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
    console.log('🚦 Valid stoplight groups created:', Object.keys(groups).length);
    return Object.values(groups);
  }, [validStoplights]);

  // Determine map center - use valid stoplights
  const center = data?.speed_points?.[0]
    ? [data.speed_points[0][0], data.speed_points[0][1]]
    : alignedOutput[0]
      ? [alignedOutput[0].lat, alignedOutput[0].lon]
      : objectDetections[0]
        ? [objectDetections[0].lat, objectDetections[0].lon]
        : validStoplights[0]
          ? [validStoplights[0].lat, validStoplights[0].lon]
          : [44.9778, -93.2650];

  console.log('🗺️ Map center:', center);

  // Fixed icons using CSS instead of missing images
  const icons = {
    pedestrian: new L.DivIcon({
      className: 'custom-div-icon',
      html: '<div style="background-color: #00FF00; width: 20px; height: 20px; border-radius: 50%; border: 2px solid #000; display: flex; align-items: center; justify-content: center; font-size: 12px;">🚶</div>',
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    }),
    car: new L.DivIcon({
      className: 'custom-div-icon',
      html: '<div style="background-color: #FF0000; width: 20px; height: 20px; border-radius: 50%; border: 2px solid #000; display: flex; align-items: center; justify-content: center; font-size: 12px;">🚗</div>',
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    }),
    bike: new L.DivIcon({
      className: 'custom-div-icon',
      html: '<div style="background-color: #00CED1; width: 20px; height: 20px; border-radius: 50%; border: 2px solid #000; display: flex; align-items: center; justify-content: center; font-size: 12px;">🚲</div>',
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    }),
    bus: new L.DivIcon({
      className: 'custom-div-icon',
      html: '<div style="background-color: #FF8C00; width: 20px; height: 20px; border-radius: 50%; border: 2px solid #000; display: flex; align-items: center; justify-content: center; font-size: 12px;">🚌</div>',
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    }),
    truck: new L.DivIcon({
      className: 'custom-div-icon',
      html: '<div style="background-color: #8B0000; width: 20px; height: 20px; border-radius: 50%; border: 2px solid #000; display: flex; align-items: center; justify-content: center; font-size: 12px;">🚛</div>',
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    }),
    motorcycle: new L.DivIcon({
      className: 'custom-div-icon',
      html: '<div style="background-color: #FF1493; width: 20px; height: 20px; border-radius: 50%; border: 2px solid #000; display: flex; align-items: center; justify-content: center; font-size: 12px;">🏍️</div>',
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    }),
    light: new L.DivIcon({
      className: 'custom-div-icon',
      html: '<div style="background-color: #FFFF00; width: 25px; height: 25px; border-radius: 50%; border: 3px solid #000; display: flex; align-items: center; justify-content: center; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.5);">🚦</div>',
      iconSize: [30, 30],
      iconAnchor: [15, 15]
    })
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

  // Determine what data should be auto-fitted - prioritize valid stoplights
  const autoFitData = alignedOutput.length > 0 ? alignedOutput : 
                     mergedOutput.length > 0 ? mergedOutput : 
                     objectDetections.length > 0 ? objectDetections :
                     validStoplights.length > 0 ? validStoplights :
                     soundPoints.length > 0 ? soundPoints :
                     frameNoise.length > 0 ? frameNoise : [];

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

        {/* EXTERNAL FIT BUTTONS - PROMINENT STYLING */}
        {personal && (
          <>
            {alignedOutput.length > 0 && (
              <button 
                onClick={() => fitMapToData(alignedOutput, 'Aligned Data')}
                style={{
                  padding: '6px 15px',
                  backgroundColor: '#87CEEB',
                  border: '2px outset #ddd',
                  cursor: 'pointer',
                  fontFamily: 'MS Sans Serif, sans-serif',
                  fontWeight: 'bold'
                }}
              >
                🗺️ FIT → Aligned Data ({alignedOutput.length})
              </button>
            )}
            
            {mergedOutput.length > 0 && (
              <button 
                onClick={() => fitMapToData(mergedOutput, 'Merged Data')}
                style={{
                  padding: '6px 15px',
                  backgroundColor: '#87CEEB',
                  border: '2px outset #ddd',
                  cursor: 'pointer',
                  fontFamily: 'MS Sans Serif, sans-serif',
                  fontWeight: 'bold'
                }}
              >
                🗺️ FIT → Merged Data ({mergedOutput.length})
              </button>
            )}
            
            {objectDetections.length > 0 && (
              <button 
                onClick={() => fitMapToData(objectDetections, 'Objects')}
                style={{
                  padding: '6px 15px',
                  backgroundColor: '#98FB98',
                  border: '2px outset #ddd',
                  cursor: 'pointer',
                  fontFamily: 'MS Sans Serif, sans-serif',
                  fontWeight: 'bold'
                }}
              >
                🚗 FIT → Objects ({objectDetections.length})
              </button>
            )}
            
            {/* NEW: Use validStoplights instead of stoplights */}
            {validStoplights.length > 0 && (
              <button 
                onClick={() => fitMapToData(validStoplights, 'Valid Stoplights')}
                style={{
                  padding: '8px 20px',
                  backgroundColor: '#FF6B6B',
                  border: '3px outset #ddd',
                  cursor: 'pointer',
                  fontFamily: 'MS Sans Serif, sans-serif',
                  fontWeight: 'bold',
                  fontSize: '14px',
                  color: 'white',
                  textShadow: '1px 1px 1px #000'
                }}
              >
                🚦 FIT MAP → VALID STOPLIGHTS ({validStoplights.length})
              </button>
            )}

            {/* NEW: Coordinate inspection button */}
            {stoplights.length > 0 && (
              <button 
                onClick={inspectCoordinates}
                style={{
                  padding: '6px 15px',
                  backgroundColor: '#FFD700',
                  border: '2px outset #ddd',
                  cursor: 'pointer',
                  fontFamily: 'MS Sans Serif, sans-serif',
                  fontWeight: 'bold'
                }}
              >
                🔍 INSPECT COORDINATES ({stoplights.length})
              </button>
            )}

            {/* Convert markers to stoplights (backup) */}
            {localMarkers.length > 0 && (
              <button 
                onClick={() => {
                  console.log('🔄 Converting markers to stoplights...');
                  console.log('🔄 First marker sample:', localMarkers[0]);
                  
                  // Check if markers look like stoplights
                  const hasTrafficLights = localMarkers.some(m => 
                    m.object_class === 'traffic light' || 
                    m.type === 'traffic light' ||
                    'stoplight_color' in m ||
                    'object_class' in m
                  );
                  
                  if (hasTrafficLights) {
                    // Convert markers to stoplights
                    setStoplights(localMarkers);
                    setLocalMarkers([]);
                    alert(`✅ Converted ${localMarkers.length} markers to stoplights!`);
                  } else {
                    alert('❌ These markers do not appear to be stoplights');
                    console.log('🔄 Marker structure:', Object.keys(localMarkers[0] || {}));
                  }
                }}
                style={{
                  padding: '6px 15px',
                  backgroundColor: '#FFA500',
                  border: '2px outset #ddd',
                  cursor: 'pointer',
                  fontFamily: 'MS Sans Serif, sans-serif',
                  fontWeight: 'bold'
                }}
              >
                🔄 Convert Markers→Stoplights ({localMarkers.length})
              </button>
            )}
            
            {soundPoints.length > 0 && (
              <button 
                onClick={() => fitMapToData(soundPoints, 'Sound Data')}
                style={{
                  padding: '6px 15px',
                  backgroundColor: '#DDA0DD',
                  border: '2px outset #ddd',
                  cursor: 'pointer',
                  fontFamily: 'MS Sans Serif, sans-serif',
                  fontWeight: 'bold'
                }}
              >
                🔊 FIT → Sound Data ({soundPoints.length})
              </button>
            )}

            {frameNoise.length > 0 && (
              <button 
                onClick={() => fitMapToData(frameNoise, 'Frame Noise')}
                style={{
                  padding: '6px 15px',
                  backgroundColor: '#F0E68C',
                  border: '2px outset #ddd',
                  cursor: 'pointer',
                  fontFamily: 'MS Sans Serif, sans-serif',
                  fontWeight: 'bold'
                }}
              >
                📊 FIT → Frame Noise ({frameNoise.length})
              </button>
            )}
          </>
        )}

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

      {/* Enhanced Debug Panel */}
      {personal && (
        <div style={{ 
          margin: '10px', 
          padding: '10px', 
          backgroundColor: '#ffe4e1',
          border: '2px ridge #999',
          fontFamily: 'Courier, monospace',
          fontSize: '0.85em'
        }}>
          <strong>🐛 Debug Info:</strong><br/>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '10px', marginTop: '5px' }}>
            <div>
              <strong>Data Counts:</strong><br/>
              Aligned Output: <span style={{color: alignedOutput.length > 0 ? 'green' : 'gray'}}>{alignedOutput.length}</span><br/>
              Merged Output: <span style={{color: mergedOutput.length > 0 ? 'green' : 'gray'}}>{mergedOutput.length}</span><br/>
              Object Detections: <span style={{color: objectDetections.length > 0 ? 'green' : 'gray'}}>{objectDetections.length}</span><br/>
              <strong style={{color: stoplights.length > 0 ? 'red' : 'gray'}}>
                Stoplights: {stoplights.length} (Valid: <span style={{color: validStoplights.length > 0 ? 'green' : 'red'}}>{validStoplights.length}</span>)
              </strong><br/>
              Sound Points: <span style={{color: soundPoints.length > 0 ? 'green' : 'gray'}}>{soundPoints.length}</span><br/>
              Local Markers: <span style={{color: localMarkers.length > 0 ? 'green' : 'gray'}}>{localMarkers.length}</span><br/>
              Frame Noise: <span style={{color: frameNoise.length > 0 ? 'green' : 'gray'}}>{frameNoise.length}</span>
            </div>
            <div>
              <strong>Map Status:</strong><br/>
              Map Instance: <span style={{color: mapInstance ? 'green' : 'red'}}>{mapInstance ? 'Ready' : 'Not Ready'}</span><br/>
              Auto-fit Trigger: {autoFitTrigger}<br/>
              Map Center: [{center[0].toFixed(4)}, {center[1].toFixed(4)}]<br/>
              {validStoplights.length > 0 && (
                <>
                  <strong style={{color: 'green'}}>Valid Stoplight Sample:</strong><br/>
                  Lat: {validStoplights[0]?.lat.toFixed(4)}, Lon: {validStoplights[0]?.lon.toFixed(4)}
                </>
              )}
              {stoplights.length > 0 && validStoplights.length === 0 && (
                <>
                  <strong style={{color: 'red'}}>Invalid Coordinates:</strong><br/>
                  Lat: {stoplights[0]?.lat}, Lon: {stoplights[0]?.lon}
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {loading && <p style={{ color: '#888', margin: '10px' }}>⏳ Loading data...</p>}
      {error && <p style={{ color: 'red', margin: '10px' }}>❌ {error}</p>}

      <MapContainer 
        center={center} 
        zoom={13} 
        style={{ height: "600px", margin: '10px', border: '2px inset #999' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
        />

        {/* Map Reference Capture */}
        <MapRef setMapInstance={setMapInstance} />

        {/* Auto-fit for various data types */}
        {personal && autoFitData.length > 0 && (
          <FitMapToAligned points={autoFitData} trigger={autoFitTrigger} />
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

        {/* Stoplight Visualizations - Use VALID stoplights only */}
        {personal && showStoplights && stoplightGroups.map((group, idx) => {
          console.log(`🚦 Rendering valid stoplight group ${idx}:`, group);
          
          // Check what values we actually have in the data
          const statusCounts = {};
          group.detections.forEach(d => {
            const status = d.stoplight_color || d.status || 'detected';
            const statusLower = status.toLowerCase();
            statusCounts[statusLower] = (statusCounts[statusLower] || 0) + 1;
          });
          
          const total = group.detections.length;
          
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
                  Location: {group.lat.toFixed(5)}, {group.lon.toFixed(5)}<br/>
                  <hr style={{ margin: '5px 0' }}/>
                  <strong>Status breakdown:</strong><br/>
                  {Object.entries(statusCounts).map(([status, count]) => (
                    <div key={status} style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>{status}:</span>
                      <span>{count} ({(count/total*100).toFixed(1)}%)</span>
                    </div>
                  ))}
                  <hr style={{ margin: '5px 0' }}/>
                  <small>
                    Valid Group #{idx + 1} of {stoplightGroups.length}
                  </small>
                </div>
              </Popup>
            </Marker>
          );
        })}

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