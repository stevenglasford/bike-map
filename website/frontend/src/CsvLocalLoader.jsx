import React, { useState } from 'react';

function CsvLocalLoader({ onDataLoaded }) {
  const [error, setError] = useState(null);
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState(null);

  const handleFileChange = e => {
    const selected = e.target.files[0];
    if (!selected || !selected.name.toLowerCase().endsWith('.csv')) {
      setError("Only .csv files are supported.");
      return;
    }

    setFile(selected);
    setFileName(selected.name);
    setError(null);
  };

  const loadStoplights = () => {
    if (!file) return;

    const reader = new FileReader();
    reader.onload = e => {
      try {
        const text = e.target.result;
        const lines = text.split('\n').filter(line => line.trim() !== '');
        
        console.log('🚦 === CORRECTED STOPLIGHT PARSER ===');
        console.log('🚦 Total lines:', lines.length);
        console.log('🚦 Header:', lines[0]);
        console.log('🚦 Sample data line:', lines[1]);
        
        const stoplights = [];
        
        for (let i = 1; i < lines.length && i < 1001; i++) {
          const line = lines[i].trim();
          if (!line) continue;
          
          try {
            console.log(`🚦 Processing line ${i}: ${line}`);
            
            // Your format: frame_second,object_class,"[bbox]",confidence,lat,lon,gps_time,video_type,stoplight_color
            // We need to handle the quoted bbox properly
            
            // Find the quoted section (bbox)
            const firstQuote = line.indexOf('"');
            const lastQuote = line.lastIndexOf('"');
            
            if (firstQuote === -1 || lastQuote === -1 || firstQuote === lastQuote) {
              console.warn(`🚦 Line ${i}: No proper quoted section found`);
              continue;
            }
            
            // Split into: before_bbox + bbox + after_bbox
            const beforeBbox = line.substring(0, firstQuote - 1); // -1 to remove comma
            const bbox = line.substring(firstQuote + 1, lastQuote); // Remove quotes
            const afterBbox = line.substring(lastQuote + 2); // +2 to skip quote and comma
            
            console.log(`🚦 Line ${i} before bbox: "${beforeBbox}"`);
            console.log(`🚦 Line ${i} bbox: "${bbox}"`);
            console.log(`🚦 Line ${i} after bbox: "${afterBbox}"`);
            
            // Parse before bbox (should be: frame_second,object_class)
            const beforeParts = beforeBbox.split(',');
            console.log(`🚦 Line ${i} before parts:`, beforeParts);
            
            // Parse after bbox (should be: confidence,lat,lon,gps_time,video_type,stoplight_color)
            const afterParts = afterBbox.split(',');
            console.log(`🚦 Line ${i} after parts:`, afterParts);
            
            if (beforeParts.length < 2 || afterParts.length < 6) {
              console.warn(`🚦 Line ${i}: Wrong number of parts. Before: ${beforeParts.length}, After: ${afterParts.length}`);
              continue;
            }
            
            // Extract fields correctly
            const frame_second = beforeParts[0] || '0';
            const object_class = beforeParts[1] || '';
            const confidence = afterParts[0] || '0';
            const lat = afterParts[1] || '0';
            const lon = afterParts[2] || '0';
            const gps_time = afterParts[3] || '';
            const video_type = afterParts[4] || '';
            const stoplight_color = afterParts[5] || 'detected';
            
            console.log(`🚦 Line ${i} extracted fields:`);
            console.log(`  frame_second: "${frame_second}"`);
            console.log(`  object_class: "${object_class}"`);
            console.log(`  confidence: "${confidence}"`);
            console.log(`  lat: "${lat}"`);
            console.log(`  lon: "${lon}"`);
            console.log(`  gps_time: "${gps_time}"`);
            console.log(`  video_type: "${video_type}"`);
            console.log(`  stoplight_color: "${stoplight_color}"`);
            
            // Check if it's a traffic light
            if (object_class.trim() !== 'traffic light') {
              console.log(`🚦 Line ${i}: Not a traffic light, skipping`);
              continue;
            }
            
            // Parse and validate coordinates
            const latNum = parseFloat(lat);
            const lonNum = parseFloat(lon);
            
            console.log(`🚦 Line ${i} coordinates: lat=${latNum}, lon=${lonNum}`);
            
            if (isNaN(latNum) || isNaN(lonNum)) {
              console.warn(`🚦 Line ${i}: NaN coordinates: lat=${latNum}, lon=${lonNum}`);
              continue;
            }
            
            if (latNum < -90 || latNum > 90 || lonNum < -180 || lonNum > 180) {
              console.warn(`🚦 Line ${i}: Invalid range: lat=${latNum}, lon=${lonNum}`);
              continue;
            }
            
            // Create stoplight object
            const stoplight = {
              frame_second: parseInt(frame_second) || 0,
              object_class: object_class.trim(),
              bbox: bbox,
              confidence: parseFloat(confidence) || 0,
              lat: latNum,
              lon: lonNum,
              gps_time: gps_time.trim(),
              video_type: video_type.trim(),
              stoplight_color: stoplight_color.trim(),
              status: stoplight_color.trim()
            };
            
            stoplights.push(stoplight);
            
            if (stoplights.length <= 5) {
              console.log(`🚦 Valid stoplight ${stoplights.length}:`, stoplight);
            }
            
          } catch (lineError) {
            console.error(`🚦 Error parsing line ${i}:`, lineError);
          }
        }
        
        console.log('🚦 === RESULTS ===');
        console.log(`🚦 Total stoplights: ${stoplights.length}`);
        
        if (stoplights.length > 0) {
          const lats = stoplights.map(s => s.lat);
          const lons = stoplights.map(s => s.lon);
          console.log(`🚦 Coordinate range:`);
          console.log(`  Lat: ${Math.min(...lats)} to ${Math.max(...lats)}`);
          console.log(`  Lon: ${Math.min(...lons)} to ${Math.max(...lons)}`);
          
          onDataLoaded({ type: "stoplights", data: stoplights });
          alert(`🚦 SUCCESS! Loaded ${stoplights.length} valid stoplights!\n\nCoordinate range:\nLat: ${Math.min(...lats).toFixed(4)} to ${Math.max(...lats).toFixed(4)}\nLon: ${Math.min(...lons).toFixed(4)} to ${Math.max(...lons).toFixed(4)}`);
        } else {
          throw new Error("No valid stoplights found with corrected parsing");
        }
        
      } catch (err) {
        console.error('🚦 Corrected parser error:', err);
        setError("Failed to parse with corrected parser: " + err.message);
      }
    };
    reader.readAsText(file);
  };

  // Load object tracking data (similar to stoplights but different objects)
  const loadObjectTracking = () => {
    if (!file) return;

    const reader = new FileReader();
    reader.onload = e => {
      try {
        const text = e.target.result;
        const lines = text.split('\n').filter(line => line.trim() !== '');
        
        console.log('🚗 === OBJECT TRACKING PARSER ===');
        console.log('🚗 Total lines:', lines.length);
        console.log('🚗 Header:', lines[0]);
        console.log('🚗 Sample data line:', lines[1]);
        
        const objects = [];
        
        for (let i = 1; i < lines.length && i < 1001; i++) {
          const line = lines[i].trim();
          if (!line) continue;
          
          try {
            // Format: frame_second,object_class,"[bbox]",confidence,lat,lon,gps_time,video_type
            
            // Find the quoted section (bbox)
            const firstQuote = line.indexOf('"');
            const lastQuote = line.lastIndexOf('"');
            
            if (firstQuote === -1 || lastQuote === -1 || firstQuote === lastQuote) {
              console.warn(`🚗 Line ${i}: No proper quoted section found`);
              continue;
            }
            
            // Split into: before_bbox + bbox + after_bbox
            const beforeBbox = line.substring(0, firstQuote - 1);
            const bbox = line.substring(firstQuote + 1, lastQuote);
            const afterBbox = line.substring(lastQuote + 2);
            
            const beforeParts = beforeBbox.split(',');
            const afterParts = afterBbox.split(',');
            
            if (beforeParts.length < 2 || afterParts.length < 4) {
              console.warn(`🚗 Line ${i}: Wrong number of parts`);
              continue;
            }
            
            // Extract fields: frame_second,object_class,bbox,confidence,lat,lon,gps_time,video_type
            const frame_second = beforeParts[0] || '0';
            const object_class = beforeParts[1] || '';
            const confidence = afterParts[0] || '0';
            const lat = afterParts[1] || '0';
            const lon = afterParts[2] || '0';
            const gps_time = afterParts[3] || '';
            const video_type = afterParts[4] || '';
            
            // Parse and validate coordinates
            const latNum = parseFloat(lat);
            const lonNum = parseFloat(lon);
            
            if (isNaN(latNum) || isNaN(lonNum)) continue;
            if (latNum < -90 || latNum > 90 || lonNum < -180 || lonNum > 180) continue;
            
            // Create object detection entry
            const objectDetection = {
              frame: parseInt(frame_second) || 0,
              seconds: parseInt(frame_second) || 0,
              id: `${object_class}_${i}`, // Create unique ID
              type: object_class.trim(),
              object_class: object_class.trim(),
              bbox: bbox,
              confidence: parseFloat(confidence) || 0,
              lat: latNum,
              lon: lonNum,
              gps_time: gps_time.trim(),
              video_type: video_type.trim(),
              speed: 0, // Default since not provided
              bearing: null,
              distance: null,
              time: gps_time.trim()
            };
            
            objects.push(objectDetection);
            
            if (objects.length <= 5) {
              console.log(`🚗 Valid object ${objects.length}:`, objectDetection);
            }
            
          } catch (lineError) {
            console.error(`🚗 Error parsing line ${i}:`, lineError);
          }
        }
        
        console.log('🚗 === RESULTS ===');
        console.log(`🚗 Total objects: ${objects.length}`);
        
        if (objects.length > 0) {
          const lats = objects.map(o => o.lat);
          const lons = objects.map(o => o.lon);
          console.log(`🚗 Coordinate range: Lat ${Math.min(...lats)} to ${Math.max(...lats)}, Lon ${Math.min(...lons)} to ${Math.max(...lons)}`);
          
          onDataLoaded({ type: "object_detection", data: objects });
          alert(`🚗 SUCCESS! Loaded ${objects.length} object detections!`);
        } else {
          throw new Error("No valid object detections found");
        }
        
      } catch (err) {
        console.error('🚗 Object tracking parser error:', err);
        setError("Failed to parse object tracking: " + err.message);
      }
    };
    reader.readAsText(file);
  };

  // Load audio analysis data
  const loadAudioData = () => {
    if (!file) return;

    const reader = new FileReader();
    reader.onload = e => {
      try {
        const text = e.target.result;
        const lines = text.split('\n').filter(line => line.trim() !== '');
        
        console.log('🔊 === AUDIO DATA PARSER ===');
        console.log('🔊 Total lines:', lines.length);
        console.log('🔊 Header:', lines[0]);
        console.log('🔊 Sample data line:', lines[1]);
        
        if (lines.length < 2) throw new Error("CSV has no data");

        const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
        console.log('🔊 Headers:', headers);
        
        const audioPoints = [];
        
        for (let i = 1; i < lines.length && i < 1001; i++) {
          const line = lines[i].trim();
          if (!line) continue;
          
          try {
            const values = line.split(',');
            
            if (values.length < headers.length) {
              console.warn(`🔊 Line ${i}: Expected ${headers.length} values, got ${values.length}`);
              continue;
            }
            
            const row = {};
            headers.forEach((header, index) => {
              row[header] = values[index] ? values[index].trim() : '';
            });
            
            // Extract audio data
            const second = parseInt(row.second || 0);
            const noise_level = parseFloat(row.noise_level || 0);
            const lat = parseFloat(row.lat || 0);
            const lon = parseFloat(row.lon || 0);
            const gps_time = row.gps_time || '';
            const rms_energy = parseFloat(row.rms_energy || 0);
            const spectral_centroid = parseFloat(row.spectral_centroid || 0);
            const zero_crossing_rate = parseFloat(row.zero_crossing_rate || 0);
            
            // Validate coordinates
            if (isNaN(lat) || isNaN(lon)) continue;
            if (lat < -90 || lat > 90 || lon < -180 || lon > 180) continue;
            if (isNaN(noise_level)) continue;
            
            const audioPoint = {
              second: second,
              index: second,
              lat: lat,
              lon: lon,
              noise: noise_level,
              time: gps_time,
              rms_energy: rms_energy,
              spectral_centroid: spectral_centroid,
              zero_crossing_rate: zero_crossing_rate,
              // Include other audio features
              match_quality: row.match_quality || '',
              match_confidence: parseFloat(row.match_confidence || 0),
              match_avg_speed: parseFloat(row.match_avg_speed || 0)
            };
            
            audioPoints.push(audioPoint);
            
            if (audioPoints.length <= 5) {
              console.log(`🔊 Valid audio point ${audioPoints.length}:`, audioPoint);
            }
            
          } catch (lineError) {
            console.error(`🔊 Error parsing line ${i}:`, lineError);
          }
        }
        
        console.log('🔊 === RESULTS ===');
        console.log(`🔊 Total audio points: ${audioPoints.length}`);
        
        if (audioPoints.length > 0) {
          const lats = audioPoints.map(p => p.lat);
          const lons = audioPoints.map(p => p.lon);
          const noises = audioPoints.map(p => p.noise);
          console.log(`🔊 Coordinate range: Lat ${Math.min(...lats)} to ${Math.max(...lats)}, Lon ${Math.min(...lons)} to ${Math.max(...lons)}`);
          console.log(`🔊 Noise level range: ${Math.min(...noises)} to ${Math.max(...noises)} dB`);
          
          onDataLoaded({ type: "sound_heatmap", data: audioPoints });
          alert(`🔊 SUCCESS! Loaded ${audioPoints.length} audio data points!`);
        } else {
          throw new Error("No valid audio data found");
        }
        
      } catch (err) {
        console.error('🔊 Audio data parser error:', err);
        setError("Failed to parse audio data: " + err.message);
      }
    };
    reader.readAsText(file);
  };

  // Fallback parsers for other CSV types
  const loadOtherFormats = () => {
    if (!file) return;

    const reader = new FileReader();
    reader.onload = e => {
      try {
        const text = e.target.result;
        const lines = text.split('\n').filter(line => line.trim() !== '');
        
        if (lines.length < 2) throw new Error("CSV has no data");

        const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
        const rows = lines.slice(1).map(line => {
          const values = line.split(',');
          const row = {};
          headers.forEach((header, index) => {
            row[header] = values[index] ? values[index].trim() : '';
          });
          return row;
        });

        // Enhanced object detection (with object id)
        if (headers.includes("frame") && headers.includes("seconds") && headers.includes("object id")) {
          const objects = rows
            .filter(row => {
              const lat = parseFloat(row["object lat"]);
              const lon = parseFloat(row["object lon"] || row["object long"]);
              return !isNaN(lat) && !isNaN(lon) && lat !== 0 && lon !== 0;
            })
            .map(row => ({
              frame: parseInt(row.frame) || 0,
              seconds: parseInt(row.seconds) || 0,
              id: row["object id"],
              type: row["object type"],
              speed: parseFloat(row["object speed (mph)"] || 0),
              bearing: row["bearing (deg)"] ? parseFloat(row["bearing (deg)"]) : null,
              distance: row["distance (m)"] ? parseFloat(row["distance (m)"]) : null,
              lat: parseFloat(row["object lat"]),
              lon: parseFloat(row["object lon"] || row["object long"]),
              camLat: parseFloat(row["cam lat"] || 0),
              camLon: parseFloat(row["cam lon"] || row["cam long"] || 0),
              time: row.time || ''
            }));

          if (objects.length > 0) {
            onDataLoaded({ type: "object_detection", data: objects });
            return;
          }
        }

        // Aligned output
        if (headers.includes("second") && headers.includes("speed_mph")) {
          const points = rows
            .filter(row => {
              const lat = parseFloat(row.lat);
              const lon = parseFloat(row.lon);
              return !isNaN(lat) && !isNaN(lon);
            })
            .map(row => ({
              lat: parseFloat(row.lat),
              lon: parseFloat(row.lon),
              speed: parseFloat(row.speed_mph),
              second: parseInt(row.second)
            }));

          if (points.length > 0) {
            onDataLoaded({ type: "aligned_output", data: points });
            return;
          }
        }

        // Generic markers
        if (headers.includes("lat") && headers.includes("lon")) {
          const markers = rows
            .filter(row => {
              const lat = parseFloat(row.lat);
              const lon = parseFloat(row.lon);
              return !isNaN(lat) && !isNaN(lon);
            })
            .map(row => ({
              lat: parseFloat(row.lat),
              lon: parseFloat(row.lon),
              speed: parseFloat(row.speed_mph || 0),
              direction: parseFloat(row.direction || 0),
              type: (row.type || 'pedestrian').toLowerCase()
            }));

          if (markers.length > 0) {
            onDataLoaded({ type: "markers", data: markers });
            return;
          }
        }

        throw new Error("Unrecognized CSV format");

      } catch (err) {
        console.error('Other formats error:', err);
        setError("Failed to parse other formats: " + err.message);
      }
    };
    reader.readAsText(file);
  };

  return (
    <div style={{ 
      margin: '10px', 
      padding: '10px', 
      backgroundColor: '#eef', 
      border: '2px ridge #888',
      boxShadow: '2px 2px 4px rgba(0,0,0,0.3)'
    }}>
      <h4 style={{ fontFamily: 'MS Sans Serif, sans-serif', margin: '0 0 10px 0' }}>
        📂 MULTI-FORMAT CSV LOADER
      </h4>
      
      <div style={{ 
        backgroundColor: '#d1ecf1', 
        border: '1px solid #bee5eb', 
        padding: '8px', 
        borderRadius: '4px',
        marginBottom: '10px',
        fontSize: '0.85em'
      }}>
        <strong>🎯 Multi-Format Support:</strong> Handles stoplights, object tracking, audio data, and more!
      </div>

      <div style={{ marginBottom: '10px' }}>
        <input 
          type="file" 
          accept=".csv" 
          onChange={handleFileChange}
          style={{ fontFamily: 'Courier, monospace' }}
        />
      </div>
      
      {fileName && (
        <p style={{ fontSize: '0.9em', color: '#006', fontFamily: 'Courier, monospace' }}>
          💾 Selected: {fileName}
        </p>
      )}
      
      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        <button 
          onClick={loadStoplights} 
          disabled={!file}
          style={{
            padding: '6px 15px',
            backgroundColor: '#28a745',
            border: '2px outset #ddd',
            cursor: file ? 'pointer' : 'not-allowed',
            fontFamily: 'MS Sans Serif, sans-serif',
            fontWeight: 'bold',
            color: 'white'
          }}
        >
          🚦 STOPLIGHTS
        </button>

        <button 
          onClick={loadObjectTracking} 
          disabled={!file}
          style={{
            padding: '6px 15px',
            backgroundColor: '#007bff',
            border: '2px outset #ddd',
            cursor: file ? 'pointer' : 'not-allowed',
            fontFamily: 'MS Sans Serif, sans-serif',
            fontWeight: 'bold',
            color: 'white'
          }}
        >
          🚗 OBJECT TRACKING
        </button>

        <button 
          onClick={loadAudioData} 
          disabled={!file}
          style={{
            padding: '6px 15px',
            backgroundColor: '#6f42c1',
            border: '2px outset #ddd',
            cursor: file ? 'pointer' : 'not-allowed',
            fontFamily: 'MS Sans Serif, sans-serif',
            fontWeight: 'bold',
            color: 'white'
          }}
        >
          🔊 AUDIO DATA
        </button>

        <button 
          onClick={loadOtherFormats} 
          disabled={!file}
          style={{
            padding: '6px 15px',
            backgroundColor: '#6c757d',
            border: '2px outset #ddd',
            cursor: file ? 'pointer' : 'not-allowed',
            fontFamily: 'MS Sans Serif, sans-serif',
            fontWeight: 'bold',
            color: 'white'
          }}
        >
          📊 OTHER FORMATS
        </button>
      </div>
      
      {error && (
        <div style={{ 
          color: 'red', 
          marginTop: '10px', 
          fontFamily: 'Courier, monospace',
          fontSize: '0.85em',
          backgroundColor: '#ffe6e6',
          padding: '8px',
          border: '1px solid #ff9999',
          borderRadius: '4px'
        }}>
          <strong>⚠️ Error:</strong><br/>
          {error}
        </div>
      )}
      
      <div style={{ 
        marginTop: '10px', 
        fontSize: '0.8em', 
        color: '#666',
        fontFamily: 'Arial, sans-serif'
      }}>
        <strong>✅ Supported formats:</strong>
        <ul style={{ marginTop: '5px', paddingLeft: '20px' }}>
          <li><strong>🚦 Stoplights:</strong> frame_second,object_class,"[bbox]",confidence,lat,lon,gps_time,video_type,stoplight_color</li>
          <li><strong>🚗 Object Tracking:</strong> frame_second,object_class,"[bbox]",confidence,lat,lon,gps_time,video_type</li>
          <li><strong>🔊 Audio Data:</strong> second,rms_energy,spectral_centroid,zero_crossing_rate,noise_level,lat,lon,gps_time,...</li>
          <li><strong>📊 Enhanced Object Detection:</strong> frame,seconds,object id,object type,object speed (mph),object lat,object lon,...</li>
          <li><strong>🛣️ GPS Tracking:</strong> second,lat,lon,speed_mph,gpx_time</li>
        </ul>
        <div style={{ marginTop: '8px', fontSize: '0.75em', color: '#888' }}>
          <strong>Expected coordinate ranges for Minnesota:</strong><br/>
          Lat: 44.863 to 44.864 | Lon: -93.245 to -93.244
        </div>
      </div>
    </div>
  );
}

export default CsvLocalLoader;
