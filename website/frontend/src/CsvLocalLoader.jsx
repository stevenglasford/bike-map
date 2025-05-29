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

  const parseAndLoadFile = () => {
    if (!file) return;

    const reader = new FileReader();

    reader.onload = e => {
      try {
        const lines = e.target.result.split('\n').filter(line => line.trim() !== '');
        if (lines.length < 2) throw new Error("CSV has no data");

        const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
        const rows = lines.slice(1).map(line => {
          const values = line.split(',').map(v => v.trim());
          return Object.fromEntries(values.map((v, i) => [headers[i], v]));
        });

        // Enhanced object detection CSV (enhanced_output or speed_table)
        if (
          headers.includes("frame") &&
          headers.includes("seconds") &&
          headers.includes("object id") &&
          headers.includes("object type") &&
          headers.includes("object speed (mph)")
        ) {
          const objects = rows
            .filter(row => {
              // Only include rows where we have actual coordinate data
              const hasCoords = (row["object lat"] && row["object lon"]) || 
                                (row["object lat"] && row["object long"]);
              if (!hasCoords) return false;
              
              const lat = parseFloat(row["object lat"]);
              const lon = parseFloat(row["object lon"] || row["object long"]);
              
              // Make sure coordinates are valid numbers
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

          if (objects.length === 0) throw new Error("No valid object detection data found");
          onDataLoaded({ type: "object_detection", data: objects });
          return;
        }

        // Object counts summary CSV
        if (
          headers.includes("object_id") &&
          headers.includes("type") &&
          headers.includes("frames_seen")
        ) {
          const counts = rows
            .filter(row => row.object_id && row.type)
            .map(row => ({
              id: row.object_id,
              type: row.type,
              motion: row.motion || 'unknown',
              framesSeen: parseInt(row.frames_seen),
              lastLat: parseFloat(row.last_lat),
              lastLon: parseFloat(row.last_lon)
            }));

          if (counts.length === 0) throw new Error("No valid object count data found");
          onDataLoaded({ type: "object_counts", data: counts });
          return;
        }

        // Stoplight detection CSV
        if (
          headers.includes("stoplight") &&
          headers.includes("second") &&
          headers.includes("lat")
        ) {
          const stoplights = rows
            .filter(row => {
              const lat = parseFloat(row.lat);
              const lon = parseFloat(row.long || row.lon);
              return !isNaN(lat) && !isNaN(lon);
            })
            .map(row => ({
              status: row.stoplight,
              second: parseInt(row.second),
              lat: parseFloat(row.lat),
              lon: parseFloat(row.long || row.lon),
              time: row.gpx_time
            }));

          if (stoplights.length === 0) throw new Error("No valid stoplight data found");
          onDataLoaded({ type: "stoplights", data: stoplights });
          return;
        }

        // Frame noise CSV
        if (
          headers.includes("frame") &&
          headers.includes("noise") &&
          headers.includes("lat")
        ) {
          const points = rows
            .filter(row => {
              const lat = parseFloat(row.lat);
              const lon = parseFloat(row.long || row.lon);
              const noise = parseFloat(row.noise);
              return !isNaN(lat) && !isNaN(lon) && !isNaN(noise);
            })
            .map(row => ({
              lat: parseFloat(row.lat),
              lon: parseFloat(row.long || row.lon),
              noise: parseFloat(row.noise),
              time: row.gpx_time,
              index: parseInt(row.frame)
            }));

          if (points.length === 0) throw new Error("No valid frame noise data found");
          onDataLoaded({ type: "frame_noise", data: points });
          return;
        }

        // Existing handlers...
        if (headers.includes("second") && headers.includes("lat") && headers.includes("lon") && headers.includes("speed_mph")) {
          const points = rows
            .filter(row =>
              row.lat && row.lon && row.speed_mph &&
              !isNaN(parseFloat(row.lat)) &&
              !isNaN(parseFloat(row.lon)) &&
              !isNaN(parseFloat(row.speed_mph))
            )
            .map(row => ({
              lat: parseFloat(row.lat),
              lon: parseFloat(row.lon),
              speed: parseFloat(row.speed_mph),
              second: parseInt(row.second)
            }));

          if (points.length === 0) throw new Error("No valid speed data found");
          onDataLoaded({ type: "aligned_output", data: points });
          return;
        }

        if (headers.includes("lat") && headers.includes("lon")) {
          const markers = rows
            .filter(row =>
              row.lat && row.lon &&
              !isNaN(parseFloat(row.lat)) &&
              !isNaN(parseFloat(row.lon))
            )
            .map(row => ({
              lat: parseFloat(row.lat),
              lon: parseFloat(row.lon),
              speed: parseFloat(row.speed_mph || 0),
              direction: parseFloat(row.direction || 0),
              type: (row.type || 'pedestrian').toLowerCase()
            }));

          if (markers.length === 0) throw new Error("No valid marker data found");
          onDataLoaded({ type: "markers", data: markers });
          return;
        }
        
        if (
          headers.includes("second") &&
          headers.includes("lat") &&
          (headers.includes("lon") || headers.includes("long")) &&
          headers.includes("speed_mph") &&
          headers.includes("gpx_time")
        ) {
          const points = rows
            .filter(row =>
              row.lat && (row.lon || row.long) && row.speed_mph &&
              !isNaN(parseFloat(row.lat)) &&
              !isNaN(parseFloat(row.lon || row.long)) &&
              !isNaN(parseFloat(row.speed_mph))
            )
            .map(row => ({
              lat: parseFloat(row.lat),
              lon: parseFloat(row.lon || row.long),
              speed: parseFloat(row.speed_mph),
              second: parseInt(row.second),
              gpx_time: row.gpx_time
            }));
        
          if (points.length === 0) throw new Error("No valid merged_output data found");
          onDataLoaded({ type: "merged_output", data: points });
          return;
        }
        
        if (
          (headers.includes("frame") || headers.includes("second")) &&
          headers.includes("gpx_time") &&
          headers.includes("lat") &&
          (headers.includes("lon") || headers.includes("long")) &&
          headers.includes("noise")
        ) {
          const keyField = headers.includes("frame") ? "frame" : "second";
          const points = rows
            .filter(row =>
              row.lat && (row.lon || row.long) && row.noise &&
              !isNaN(parseFloat(row.lat)) &&
              !isNaN(parseFloat(row.lon || row.long)) &&
              !isNaN(parseFloat(row.noise))
            )
            .map(row => ({
              lat: parseFloat(row.lat),
              lon: parseFloat(row.lon || row.long),
              noise: parseFloat(row.noise),
              time: row.gpx_time,
              index: parseInt(row[keyField])
            }));
        
          if (points.length === 0) throw new Error("No valid noise points found");
          onDataLoaded({ type: "sound_heatmap", data: points });
          return;
        }
              
        throw new Error("Unrecognized CSV format.");
      } catch (err) {
        console.error(err);
        setError("Failed to parse CSV file: " + err.message);
        onDataLoaded({ type: "error", data: [] });
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
        📂 Load Local CSV Data
      </h4>
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
      <button 
        onClick={parseAndLoadFile} 
        disabled={!file}
        style={{
          padding: '4px 12px',
          backgroundColor: '#c0c0c0',
          border: '2px outset #ddd',
          cursor: file ? 'pointer' : 'not-allowed',
          fontFamily: 'MS Sans Serif, sans-serif'
        }}
      >
        ▶️ Load CSV to Map
      </button>
      {error && (
        <p style={{ 
          color: 'red', 
          marginTop: '10px', 
          fontFamily: 'Courier, monospace',
          fontSize: '0.9em' 
        }}>
          ⚠️ {error}
        </p>
      )}
      <div style={{ 
        marginTop: '10px', 
        fontSize: '0.8em', 
        color: '#666',
        fontFamily: 'Arial, sans-serif'
      }}>
        <strong>Supported formats:</strong>
        <ul style={{ marginTop: '5px', paddingLeft: '20px' }}>
          <li>Speed tracking (aligned_output.csv, merged_output.csv)</li>
          <li>Object detection (enhanced_output_*.csv, speed_table_*.csv)</li>
          <li>Object counts summary (*_object_counts.csv)</li>
          <li>Stoplight detection (stoplights_*.csv)</li>
          <li>Sound/noise data (sound_output_*.csv, frame_noise_*.csv)</li>
        </ul>
      </div>
    </div>
  );
}

export default CsvLocalLoader;