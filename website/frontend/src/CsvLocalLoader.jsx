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
        
        // Detect merged_output.csv
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
    <div style={{ margin: '10px', padding: '10px', backgroundColor: '#eef', border: '1px solid #ccc' }}>
      <h4>Load Local CSV Data</h4>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      {fileName && <p style={{ fontSize: '0.9em' }}>Selected file: {fileName}</p>}
      <button onClick={parseAndLoadFile} disabled={!file}>
        Load CSV to Map
      </button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}

export default CsvLocalLoader;