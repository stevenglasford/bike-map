import React, { useState } from 'react';

function CsvLocalLoader({ onDataLoaded }) {
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState(null);

  const handleFileChange = e => {
    const file = e.target.files[0];
    if (!file || !file.name.toLowerCase().endsWith('.csv')) {
      setError("Only .csv files are supported.");
      return;
    }

    setFileName(file.name);
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

        if (markers.length === 0) throw new Error("No valid location data found");

        onDataLoaded(markers);
        setError(null);
      } catch (err) {
        console.error(err);
        setError("Failed to parse CSV file: " + err.message);
        onDataLoaded([]);  // clear previous markers
      }
    };

    reader.readAsText(file);
  };

  return (
    <div style={{ margin: '10px', padding: '10px', backgroundColor: '#eef', border: '1px solid #ccc' }}>
      <h4>Load Local CSV Data</h4>
      <input type="file" accept=".csv" onChange={handleFileChange} />
      {fileName && <p style={{ fontSize: '0.9em' }}>Loaded file: {fileName}</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}

export default CsvLocalLoader;