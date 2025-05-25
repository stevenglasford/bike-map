import React from 'react';
import MapView from './MapView';

function App() {
  return (
    <div style={{textAlign: 'center', fontFamily: 'Courier, monospace'}}>
      <h1 style={{background:'#000080', color:'#fff', padding:'10px'}}>
        Glasford Traffic Analyzer
      </h1>
      
      <MapView />
    </div>
  );
}

export default App;