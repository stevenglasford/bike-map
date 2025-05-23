// frontend/src/App.js
import React from 'react';
import UploadForm from './UploadForm';
import MapView from './MapView';

function App() {
  return (
    <div style={{textAlign: 'center', fontFamily: 'Courier, monospace'}}>
      <h1 style={{background:'#000080', color:'#fff', padding:'10px'}}>
        Glasford Traffic Analyzer
      </h1>
      <UploadForm />
      <hr/>
      <MapView />
    </div>
  );
}

export default App;