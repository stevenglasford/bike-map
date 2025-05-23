// frontend/src/UploadForm.jsx
import React, { useState } from 'react';

function UploadForm() {
  const [videoFile, setVideoFile] = useState(null);
  const [gpxFile, setGpxFile] = useState(null);
  const [email, setEmail] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!videoFile && !gpxFile) return;
    const formData = new FormData();
    if (videoFile) formData.append("video", videoFile);
    if (gpxFile)   formData.append("gpx", gpxFile);
    if (email)     formData.append("email", email);

    const response = await fetch('http://localhost:5000/api/upload', {
      method: 'POST',
      body: formData
    });
    const result = await response.json();
    if (response.ok) {
      alert("Files uploaded successfully!");
    } else {
      alert("Upload failed: " + (result.error||"Unknown error"));
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ margin: '20px', padding: '10px', border: '2px solid #000', backgroundColor: '#fff' }}>
      <h2>Upload Video and GPX</h2>
      <div>
        <label>Video (.mp4): </label>
        <input type="file" accept="video/mp4" onChange={e => setVideoFile(e.target.files[0])} />
      </div>
      <div>
        <label>Track (.gpx): </label>
        <input type="file" accept=".gpx" onChange={e => setGpxFile(e.target.files[0])} />
      </div>
      <div>
        <label>Email (optional): </label>
        <input type="email" value={email} onChange={e => setEmail(e.target.value)} 
               placeholder="you@example.com" />
      </div>
      <button type="submit">Submit</button>
    </form>
  );
}

export default UploadForm;