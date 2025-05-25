import React, { useState } from 'react';

function UploadForm({ onUploadComplete }) {
  const [files, setFiles] = useState([]);
  const [email, setEmail] = useState("");
  const [uploading, setUploading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (files.length === 0) {
      alert("Please select at least one .mp4 or .gpx file.");
      return;
    }

    setUploading(true);
    const formData = new FormData();
    files.forEach(file => {
      const name = file.name.toLowerCase();
      if (name.endsWith('.mp4')) formData.append("video", file);
      else if (name.endsWith('.gpx')) formData.append("gpx", file);
    });
    if (email) formData.append("email", email);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const result = await response.json();
      if (response.ok) {
        alert("Files uploaded successfully!");
        if (onUploadComplete) onUploadComplete();  // Refresh map
      } else {
        alert("Upload failed: " + (result.error || "Unknown error"));
      }
    } catch (err) {
      alert("Upload failed: Network or server error.");
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ margin: '20px', padding: '10px', border: '2px solid #000', backgroundColor: '#fff' }}>
      <h3>Upload Video or GPX</h3>
      <input
        type="file"
        accept=".mp4,.gpx"
        multiple
        onChange={e => setFiles(Array.from(e.target.files))}
      />
      <div>
        <label>Email (optional): </label>
        <input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@example.com" />
      </div>
      <button type="submit" disabled={uploading}>
        {uploading ? "Uploading..." : "Upload File(s)"}
      </button>
    </form>
  );
}

export default UploadForm;