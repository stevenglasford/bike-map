# backend/app.py
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import os, shutil, hashlib, uuid
from datetime import datetime
import subprocess
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__, static_folder="../frontend/build", static_url_path="/")
CORS(app)  # Allow cross-origin requests [oai_citation:2‡stackoverflow.com](https://stackoverflow.com/questions/25594893/how-to-enable-cors-in-flask#:~:text=I%20resolved%20this%20same%20problem,py)


def cleanup_orphans():
    for uid in os.listdir(UPLOAD_BASE):
        user_dir = os.path.join(UPLOAD_BASE, uid)
        if not os.path.isdir(user_dir): 
            continue
        files = os.listdir(user_dir)
        has_video = any(f.lower().endswith('.mp4') for f in files)
        has_gpx   = any(f.lower().endswith('.gpx') for f in files)
        # If one file missing, delete entire directory
        if has_video != has_gpx:
            shutil.rmtree(user_dir)
            print(f"Deleted orphan directory: {user_dir}")

scheduler = BackgroundScheduler()
# Schedule cleanup at midnight every day (server time)
scheduler.add_job(cleanup_orphans, 'cron', hour=0, minute=0)
scheduler.start()

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static_file(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

@app.route('/api/data', methods=['GET'])
def get_data():
    user_id = request.args.get('user') or request.cookies.get('user_id')
    personal = request.args.get('personal') == '1' or bool(user_id)
    data = {
        "speed_points": [],   # for speed heatmap
        "sound_points": [],   # for sound heatmap
        "markers": [],        # dynamic object icons
        "traffic_lights": []  # stoplight metadata
    }
    # Collect data from each upload directory
    for uid in os.listdir(UPLOAD_BASE):
        if personal and user_id and not uid.startswith(user_id):
            # skip other users if viewing personal data
            continue
        user_dir = os.path.join(UPLOAD_BASE, uid)
        # Example: assume main_processor output CSV with lat/lon/speed
        aligned_csv = os.path.join(user_dir, "aligned_output.csv")
        if os.path.exists(aligned_csv):
            import csv
            with open(aligned_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lat = float(row['lat']); lon = float(row['lon'])
                    speed = float(row['speed_mph'])
                    # Add to speed heatmap points (intensity = speed)
                    data["speed_points"].append([lat, lon, speed])
                    # For demonstration, add to sound heatmap too (fake values)
                    sound_level = 30 + speed * 0.5  # example formula
                    data["sound_points"].append([lat, lon, sound_level])
        # Example: parse human_counter CSV output for markers
        humans_csv = os.path.join(user_dir, "humans_counts.csv")
        if os.path.exists(humans_csv):
            # (Imagine this CSV has counts or object speeds; we'll mock one marker per video)
            # For simplicity, place markers at first GPS point with type and speed
            with open(humans_csv) as f:
                reader = csv.DictReader(f)
                first = next(reader, None)
                if first:
                    lat = float(first['lat']); lon = float(first['lon'])
                    # Random example: one pedestrian, one car
                    data["markers"].append({
                        "type": "pedestrian", "lat": lat, "lon": lon, 
                        "speed": first.get('walking_speed',0), "direction": 90
                    })
                    data["markers"].append({
                        "type": "car", "lat": lat, "lon": lon, 
                        "speed": first.get('car_speed',0), "direction": 45
                    })
        # Example: Traffic light metadata
        traffic_csv = os.path.join(user_dir, "traffic_light.csv")
        if os.path.exists(traffic_csv):
            # Suppose CSV has columns avg_wait, green_success, lat, lon
            import csv
            with open(traffic_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data["traffic_lights"].append({
                        "lat": float(row["lat"]), "lon": float(row["lon"]),
                        "avg_wait": float(row["avg_wait"]), 
                        "green_success": float(row["green_success"])
                    })
    return jsonify(data)



@app.route('/api/upload', methods=['POST'])
def upload_files():
    # Expect fields: 'video' (MP4), 'gpx' (GPX), optional 'email'
    video = request.files.get('video')
    gpx = request.files.get('gpx')
    email = request.form.get('email', '').strip()
    if not video and not gpx:
        return jsonify({"error": "No file provided"}), 400

    # Create a unique directory for this upload (hashed for distribution)
    uid = uuid.uuid4().hex  # unique ID for directory
    if email:
        # Optionally incorporate email hash (e.g. for personal data grouping)
        email_id = hashlib.md5(email.encode()).hexdigest()[:8]
        uid = f"{email_id}_{uid}"
    upload_dir = os.path.join(UPLOAD_BASE, uid)
    os.makedirs(upload_dir, exist_ok=True)

    # Save uploaded files
    if video:
        video_path = os.path.join(upload_dir, video.filename)
        video.save(video_path)
    if gpx:
        gpx_path = os.path.join(upload_dir, gpx.filename)
        gpx.save(gpx_path)

    # If both files are present, trigger processing immediately
    if video and gpx:
        # We call the existing processing pipeline scripts asynchronously
        # For example, using subprocess to run main_processor.py
        subprocess.Popen([
            "python", "main_processor.py", upload_dir
        ])
        # Also run human counter on the video (if needed by pipeline)
        if video:
            subprocess.Popen([
                "python", "human_counter_extended_with_totals_fixed.py", video_path
            ])
    else:
        # One file is missing; it will be handled by nightly cleanup (see below)
        pass

    # Set a cookie to identify this user (for personal data viewing)
    resp = make_response(jsonify({"status": "uploaded", "id": uid}))
    resp.set_cookie('user_id', uid, max_age=30*24*3600)  # expire in 30 days
    return resp

# Directory to store uploads and processing output
UPLOAD_BASE = '/mnt/penis/uploads'
os.makedirs(UPLOAD_BASE, exist_ok=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, ssl_context=('/mnt/penis/cert.pem', '/mnt/penis/key.pem'))