# app.py
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'gpx', 'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        gpx = request.files.get('gpx_file')
        mp4 = request.files.get('mp4_file')

        if not gpx or not mp4 or not allowed_file(gpx.filename) or not allowed_file(mp4.filename):
            flash('Invalid file(s).')
            return redirect(request.url)

        gpx_filename = secure_filename(gpx.filename)
        mp4_filename = secure_filename(mp4.filename)
        gpx.save(os.path.join(app.config['UPLOAD_FOLDER'], gpx_filename))
        mp4.save(os.path.join(app.config['UPLOAD_FOLDER'], mp4_filename))

        flash('Files uploaded successfully.')
        return redirect(url_for('upload_file'))

    return render_template('index.html')
