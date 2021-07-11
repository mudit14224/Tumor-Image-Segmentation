from flask import Flask
from flask.templating import render_template
from flask import request
from werkzeug.utils import secure_filename
from MRI import mri
import os
from App import app

@app.route('/')
@app.route('/home')
def home_page():
    return render_template("index.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_PATH'], filename)
        f.save(path)
        result = mri.result(path, filename)
        path = 'static/scans/results/' + filename + "_mask.jpeg"
        return render_template("result.html", path=path)

@app.route('/dev')
def dev_page():
    return render_template("developers.html")

@app.route('/info')
def info_page():
    return render_template("info.html")


    