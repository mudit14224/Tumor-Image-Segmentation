from flask import Flask


app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'App/static/scans/mri'
from App import routes