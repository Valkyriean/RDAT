from flask import Flask, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = "nana7mi"

@app.route("/")
def serve():
    return send_file('index.html')