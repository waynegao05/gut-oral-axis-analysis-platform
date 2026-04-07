from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from config.settings import APP_NAME, DEBUG, HOST, PORT
from src.pipeline import run_pipeline

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', app_name=APP_NAME)


@app.route('/analyze', methods=['POST'])
def analyze():
    payload = request.get_json(force=True)
    return jsonify(run_pipeline(payload))


if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)
