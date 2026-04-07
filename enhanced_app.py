from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from config.settings import APP_NAME, DEBUG, HOST, PORT
from src.export_utils import export_report
from src.logging_utils import get_logger
from src.pipeline import run_pipeline
from src.validators import validate_payload

app = Flask(__name__)
logger = get_logger("gut_oral_axis")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', app_name=APP_NAME)


@app.route('/analyze', methods=['POST'])
def analyze():
    payload = request.get_json(force=True)
    valid, errors = validate_payload(payload)
    if not valid:
        logger.warning("Invalid payload: %s", errors)
        return jsonify({"ok": False, "errors": errors}), 400

    report = run_pipeline(payload)
    output_path = export_report(report)
    logger.info("Analysis completed. Report saved to %s", output_path)
    return jsonify({"ok": True, "report": report, "saved_to": output_path})


if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)
