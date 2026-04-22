from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from config.settings import APP_NAME, DEBUG, HOST, PORT
from src.clinical_standardizer import standardize_raw_payload
from src.export_utils import export_report
from src.logging_utils import get_logger
from src.pipeline import run_pipeline
from src.validators import REQUIRED_TOP_LEVEL_KEYS, validate_payload

app = Flask(__name__)
logger = get_logger("gut_oral_axis")


def _is_canonical_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return all(isinstance(payload.get(key), dict) for key in REQUIRED_TOP_LEVEL_KEYS)


def _normalize_payload(payload: object) -> tuple[dict, str]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    if _is_canonical_payload(payload):
        return payload, "canonical"

    return standardize_raw_payload(payload), "raw_standardized"


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', app_name=APP_NAME)


@app.route('/standardize', methods=['POST'])
def standardize():
    payload = request.get_json(force=True)
    try:
        standardized_payload, source_format = _normalize_payload(payload)
    except ValueError as exc:
        return jsonify({"ok": False, "errors": [str(exc)]}), 400

    valid, errors = validate_payload(standardized_payload)
    if not valid:
        logger.warning("Invalid standardized payload: %s", errors)
        return jsonify({"ok": False, "errors": errors}), 400

    return jsonify(
        {
            "ok": True,
            "source_format": source_format,
            "standardized_payload": standardized_payload,
        }
    )


@app.route('/analyze', methods=['POST'])
def analyze():
    payload = request.get_json(force=True)
    try:
        standardized_payload, source_format = _normalize_payload(payload)
    except ValueError as exc:
        return jsonify({"ok": False, "errors": [str(exc)]}), 400

    valid, errors = validate_payload(standardized_payload)
    if not valid:
        logger.warning("Invalid payload: %s", errors)
        return jsonify({"ok": False, "errors": errors}), 400

    report = run_pipeline(standardized_payload)
    output_path = export_report(report)
    logger.info("Analysis completed. Report saved to %s", output_path)
    return jsonify(
        {
            "ok": True,
            "source_format": source_format,
            "standardized_payload": standardized_payload,
            "report": report,
            "risk_result": report.get("risk_result", {}),
            "recommendations": report.get("recommendations", []),
            "top_microbes": report.get("top_microbes", []),
            "saved_to": output_path,
        }
    )


if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)
