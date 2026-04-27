from __future__ import annotations

import errno

from flask import Flask, jsonify, render_template, request

from config.settings import APP_NAME, DEBUG, HOST, PORT, USE_RELOADER
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

    try:
        report = run_pipeline(standardized_payload)
        output_path = export_report(report)
        logger.info("Analysis completed. Report saved to %s", output_path)
    except Exception as exc:
        logger.exception("Analysis failed.")
        return jsonify({"ok": False, "errors": [f"Analysis backend failed: {exc}"]}), 500
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


def _is_bind_error(exc: OSError) -> bool:
    return getattr(exc, "winerror", None) == 10013 or getattr(exc, "errno", None) in {
        errno.EACCES,
        errno.EADDRINUSE,
    }


def _run_app() -> None:
    candidate_ports = [PORT, 8765, 8000, 8080]
    tried_ports: list[int] = []
    last_error: OSError | None = None

    for candidate_port in candidate_ports:
        if candidate_port in tried_ports:
            continue
        tried_ports.append(candidate_port)
        try:
            print(f"Starting web app at http://{HOST}:{candidate_port}", flush=True)
            app.run(host=HOST, port=candidate_port, debug=DEBUG, use_reloader=USE_RELOADER)
            return
        except OSError as exc:
            if not _is_bind_error(exc):
                raise
            last_error = exc
            print(f"Port {candidate_port} unavailable: {exc}", flush=True)

    if last_error is not None:
        raise RuntimeError(
            "Unable to start Flask app because all candidate ports were blocked. "
            "Set GOA_PORT to an allowed port, for example: set GOA_PORT=8765"
        ) from last_error


if __name__ == '__main__':
    _run_app()
