from __future__ import annotations

import errno

from flask import Flask, jsonify, render_template, request

from config.settings import (
    AC_ICAM_V8_RELEASE_NAME,
    APP_NAME,
    DEBUG,
    ENABLE_INTERNAL_ORAL_ADENOMA,
    HOST,
    PORT,
    RESEARCH_MODEL_RELEASE_NAME,
    TEMPORAL_TOPOLOGY_RELEASE_NAME,
    USE_RELOADER,
    WEB_MODEL_BACKEND,
)
from src.clinical_standardizer import standardize_raw_payload
from src.export_utils import export_report
from src.logging_utils import get_logger
from src.pipeline import run_pipeline
from src.validators import (
    REQUIRED_TOP_LEVEL_KEYS,
    V8_WEB_REQUIRED_CLINICAL_FIELDS,
    validate_payload,
)

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


def _validate_for_active_backend(payload: dict) -> tuple[bool, list[str]]:
    return validate_payload(
        payload,
        require_positive_microbes=WEB_MODEL_BACKEND != "ac_icam_v8",
        required_clinical_fields=(
            V8_WEB_REQUIRED_CLINICAL_FIELDS
            if WEB_MODEL_BACKEND == "ac_icam_v8"
            else ()
        ),
    )


def _active_release_name() -> str:
    return {
        "ac_icam_v8": AC_ICAM_V8_RELEASE_NAME,
        "temporal_topology": TEMPORAL_TOPOLOGY_RELEASE_NAME,
        "legacy_cox": RESEARCH_MODEL_RELEASE_NAME,
    }.get(WEB_MODEL_BACKEND, WEB_MODEL_BACKEND)


@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        app_name=APP_NAME,
        web_model_backend=WEB_MODEL_BACKEND,
        model_release=_active_release_name(),
        internal_oral_adenoma_enabled=ENABLE_INTERNAL_ORAL_ADENOMA,
    )


@app.route('/internal/oral-adenoma/schema', methods=['GET'])
def oral_adenoma_schema():
    if not ENABLE_INTERNAL_ORAL_ADENOMA:
        return jsonify({"ok": False, "errors": ["内部口腔腺瘤模型未启用。"]}), 404

    try:
        from src.oral_adenoma_bridge import get_oral_adenoma_bridge

        bridge = get_oral_adenoma_bridge()
    except Exception:
        logger.exception("Unable to load the oral adenoma release.")
        return jsonify(
            {"ok": False, "errors": ["内部口腔腺瘤模型暂时不可用。"]}
        ), 503

    return jsonify(
        {
            "ok": True,
            "model_release": bridge.release_name,
            "research_only": True,
            "input_unit": "percent",
            "required_sum_range_percent": [bridge.sum_min, bridge.sum_max],
            "feature_count": len(bridge.feature_ids),
            "feature_ids": list(bridge.feature_ids),
            "taxonomies": list(bridge.taxonomies),
            "accepted_sample_types": sorted(bridge.artifact["allowed_sample_types"]),
            "claim_boundary": bridge.artifact["claim_boundary"],
        }
    )


@app.route('/internal/oral-adenoma/analyze', methods=['POST'])
def analyze_oral_adenoma():
    if not ENABLE_INTERNAL_ORAL_ADENOMA:
        return jsonify({"ok": False, "errors": ["内部口腔腺瘤模型未启用。"]}), 404

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "errors": ["输入必须是 JSON 对象。"]}), 400

    try:
        from src.oral_adenoma_bridge import get_oral_adenoma_bridge

        prediction = get_oral_adenoma_bridge().score(
            payload.get("oral_abundances"),
            sample_type=payload.get("sample_type"),
        )
    except ValueError as exc:
        logger.warning("Invalid oral adenoma input: %s", exc)
        return jsonify({"ok": False, "errors": [str(exc)]}), 400
    except Exception:
        logger.exception("Oral adenoma analysis failed.")
        return jsonify(
            {"ok": False, "errors": ["内部口腔腺瘤模型运行失败。"]}
        ), 503

    return jsonify({"ok": True, "oral_adenoma_result": prediction.as_dict()})


@app.route('/standardize', methods=['POST'])
def standardize():
    payload = request.get_json(force=True)
    try:
        standardized_payload, source_format = _normalize_payload(payload)
    except ValueError as exc:
        return jsonify({"ok": False, "errors": [str(exc)]}), 400

    valid, errors = _validate_for_active_backend(standardized_payload)
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

    valid, errors = _validate_for_active_backend(standardized_payload)
    if not valid:
        logger.warning("Invalid payload: %s", errors)
        return jsonify({"ok": False, "errors": errors}), 400

    try:
        report = run_pipeline(standardized_payload)
        output_path = export_report(report)
        logger.info("Analysis completed. Report saved to %s", output_path)
    except ValueError as exc:
        logger.warning("Analysis rejected invalid input: %s", exc)
        return jsonify({"ok": False, "errors": [str(exc)]}), 400
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
            "general_risk_result": report.get("general_risk_result", {}),
            "recommendations": report.get("recommendations", []),
            "pharmacy_assessment": report.get("pharmacy_assessment", {}),
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
