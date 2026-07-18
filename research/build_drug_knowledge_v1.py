from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_PATH = ROOT / "data" / "pharmacy_knowledge" / "medication_seed_v1.json"
DEFAULT_OUTPUT_PATH = (
    ROOT / "data" / "pharmacy_knowledge" / "openfda_label_evidence_v1.json"
)

RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
USER_AGENT = (
    "gut-oral-axis-analysis-platform/1.1 "
    "(https://github.com/waynegao05/gut-oral-axis-analysis-platform)"
)

LABEL_SECTIONS = (
    "boxed_warning",
    "indications_and_usage",
    "purpose",
    "uses",
    "dosage_and_administration",
    "directions",
    "dosage_forms_and_strengths",
    "contraindications",
    "warnings",
    "warnings_and_cautions",
    "do_not_use",
    "ask_doctor",
    "ask_doctor_or_pharmacist",
    "stop_use",
    "drug_interactions",
    "use_in_specific_populations",
    "pregnancy",
    "pregnancy_or_breast_feeding",
    "pediatric_use",
    "geriatric_use",
    "overdosage",
)
CORE_LABEL_SECTIONS = (
    "indications_and_usage",
    "uses",
    "dosage_and_administration",
    "directions",
    "contraindications",
    "warnings",
    "warnings_and_cautions",
    "drug_interactions",
)
DEFAULT_PRODUCT_TYPES = ("HUMAN PRESCRIPTION DRUG",)

JsonFetcher = Callable[[str], Mapping[str, Any]]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _validate_seed(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    medications = payload.get("medications")
    if not isinstance(medications, list) or not medications:
        raise ValueError("Medication seed must contain a non-empty medications list.")

    required = {
        "drug_id",
        "display_name",
        "rxnorm_query",
        "openfda_generic_names",
        "aliases",
    }
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(medications):
        if not isinstance(item, dict):
            raise ValueError(f"Medication seed item {index} must be an object.")
        missing = sorted(required.difference(item))
        if missing:
            raise ValueError(
                f"Medication seed item {index} is missing: {', '.join(missing)}"
            )
        drug_id = str(item["drug_id"]).strip()
        if not drug_id or drug_id in seen_ids:
            raise ValueError(f"Medication seed has invalid or duplicate drug_id: {drug_id}")
        generic_names = _string_list(item["openfda_generic_names"])
        if not generic_names:
            raise ValueError(f"Medication {drug_id} has no openFDA generic name.")
        seen_ids.add(drug_id)
        records.append(dict(item))
    return records


def _fetch_json(
    url: str,
    *,
    timeout: float,
    attempts: int = 3,
    retry_delay: float = 1.0,
) -> Mapping[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object from {url}")
            return payload
        except HTTPError as exc:
            if exc.code == 404:
                return {}
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(retry_delay * (2**attempt))
    raise RuntimeError(f"Unable to retrieve {url}: {last_error}") from last_error


def _append_api_key(url: str, api_key: str | None) -> str:
    if not api_key:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode({'api_key': api_key})}"


def _rxnorm_lookup(
    seed: Mapping[str, Any],
    *,
    fetcher: JsonFetcher,
) -> dict[str, Any]:
    query = str(seed["rxnorm_query"]).strip()
    lookup_url = f"{RXNORM_BASE_URL}/rxcui.json?{urlencode({'name': query, 'search': 2})}"
    lookup = fetcher(lookup_url)
    id_group = lookup.get("idGroup", {})
    rxcuids = _string_list(id_group.get("rxnormId", [])) if isinstance(id_group, dict) else []
    if not rxcuids:
        raise ValueError(f"RxNorm returned no RXCUI for {query!r}.")

    rxcui = rxcuids[0]
    properties_url = f"{RXNORM_BASE_URL}/rxcui/{rxcui}/properties.json"
    properties_payload = fetcher(properties_url)
    properties = properties_payload.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    return {
        "query": query,
        "rxcui": rxcui,
        "candidate_rxcuids": rxcuids,
        "name": str(properties.get("name", query)),
        "synonym": str(properties.get("synonym", "")),
        "tty": str(properties.get("tty", "")),
        "lookup_url": lookup_url,
        "properties_url": properties_url,
    }


def _openfda_query_url(generic_name: str, product_type: str) -> str:
    search = (
        f'openfda.generic_name.exact:"{generic_name}" AND '
        f'openfda.product_type.exact:"{product_type}"'
    )
    return f"{OPENFDA_LABEL_URL}?{urlencode({'search': search, 'sort': 'effective_time:desc', 'limit': 10})}"


def _openfda_values(record: Mapping[str, Any], key: str) -> list[str]:
    openfda = record.get("openfda", {})
    if not isinstance(openfda, dict):
        return []
    return _string_list(openfda.get(key, []))


def _section_values(record: Mapping[str, Any], key: str) -> list[str]:
    value = record.get(key, [])
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return _string_list(value)


def _route_match(record: Mapping[str, Any], route_preferences: Iterable[str]) -> bool:
    wanted = {str(item).strip().upper() for item in route_preferences if str(item).strip()}
    if not wanted:
        return True
    available = {item.upper() for item in _openfda_values(record, "route")}
    return bool(wanted.intersection(available))


def _record_score(
    record: Mapping[str, Any],
    *,
    route_preferences: Iterable[str],
) -> tuple[int, int, int, int]:
    route_score = int(_route_match(record, route_preferences))
    core_count = sum(bool(_section_values(record, key)) for key in CORE_LABEL_SECTIONS)
    try:
        effective_time = int(str(record.get("effective_time", "0")))
    except ValueError:
        effective_time = 0
    try:
        version = int(str(record.get("version", "0")))
    except ValueError:
        version = 0
    return route_score, core_count, effective_time, version


def _select_openfda_label(
    seed: Mapping[str, Any],
    *,
    fetcher: JsonFetcher,
    api_key: str | None,
) -> tuple[dict[str, Any], list[str]]:
    generic_names = _string_list(seed.get("openfda_generic_names", []))
    product_types = _string_list(seed.get("product_types", DEFAULT_PRODUCT_TYPES))
    route_preferences = _string_list(seed.get("route_preferences", []))
    candidates: list[tuple[Mapping[str, Any], str]] = []
    query_urls: list[str] = []

    for generic_name in generic_names:
        for product_type in product_types:
            public_url = _openfda_query_url(generic_name, product_type)
            query_urls.append(public_url)
            payload = fetcher(_append_api_key(public_url, api_key))
            results = payload.get("results", [])
            if not isinstance(results, list):
                continue
            for record in results:
                if isinstance(record, dict):
                    candidates.append((record, public_url))

    if not candidates:
        raise ValueError("openFDA returned no exact generic-name label records.")

    matching_route = [
        item for item in candidates if _route_match(item[0], route_preferences)
    ]
    if route_preferences and not matching_route:
        raise ValueError(
            "openFDA returned labels, but none matched the reviewed route: "
            f"{', '.join(route_preferences)}."
        )
    eligible = matching_route or candidates
    record, query_url = max(
        eligible,
        key=lambda item: _record_score(
            item[0], route_preferences=route_preferences
        ),
    )
    if sum(bool(_section_values(record, key)) for key in CORE_LABEL_SECTIONS) < 2:
        raise ValueError("Selected openFDA label lacks sufficient core sections.")

    sections = {
        key: _section_values(record, key)
        for key in LABEL_SECTIONS
        if _section_values(record, key)
    }
    set_id = str(record.get("set_id", "")).strip()
    openfda = record.get("openfda", {})
    openfda = openfda if isinstance(openfda, dict) else {}
    selected = {
        "generic_names": _openfda_values(record, "generic_name"),
        "brand_names": _openfda_values(record, "brand_name"),
        "substance_names": _openfda_values(record, "substance_name"),
        "manufacturer_names": _openfda_values(record, "manufacturer_name"),
        "product_types": _openfda_values(record, "product_type"),
        "routes": _openfda_values(record, "route"),
        "rxcuis": _openfda_values(record, "rxcui"),
        "application_numbers": _openfda_values(record, "application_number"),
        "spl_set_ids": _openfda_values(record, "spl_set_id"),
        "set_id": set_id,
        "spl_id": str(record.get("id", "")).strip(),
        "version": str(record.get("version", "")).strip(),
        "effective_time": str(record.get("effective_time", "")).strip(),
        "query_url": query_url,
        "dailymed_url": (
            f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
            if set_id
            else ""
        ),
        "route_preference_matched": _route_match(record, route_preferences),
        "section_names": sorted(sections),
    }
    selected["selected_metadata_sha256"] = _sha256(selected)
    return {"metadata": selected, "sections": sections}, query_urls


def _aliases(seed: Mapping[str, Any], rxnorm: Mapping[str, Any], label: Mapping[str, Any]) -> list[str]:
    metadata = label.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    values = [
        str(seed.get("drug_id", "")),
        str(seed.get("display_name", "")),
        str(seed.get("rxnorm_query", "")),
        str(rxnorm.get("name", "")),
        str(rxnorm.get("synonym", "")),
        *_string_list(seed.get("aliases", [])),
        *_string_list(metadata.get("generic_names", [])),
        *_string_list(metadata.get("brand_names", [])),
        *_string_list(metadata.get("substance_names", [])),
    ]
    deduplicated: dict[str, str] = {}
    for value in values:
        text = value.strip()
        if text:
            deduplicated.setdefault(text.casefold(), text)
    return sorted(deduplicated.values(), key=lambda item: (item.casefold(), item))


def build_database(
    seed_payload: Mapping[str, Any],
    *,
    fetcher: JsonFetcher,
    api_key: str | None = None,
    selected_drug_ids: set[str] | None = None,
    delay_seconds: float = 0.0,
) -> dict[str, Any]:
    seeds = _validate_seed(seed_payload)
    if selected_drug_ids:
        seeds = [item for item in seeds if str(item["drug_id"]) in selected_drug_ids]
        unknown = sorted(selected_drug_ids.difference(str(item["drug_id"]) for item in seeds))
        if unknown:
            raise ValueError(f"Unknown requested drug_id values: {', '.join(unknown)}")

    try:
        rxnorm_version = dict(fetcher(f"{RXNORM_BASE_URL}/version.json"))
    except RuntimeError as exc:
        rxnorm_version = {"retrieval_error": str(exc)}

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, seed in enumerate(seeds, start=1):
        drug_id = str(seed["drug_id"])
        print(f"[{index:02d}/{len(seeds):02d}] {drug_id}", flush=True)
        try:
            rxnorm = _rxnorm_lookup(seed, fetcher=fetcher)
            if delay_seconds:
                time.sleep(delay_seconds)
            label, query_urls = _select_openfda_label(
                seed,
                fetcher=fetcher,
                api_key=api_key,
            )
            record = {
                "drug_id": drug_id,
                "display_name": str(seed["display_name"]),
                "aliases": _aliases(seed, rxnorm, label),
                "rxnorm": rxnorm,
                "openfda": label["metadata"],
                "label_sections": label["sections"],
                "attempted_openfda_query_urls": query_urls,
            }
            record["record_sha256"] = _sha256(record)
            records.append(record)
        except (RuntimeError, ValueError) as exc:
            failures.append({"drug_id": drug_id, "error": str(exc)})
        if delay_seconds:
            time.sleep(delay_seconds)

    records.sort(key=lambda item: str(item["drug_id"]))
    failures.sort(key=lambda item: item["drug_id"])
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "schema_version": "1.0",
        "dataset_id": "goa_openfda_label_evidence_v1",
        "generated_at": generated_at,
        "seed_dataset_id": str(seed_payload.get("dataset_id", "unknown")),
        "scope_note": str(seed_payload.get("scope_note", "")),
        "source_versions": {
            "rxnorm": rxnorm_version,
            "openfda_retrieved_at": generated_at,
        },
        "sources": [
            {
                "source_id": "RXNORM_API_CURRENT",
                "url": "https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html",
                "role": "Normalized medication names and RXCUIs",
            },
            {
                "source_id": "OPENFDA_LABEL_CURRENT",
                "url": "https://open.fda.gov/apis/drug/label/",
                "role": "Structured Product Label metadata and sections",
            },
            {
                "source_id": "DAILYMED_SPL_CURRENT",
                "url": "https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm",
                "role": "Human-readable SPL record links by SET ID",
            },
        ],
        "coverage": {
            "requested_medication_count": len(seeds),
            "label_record_count": len(records),
            "failure_count": len(failures),
            "complete": not failures,
            "comprehensive_drug_coverage": False,
        },
        "records_sha256": _sha256(records),
        "records": records,
        "failures": failures,
        "safety_contract": {
            "label_selection_is_product_specific": True,
            "negative_lookup_excludes_contraindication": False,
            "patient_specific_dose_selected": False,
            "treatment_course_selected": False,
            "allows_automatic_medication_change": False,
        },
        "nlm_attribution": (
            "This product uses publicly available data courtesy of the U.S. "
            "National Library of Medicine (NLM), National Institutes of Health, "
            "Department of Health and Human Services; NLM is not responsible for "
            "the product and does not endorse or recommend this or any other product."
        ),
    }


def _write_database(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the versioned RxNorm/openFDA label evidence database."
    )
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--drug-ids",
        default="",
        help="Optional comma-separated drug_id subset for debugging.",
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.3,
        help="Delay between official API requests; default respects unauthenticated limits.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write a database even when one or more labels cannot be resolved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_payload = _read_json(args.seed)
    selected = {
        item.strip() for item in args.drug_ids.split(",") if item.strip()
    }
    fetcher = lambda url: _fetch_json(url, timeout=args.timeout)
    database = build_database(
        seed_payload,
        fetcher=fetcher,
        api_key=os.environ.get("OPENFDA_API_KEY"),
        selected_drug_ids=selected or None,
        delay_seconds=max(0.0, args.delay_seconds),
    )
    failures = database["failures"]
    if failures and not args.allow_partial:
        for failure in failures:
            print(f"FAILED {failure['drug_id']}: {failure['error']}")
        print("Database not written. Re-run with --allow-partial only for diagnosis.")
        return 2

    _write_database(args.output, database)
    coverage = database["coverage"]
    print(
        f"Wrote {args.output} with {coverage['label_record_count']}/"
        f"{coverage['requested_medication_count']} label records."
    )
    for failure in failures:
        print(f"UNRESOLVED {failure['drug_id']}: {failure['error']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
