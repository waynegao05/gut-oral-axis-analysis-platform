from __future__ import annotations

from research.build_drug_knowledge_v1 import build_database


def test_builder_combines_rxnorm_and_exact_openfda_label() -> None:
    seed = {
        "dataset_id": "test_seed",
        "scope_note": "test",
        "medications": [
            {
                "drug_id": "metformin",
                "display_name": "Metformin",
                "rxnorm_query": "metformin",
                "openfda_generic_names": ["METFORMIN HYDROCHLORIDE"],
                "route_preferences": ["ORAL"],
                "aliases": ["二甲双胍"],
            }
        ],
    }

    def fetcher(url: str):
        if url.endswith("/version.json"):
            return {"version": "test"}
        if "/rxcui.json?" in url:
            return {"idGroup": {"rxnormId": ["6809"]}}
        if url.endswith("/rxcui/6809/properties.json"):
            return {
                "properties": {
                    "rxcui": "6809",
                    "name": "metformin",
                    "synonym": "",
                    "tty": "IN",
                }
            }
        if "api.fda.gov/drug/label.json" in url:
            return {
                "results": [
                    {
                        "set_id": "test-set-id",
                        "id": "test-spl-id",
                        "version": "2",
                        "effective_time": "20260718",
                        "indications_and_usage": ["Test indication."],
                        "dosage_and_administration": ["Test label directions."],
                        "contraindications": ["Test contraindication."],
                        "drug_interactions": ["Test interaction section."],
                        "openfda": {
                            "generic_name": ["METFORMIN HYDROCHLORIDE"],
                            "brand_name": ["Test Brand"],
                            "substance_name": ["METFORMIN HYDROCHLORIDE"],
                            "manufacturer_name": ["Test Manufacturer"],
                            "product_type": ["HUMAN PRESCRIPTION DRUG"],
                            "route": ["ORAL"],
                            "rxcui": ["6809"],
                            "spl_set_id": ["test-set-id"],
                        },
                    }
                ]
            }
        raise AssertionError(f"Unexpected URL: {url}")

    database = build_database(seed, fetcher=fetcher)

    assert database["coverage"]["complete"] is True
    assert database["coverage"]["label_record_count"] == 1
    record = database["records"][0]
    assert record["rxnorm"]["rxcui"] == "6809"
    assert record["openfda"]["set_id"] == "test-set-id"
    assert record["openfda"]["route_preference_matched"] is True
    assert "dosage_and_administration" in record["label_sections"]
    assert "Test Brand" in record["aliases"]
    assert len(record["record_sha256"]) == 64
