from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re

from scripts.build_desktop_web import (
    DESKTOP_STYLESHEET_NAME,
    ROOT,
    build_desktop_web,
    resolve_stylesheet_source,
)


def _ids(text: str) -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', text))


def test_desktop_web_build_uses_the_existing_ui_sources(tmp_path: Path) -> None:
    result = build_desktop_web(
        tmp_path / "desktop-web",
        app_name="Desktop Test",
        model_release="model-test",
        internal_oral_adenoma_enabled=True,
    )

    template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    rendered = result.index_file.read_text(encoding="utf-8")
    assert "{{" not in rendered
    assert "{%" not in rendered
    assert "{#" not in rendered
    assert "<title>Desktop Test</title>" in rendered
    assert "model-test" in rendered
    assert 'http-equiv="Content-Security-Policy"' in rendered
    assert "script-src 'self'" in rendered
    assert "connect-src 'self'" in rendered
    assert _ids(rendered) == _ids(template)
    assert 'href="assets/app.css"' in rendered
    assert 'src="assets/app.js"' in rendered

    css_source = resolve_stylesheet_source()
    javascript_source = ROOT / "static" / "generated" / "app.js"
    assert (result.output_directory / "assets" / "app.css").read_bytes() == css_source.read_bytes()
    assert (
        result.output_directory / "assets" / "app.js"
    ).read_bytes() == javascript_source.read_bytes()

    manifest = json.loads(result.manifest_file.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["entrypoint"] == "index.html"
    assert manifest["internal_oral_adenoma_enabled"] is True
    assert manifest["stylesheet_source"] == css_source.name
    assert manifest["files"]["assets/app.css"] == sha256(
        css_source.read_bytes()
    ).hexdigest()


def test_desktop_web_build_can_preserve_the_disabled_optional_panel(
    tmp_path: Path,
) -> None:
    result = build_desktop_web(
        tmp_path / "desktop-web",
        internal_oral_adenoma_enabled=False,
    )
    rendered = result.index_file.read_text(encoding="utf-8")
    assert 'id="oral-adenoma-panel"' not in rendered


def test_desktop_web_build_prefers_the_fluent_desktop_stylesheet(tmp_path: Path) -> None:
    """桌面 GUI 必须使用 Windows 11 Fluent 样式，而不是浏览器 WebUI 的样式。"""
    desktop_stylesheet = ROOT / "static" / DESKTOP_STYLESHEET_NAME
    assert desktop_stylesheet.is_file()
    assert resolve_stylesheet_source() == desktop_stylesheet

    web_stylesheet = ROOT / "static" / "app.css"
    result = build_desktop_web(tmp_path / "desktop-web")
    packaged = (result.output_directory / "assets" / "app.css").read_bytes()
    assert packaged == desktop_stylesheet.read_bytes()
    assert packaged != web_stylesheet.read_bytes()


def test_desktop_web_build_accepts_an_explicit_stylesheet(tmp_path: Path) -> None:
    override = tmp_path / "override.css"
    override.write_text(":root { --probe: 1; }\n", encoding="utf-8")
    result = build_desktop_web(tmp_path / "desktop-web", stylesheet_source=override)
    assert result.stylesheet_source == override
    assert (
        result.output_directory / "assets" / "app.css"
    ).read_text(encoding="utf-8") == ":root { --probe: 1; }\n"


def test_input_form_uses_accessible_compact_sections() -> None:
    template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    tab_pairs = (
        ("form-tab-microbiome", "form-panel-microbiome"),
        ("form-tab-clinical", "form-panel-clinical"),
        ("form-tab-health", "form-panel-health"),
        ("form-tab-medication", "form-panel-medication"),
    )

    assert 'id="analysis-form-tabs" role="tablist"' in template
    assert template.count("data-form-tab") == len(tab_pairs)
    assert template.count('role="tabpanel"') == len(tab_pairs)
    for tab_id, panel_id in tab_pairs:
        assert template.count(f'id="{tab_id}"') == 1
        assert template.count(f'id="{panel_id}"') == 1
        assert f'aria-controls="{panel_id}"' in template
        assert f'aria-labelledby="{tab_id}"' in template

    retained_field_ids = (
        "microbe-Fusobacterium",
        "microbe-Porphyromonas",
        "microbe-Prevotella",
        "microbe-Streptococcus",
        "microbe-Lactobacillus",
        "clinical-age",
        "clinical-sex",
        "clinical-stage",
        "clinical-path-t",
        "clinical-path-n",
        "clinical-path-m",
        "clinical-tumor-location",
        "clinical-tumor-morphology",
        "clinical-icr-score",
        "clinical-bmi",
        "clinical-smoking",
        "clinical-family-history",
        "metabolite-bile-acids",
        "metabolite-scfa",
        "metabolite-tryptophan",
        "metadata-current-medications",
        "metadata-drug-allergies",
        "metadata-suspected-condition",
        "metadata-recent-antibiotics",
        "metadata-recent-probiotics",
        "metadata-renal-impairment",
        "metadata-hepatic-impairment",
        "metadata-pregnancy",
        "analyze-form",
        "reset-form",
    )
    for field_id in retained_field_ids:
        assert template.count(f'id="{field_id}"') == 1

    for stylesheet_name in ("app.css", "app.desktop.css"):
        stylesheet = (ROOT / "static" / stylesheet_name).read_text(encoding="utf-8")
        assert ".form-section-tabs" in stylesheet
        assert ".form-tab-panel[hidden]" in stylesheet
