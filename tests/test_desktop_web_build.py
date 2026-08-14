from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re

from scripts.build_desktop_web import ROOT, build_desktop_web


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

    css_source = ROOT / "static" / "app.css"
    javascript_source = ROOT / "static" / "generated" / "app.js"
    assert (result.output_directory / "assets" / "app.css").read_bytes() == css_source.read_bytes()
    assert (
        result.output_directory / "assets" / "app.js"
    ).read_bytes() == javascript_source.read_bytes()

    manifest = json.loads(result.manifest_file.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["entrypoint"] == "index.html"
    assert manifest["internal_oral_adenoma_enabled"] is True
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
