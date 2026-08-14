from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    AC_ICAM_V8_RELEASE_NAME,
    APP_NAME,
    ENABLE_INTERNAL_ORAL_ADENOMA,
    RESEARCH_MODEL_RELEASE_NAME,
    TEMPORAL_TOPOLOGY_RELEASE_NAME,
    WEB_MODEL_BACKEND,
)


DEFAULT_OUTPUT = ROOT / "frontend" / "dist"


@dataclass(frozen=True)
class DesktopWebBuild:
    output_directory: Path
    index_file: Path
    manifest_file: Path


def _active_release_name() -> str:
    return {
        "ac_icam_v8": AC_ICAM_V8_RELEASE_NAME,
        "temporal_topology": TEMPORAL_TOPOLOGY_RELEASE_NAME,
        "legacy_cox": RESEARCH_MODEL_RELEASE_NAME,
    }.get(WEB_MODEL_BACKEND, WEB_MODEL_BACKEND)


def _asset_url(_: str, *, filename: str) -> str:
    mappings = {
        "app.css": "assets/app.css",
        "generated/app.js": "assets/app.js",
    }
    try:
        return mappings[filename]
    except KeyError as exc:
        raise ValueError(f"Desktop WebUI does not allow unknown asset: {filename}") from exc


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_desktop_web(
    output_directory: Path = DEFAULT_OUTPUT,
    *,
    app_name: str = APP_NAME,
    model_release: str | None = None,
    internal_oral_adenoma_enabled: bool = ENABLE_INTERNAL_ORAL_ADENOMA,
) -> DesktopWebBuild:
    template_file = ROOT / "templates" / "index.html"
    css_source = ROOT / "static" / "app.css"
    javascript_source = ROOT / "static" / "generated" / "app.js"
    for source in (template_file, css_source, javascript_source):
        if not source.is_file():
            raise FileNotFoundError(f"Required WebUI source is missing: {source}")

    output_directory = output_directory.resolve()
    assets_directory = output_directory / "assets"
    assets_directory.mkdir(parents=True, exist_ok=True)

    environment = Environment(
        loader=FileSystemLoader(str(template_file.parent)),
        autoescape=select_autoescape(("html",)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    environment.globals["url_for"] = _asset_url
    rendered = environment.get_template(template_file.name).render(
        app_name=app_name,
        model_release=model_release or _active_release_name(),
        web_model_backend=WEB_MODEL_BACKEND,
        internal_oral_adenoma_enabled=internal_oral_adenoma_enabled,
    )
    if "{{" in rendered or "{%" in rendered or "{#" in rendered:
        raise RuntimeError("Desktop HTML still contains unresolved Jinja syntax.")

    index_file = output_directory / "index.html"
    css_target = assets_directory / "app.css"
    javascript_target = assets_directory / "app.js"
    index_file.write_text(rendered, encoding="utf-8", newline="\n")
    shutil.copyfile(css_source, css_target)
    shutil.copyfile(javascript_source, javascript_target)

    manifest = {
        "schema_version": 1,
        "application_name": app_name,
        "frontend_version": "2.0.0",
        "model_release": model_release or _active_release_name(),
        "web_model_backend": WEB_MODEL_BACKEND,
        "internal_oral_adenoma_enabled": internal_oral_adenoma_enabled,
        "entrypoint": "index.html",
        "files": {
            "index.html": _digest(index_file),
            "assets/app.css": _digest(css_target),
            "assets/app.js": _digest(javascript_target),
        },
    }
    manifest_file = output_directory / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return DesktopWebBuild(output_directory, index_file, manifest_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the existing WebUI as offline WebView2 assets.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--app-name", default=APP_NAME)
    parser.add_argument("--model-release", default=None)
    oral_group = parser.add_mutually_exclusive_group()
    oral_group.add_argument(
        "--enable-internal-oral-adenoma",
        action="store_true",
        dest="oral_enabled",
    )
    oral_group.add_argument(
        "--disable-internal-oral-adenoma",
        action="store_false",
        dest="oral_enabled",
    )
    parser.set_defaults(oral_enabled=ENABLE_INTERNAL_ORAL_ADENOMA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_desktop_web(
        args.output_dir,
        app_name=args.app_name,
        model_release=args.model_release,
        internal_oral_adenoma_enabled=args.oral_enabled,
    )
    print(f"Desktop WebUI written to {result.output_directory}")


if __name__ == "__main__":
    main()
