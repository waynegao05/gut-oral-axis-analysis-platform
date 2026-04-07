from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict


def export_report(report: Dict[str, object], output_dir: str = "outputs") -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("report_%Y%m%d_%H%M%S.json")
    file_path = path / filename
    file_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(file_path)
