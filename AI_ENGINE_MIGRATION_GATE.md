# Python AI Engine Migration Gate

## Status

This boundary is now integrated into the Windows GUI while remaining additive.
It does not replace the current Flask WebUI and does not change any model,
feature engineering, pharmacy rule, or risk definition.

The existing application remains available through:

```powershell
python enhanced_app.py
```

The new local API calls the same `src.pipeline.run_pipeline()` entry point but
does not write reports to `outputs/`. File persistence, SQLite, PDF, and print
remain outside Python and are owned by the C# host.

## Added Boundary

```text
FastAPI /api/v1
  -> AIService
     -> existing standardizer and validators
     -> existing src.pipeline.run_pipeline()
     -> existing model and pharmacy modules
```

Endpoints:

- `GET /api/v1/health`
- `POST /api/v1/standardize`
- `POST /api/v1/predict`
- `POST /api/v1/analyze`
- `GET /api/v1/oral-adenoma/schema`
- `POST /api/v1/oral-adenoma/analyze`

Every endpoint requires a per-launch `X-GOA-Engine-Token`. The runtime rejects
`0.0.0.0` and other non-loopback bind addresses. API documentation endpoints
are disabled in the local production app.

## Development Start

Install the additive API dependencies into the intended full model environment:

```powershell
python -m pip install -r requirements-ai-engine.txt -r requirements-dev.txt
```

Generate a temporary development token and start the engine:

```powershell
$bytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$env:GOA_ENGINE_TOKEN = [Convert]::ToHexString($bytes)
python -m ai_engine --host 127.0.0.1 --port 8766
```

The token belongs to the current process session only. Do not store it in the
repository, application settings, logs, or command-line arguments.

## Required Gate Before Flask or WebUI Switching

The current Flask route or WebUI transport must not be switched to the new API
until all of the following pass in the intended desktop Python runtime:

1. Existing Flask endpoint tests pass unchanged.
2. `tests/test_ai_engine_service.py` passes.
3. `tests/test_ai_engine_api.py` passes with FastAPI installed.
4. Complete-oncology V8 input produces the same report through the current
   pipeline and `AIService`.
5. The full five-microbe/no-oncology path reports `general_risk` as available
   and completes a real inference smoke test.
6. Missing optional dependencies return `CAPABILITY_UNAVAILABLE`, not a raw
   Python traceback or a generic HTML 500 page.
7. `/api/v1/analyze` creates no report file and contains no `saved_to` field.
8. The local API refuses non-loopback bind addresses and invalid tokens.
9. The current HTML, CSS, generated JavaScript, DOM IDs, and screenshots remain
   unchanged.

## Runtime Resolution

The ordinary `D:\Anaconda3` environment is still not the desktop runtime. The
Windows build now creates an isolated, one-directory `goa-ai-engine.exe` bundle
with PyTorch CPU, PyG, the V8 release, the temporal-topology fallback and the
explicit model-artifact whitelist. The desktop host prefers this executable and
does not require an installed Python interpreter on the target machine.

Missing packaged capabilities must still return an explicit unavailable state;
the host and Engine must never silently substitute another score.

## Verification Snapshot

Preparation checks completed on 2026-08-14:

- 13 service/runtime tests passed without FastAPI installed;
- 7 FastAPI contract tests passed with temporary, isolated API dependencies;
- all 19 existing Flask and oral-adenoma endpoint tests passed when the local
  CPU research dependencies were made available to the test interpreter;
- a real hidden Uvicorn process passed token rejection, authenticated health,
  complete-oncology V8, and no-oncology five-microbe inference smoke tests;
- the process smoke confirmed lazy `general_risk` load state and created no new
  `outputs/report_*.json` file;
- the new service produced the exact current pipeline report for a complete
  AC-ICAM V8 input;
- the packaged `goa-ai-engine.exe` passed a C#-managed health, standardization,
  real analysis, response-correlation and owned-process shutdown test;
- the self-contained WinUI/WebView2 release passed a hidden startup test with
  the bundled Engine, SQLite initialization and ordered shutdown;
- the full Python suite reached 346 passing tests; one pre-existing V6 archive
  hash assertion remains and is unrelated to the GUI migration;
- `src/` model-core files remain unchanged by the migration.

The combined development interpreter remains a verification workaround and is
not copied into the release. The isolated PyInstaller environment and generated
dependency inventory define the desktop Engine contents.

## Integration Status

The API gate, dual Web/desktop transport, WinUI 3 host, WebView2 bridge, SQLite
foundation and portable release build are green. Remaining external release
gates are a clean Windows target-machine qualification, installer/signing
decision, code-signing certificate, real device protocol and any clinical or
regulated-use validation.
