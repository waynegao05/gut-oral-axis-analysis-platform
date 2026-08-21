# Platform 0.4.1

Platform 0.4.1 advances the Windows desktop platform while preserving the
existing WebUI and browser workflow.

## Included

- A compact, tabbed analysis form that keeps the existing fields and behavior.
- A modular TypeScript frontend replacing the legacy monolithic application file.
- Correctly bound browser requests so imported JSON can be analyzed normally.
- A WinUI 3 acrylic title bar and application icon for the desktop host.
- Inno Setup and WiX definitions for portable, per-user, and managed deployment.
- WebView2 runtime detection, local-data-preserving uninstall behavior, checksums,
  and package smoke-test support.
- An explicit SQLite native bundle update to `2.1.12`.

## Distribution Boundary

This GitHub release contains source code only. Generated output directories,
plotting R programs, trained models, model weights, training caches, portable
archives, and installers containing those artifacts are not tracked or attached.

The desktop application remains a research-use decision-support tool. It is not
a screening or diagnostic device, does not produce medical orders, and must not
replace the judgment of qualified clinical or pharmacy personnel.

## Current Limitations

- Public binaries are not yet code-signed.
- A clean Windows machine qualification run is still required before broad
  external deployment.
- Offline inference requires separately governed model runtime artifacts.
