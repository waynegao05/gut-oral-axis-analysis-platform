# Archived vanilla JavaScript frontend

This directory preserves the web frontend immediately before the TypeScript
migration started on 2026-08-14.

## Scope

- `static/app.js`: browser behavior and API rendering.
- `static/app.css`: frontend styles.
- `templates/index.html`: Flask page template.

The archive is a rollback snapshot only. Active application files remain under
the repository-level `static/` and `templates/` directories.

## Model boundary

This snapshot used the existing AC-ICAM V8 PFS web flow. The later oral-only
adenoma model is a separate research endpoint and must not be numerically mixed
with the PFS result.

## Restore locally

Copy the three archived files back to their corresponding repository paths.
Do not restore only one file because element IDs and JavaScript behavior are a
matched set.
