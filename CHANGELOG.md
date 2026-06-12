# Changelog

## v0.1.0rc1 — 2026-06-12

Initial release candidate.

- Self-contained inference for LUMINA OPF models (HGT architecture)
- Flexible ingestion: Python dict, JSON file/string, MATPOWER `.m`
- HuggingFace model loading via `Modeler.from_pretrained()` or `Modeler.load_model()`
- Eager validation of config / checkpoint shape mismatches and required HGT output node types
