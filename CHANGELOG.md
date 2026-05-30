# Changelog

## [v7_fixed_v6] - 2026-04-04
### Added/Fixed
- Centralized PROJECT_ROOT path management
- .env loader for secrets
- FiLM fix for dynamic conditioning
### Metrics
- Test AUROC: 0.9626
- Hit Rate@50: 55.7%


All notable changes to PromptGFM-Bio are documented in this file.

The format is based on Keep a Changelog and this project uses Semantic Versioning.

## [Unreleased]

### Changed
- Documentation refreshed to align with workstation notebook workflow:
  - notebooks/workstation_training_v7_fixed_v5_Clean_Completed.ipynb
- Added latest run source-of-truth document:
  - LATEST_EVALUATION_SUMMARY.md
- Replaced stale February 2026 status/progress docs with current April 2026 run status.
- Updated setup/training/gpu/cloud/deployment/troubleshooting guides to reference configs/workstation_config.yaml and results/evaluation_results.json.

### Added
- Current baseline metrics snapshot in docs:
  - AUROC 0.8130
  - AUPR 0.4618
  - Precision@10 1.0000
  - NDCG@100 0.9286

---

## [1.0.0] - 2026-02-19

### Added
- Initial public release content and project scaffolding.

---

[Unreleased]: https://github.com/pes1ug23am910/PromptGFM-Bio/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/pes1ug23am910/PromptGFM-Bio/releases/tag/v1.0.0
