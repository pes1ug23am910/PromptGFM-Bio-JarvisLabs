# Contributing to PromptGFM-Bio

Thanks for contributing.

This guide is aligned with the current repository state as of 2026-04-01.

## 1. Before You Start

- Read README.md, TRAINING_GUIDE.md, and LATEST_EVALUATION_SUMMARY.md.
- For large changes, open an issue first and describe scope.
- Keep changes focused: one PR for one problem area.

## 2. Current Known Limitations (Please Read)

Contributors should be aware of active limitations in the validated run:

- No gene-gene message passing was active in the validated baseline run.
- Baseline constructor/signature mismatch exists in current baseline path.
- Cross-attention path has tensor-shape mismatch risk.
- Runtime dependency drift exists between requirements pins and validated runtime.

If your PR touches these areas, include explicit tests and before/after notes.

## 3. Development Setup

```bash
git clone https://github.com/pes1ug23am910/PromptGFM-Bio.git
cd PromptGFM-Bio
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

Optional runtime alignment with validated workstation stack:

```bash
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade --force-reinstall -f https://data.pyg.org/whl/torch-2.6.0+cu124.html torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
```

## 4. Branch and Commit Workflow

```bash
git checkout -b feature/short-description
```

Commit style:

- feat: new behavior
- fix: bug fix
- docs: docs only
- refactor: internal rewrite without behavior change
- test: tests only

Example:

```text
fix: align baseline model constructor arguments
```

## 5. Coding Standards

- Python style: PEP 8, keep lines readable.
- Add type hints for public functions.
- Add concise docstrings for non-trivial logic.
- Avoid changing unrelated files in the same PR.

## 6. Testing Expectations

Run these checks before opening a PR:

```bash
python scripts/verify_setup.py
pytest tests/
python scripts/train.py --help
python scripts/evaluate.py --help
```

Current test status note:

- tests/test_models.py is still a minimal placeholder test file.
- If you touch model code, add or update meaningful tests in the same PR.

## 7. Documentation Requirements

When behavior changes, update the related live docs in the same PR:

- README.md
- SETUP.md
- QUICKSTART.md
- TRAINING_GUIDE.md
- PREPROCESSING_GUIDE.md
- DATA_DOWNLOAD_GUIDE.md
- docs/TROUBLESHOOTING.md
- LATEST_EVALUATION_SUMMARY.md

Do not add new operational instructions to historical archive docs.

## 8. Pull Request Checklist

- [ ] Scope is clear and focused.
- [ ] Commands and paths are valid in the current repository.
- [ ] Tests and checks run successfully.
- [ ] Docs are updated for behavior changes.
- [ ] Limitations or tradeoffs are documented.

## 9. Code of Conduct

Be respectful, constructive, and technical in all discussions.
