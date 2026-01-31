# Lariat Bible — Project Quick Start

Purpose
- Single place to manage vendor pricing, invoices, recipe costing, inventory, kitchen ops and reports.

Quick Start
1. Create virtualenv and install deps:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Run initial setup:
   python setup.py

3. Run the app (dev):
   python app.py
   # open http://127.0.0.1:5000

Pre-commit hooks (recommended)
- Install and enable:
  python3 -m pip install --user pre-commit black flake8
  pre-commit install
- Run before push:
  pre-commit run --all-files

Where to find things (top-level)
- 01_VENDOR_MANAGEMENT/    → Vendor comparisons (SYSCO vs Shamrock)
- 02_ORDER_GUIDES/         → Order guides and par levels
- 03_INVOICES/             → Invoices + Invoice_Images/
- 04_RECIPE_COSTING/       → Recipe cost calculators
- 05_INVENTORY/            → Stock tracking
- 06_KITCHEN_OPERATIONS/   → Daily procedures & station checklists
- 07_STAFF_SCHEDULING/     → Employee schedules
- 08_MENU_ANALYTICS/       → Sales & product mix
- 09_PURCHASE_HISTORY/     → Historical vendor data
- 10_TEMPLATES/            → CSV & form templates
- 11_DOCUMENTS/            → Training & SOPs
- 12_WORKING_FILES/        → WIP files
- 13_SCRIPTS/              → Python automation scripts
- _ARCHIVE/                → Duplicates and unrelated files

Recommended next steps
- Review _ARCHIVE/Duplicates_To_Delete/ and delete after verification.
- Update any scripts in 13_SCRIPTS/ that reference old flat paths.
- Use 10_TEMPLATES/CSV_TEMPLATES_README.md for import templates.

Useful commands (git + validation)
- Validate repo paths:
  ./scripts/validate_paths.py
- Stage/commit/push helper:
  ./scripts/push_to_github.sh "chore: reorganize files"

Notes
- Keep sensitive files out of git (.env, invoices with real PII). See .gitignore.
- Use Git instead of filename suffixes (_1, _2) for versions.
