# GitHub Release Package Selection Notes

This release package was curated from the thesis working directories and is aligned with the final thesis model run: `run_111`.

Selection principles:

1. Keep the core code needed to explain the thesis methods and verify the final results.
2. Keep cleaned, public, compact modeling CSV files.
3. Keep the final thesis figures and key run111 model result tables.
4. Exclude raw large data, virtual environments, caches, logs, backups, drafts, and unrelated intermediate runs.
5. Use "Dongting Lake Ecological Economic Zone" as the study-area description.

Excluded examples:

- Large CLCD land-cover `.tif` files
- Large MODIS archives, yearbook PDFs, and ArcGIS project files
- `.venv/`, `node_modules/`, `__pycache__/`
- `.joblib`, `.keras`, `.h5`, `.pkl`, `.pt`, `.pth` model files
- Word/PDF thesis drafts, defense files, and formatting-check renders
- run112/run113 supplemental outputs not used as the final thesis model run
- Earlier standard-figure folders whose figures differ from the final thesis

The goal is to provide a clean thesis code/data appendix, not a full backup of the local working drive.
