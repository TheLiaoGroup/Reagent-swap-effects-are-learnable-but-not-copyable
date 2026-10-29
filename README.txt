Copying measured reagent effects can mislead synthesis
=================================================

Files
-----
data/ contains the analysis CSV tables and their reading definitions.
data/tables.json maps original logical source names to the supplied CSVs.
Source Data's source_analysis_relative_path column retains original logical
provenance aliases; these identify sources rather than bundled workbooks.
Resolve each alias's filename through data/tables.json.
Input/EXCLUDE_wells.csv lists the excluded measurement identifiers.
Free-text notes in the condition table and one worksheet name were
translated from Chinese to English; no measured value was changed.
Analysis/scripts/ contains the three analyses and figure generation code.
Results/ contains numerical summaries and figure source tables.
Submission/02_Figures/ChemDraw/ contains chemical drawing assets.
requirements.txt lists the Python dependencies. LICENSE covers code;
DATA_LICENSE covers data.

Running
-------
Use CPython 3.13.3 and create a virtual environment:
    python -m venv .venv
    ./.venv/Scripts/Activate.ps1
    python -m pip install -r requirements.txt

On macOS or Linux activate with: source .venv/bin/activate
Run from this folder, in order:
    python Analysis/scripts/reproduce.py --list
    python Analysis/scripts/reproduce.py --stage analysis
    python Analysis/scripts/reproduce.py --stage summaries
    python Analysis/scripts/reproduce.py --stage figures

The analysis stage runs sequentially and can take several hours. Run one
analysis process at a time. Outputs and logs are written under Analysis/.
To use another copy of the supplied data folder, set M1_TABLES_DIR.

Figures use Matplotlib, PyMuPDF and Arial. Chemical vectors are included.
Drawing an additional native chemical structure requires Windows,
ChemDraw 22 and pywin32. Source Data export uses pandas and openpyxl.
