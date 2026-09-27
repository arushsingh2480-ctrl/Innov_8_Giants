GIANTS — CORPORATE HEIST RUNNABLE PROJECT

1) Extract this ZIP fully. Open the extracted folder in VS Code.
2) Open Terminal > New Terminal. Make sure terminal is at this folder (the one containing train.csv and test.csv).
3) Windows PowerShell commands:

   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt

   If PowerShell blocks activation, run this once in that terminal:
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1

4) Run demos (each from this root folder):
   python demo1_show_cleaning.py
   python demo2_show_duplicates.py

5) Run the real pipeline (writes/replaces submission.csv):
   python code/main.py

6) Validate output format:
   python check_format.py submission.csv test.csv

7) Build lookup cache ONCE (runs full pipeline again):
   python demo3_lookup.py --load
   Then lookup any actual test candidate ID, e.g.:
   python demo3_lookup.py CH-XXXXXX
   Replace CH-XXXXXX with an ID copied from test.csv.

IMPORTANT:
- Run commands from this root folder, not from the code folder.
- The lookup cache is generated locally as pipeline_state.pkl. It is safe to delete and rebuild.
- check_format.py checks format only, not model quality.
- Use a real candidate ID for lookup.
