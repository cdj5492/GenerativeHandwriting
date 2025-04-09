"""
InkML LaTeX Extractor
---------------------

This script processes a folder of InkML files (typically from handwritten math datasets)
and extracts the associated LaTeX expressions embedded as annotations.

Functionality:
- Scans a user-selected directory for `.inkml` files.
- For each file, attempts to extract the LaTeX math expression from:
    1. <annotation> tags with type="truth"
    2. <annotation> tags containing the word "latex"
    3. <annotationXML> tags with a LaTeX type attribute
- Outputs a text file (`all_equations.txt` by default) with one LaTeX expression per line.
- If an InkML file doesn't contain a valid LaTeX annotation, a placeholder line is inserted:
    "MISSING_LATEX for filename.inkml"
- A progress bar (using tqdm) shows status during processing.

Usage:
- Run the script and select a folder containing InkML files when prompted.
- The output file will be saved in the same directory as the script.

Dependencies:
- tqdm (for progress bar)
- tkinter (for GUI folder selection)
"""
import os
import xml.etree.ElementTree as ET
from tkinter import Tk, filedialog
from tqdm import tqdm

def extract_latex_from_inkml(file_path):
    ns = {'ink': 'http://www.w3.org/2003/InkML'}
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        for ann in root.findall('ink:annotation', ns):
            if ann.attrib.get('type') == 'truth' or 'latex' in ann.text.lower():
                return ann.text.strip()

        for ann in root.findall('ink:annotationXML', ns):
            if 'latex' in ann.attrib.get('type', '').lower():
                return ann.text.strip()

        return None
    except Exception:
        return None

def collect_latex_from_directory(input_dir, output_filename="all_equations.txt"):
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".inkml")
    ])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_filename)

    with open(output_path, 'w') as out_file:
        for fname in tqdm(files, desc="Processing InkML files", unit="file"):
            full_path = os.path.join(input_dir, fname)
            latex = extract_latex_from_inkml(full_path)
            if latex is None:
                latex = f"MISSING_LATEX for {fname}"
            out_file.write(latex + "\n")

    print(f"\n✓ LaTeX list written to: {output_path}")

if __name__ == "__main__":
    Tk().withdraw()
    selected_dir = filedialog.askdirectory(title="Select folder containing InkML files")
    if selected_dir and os.path.isdir(selected_dir):
        collect_latex_from_directory(selected_dir)
    else:
        print("No valid folder selected.")
