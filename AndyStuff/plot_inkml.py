"""
InkML Viewer and LaTeX Extractor
--------------------------------

This script allows users to select and inspect a single InkML file, visualize its pen stroke
data, and extract the associated LaTeX expression if it exists.

Functionality:
- Opens a file dialog for selecting an `.inkml` file.
- Parses the file to extract pen stroke data encoded in `<trace>` elements.
- Optionally extracts a LaTeX expression from:
    - <annotation> tags with type="truth" or text containing "latex"
    - <annotationXML> tags with a LaTeX type attribute
- Displays the parsed stroke data using matplotlib as a 2D line plot.
- Prints the guessed LaTeX expression (if found) to the terminal.

Usage:
- Run the script and select an InkML file when prompted.
- View the corresponding strokes and the associated LaTeX math expression.

Requirements:
- Python 3
- matplotlib
- tkinter

Intended Use:
- Useful for debugging InkML files or manually inspecting how expressions are written.
- Can be used as a data exploration tool for datasets like CROHME or HME100K.
"""
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

def select_inkml_file():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an InkML file",
        filetypes=[("InkML files", "*.inkml")]
    )
    return file_path

def parse_inkml(file_path):
    ns = {'ink': 'http://www.w3.org/2003/InkML'}
    tree = ET.parse(file_path)
    root = tree.getroot()

    strokes = []
    for trace in root.findall('ink:trace', ns):
        points = []
        for pt in trace.text.strip().split(','):
            coords = pt.strip().split()
            if len(coords) >= 2:
                x, y = float(coords[0]), float(coords[1])
                points.append((x, y))
        strokes.append(points)

    # Try to extract LaTeX from annotations
    latex = "Not found"
    for ann in root.findall('ink:annotation', ns):
        if ann.attrib.get('type') == 'truth' or 'latex' in ann.text.lower():
            latex = ann.text.strip()

    for ann_xml in root.findall('ink:annotationXML', ns):
        if 'latex' in ann_xml.attrib.get('type', '').lower():
            latex = ann_xml.text.strip()

    return strokes, latex

def plot_strokes(strokes):
    for stroke in strokes:
        if len(stroke) > 1:
            xs, ys = zip(*stroke)
            plt.plot(xs, ys)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title("InkML Stroke Plot")
    plt.show()

if __name__ == "__main__":
    file_path = select_inkml_file()
    if not file_path or not os.path.isfile(file_path):
        print("No valid InkML file selected.")
    else:
        strokes, latex = parse_inkml(file_path)
        print(f"\nLaTeX Guess: {latex}\n")
        plot_strokes(strokes)
