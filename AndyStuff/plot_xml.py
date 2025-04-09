"""
XML Stroke Visualizer
----------------------

This script allows users to open a custom XML file containing handwritten stroke data
(captured using a stylus or mouse) and visualize the strokes as a line plot.

Functionality:
- Opens a file dialog for selecting a `.xml` stroke file.
- Parses strokes from `<Stroke>` elements and their `<Point>` children.
- Each point is expected to have `x` and `y` attributes.
- Plots each stroke as a connected line using `matplotlib`.

Usage:
- Run the script and select a stroke XML file when prompted.
- A matplotlib plot window will open, showing the handwritten strokes.

Requirements:
- Python 3
- matplotlib
- tkinter

Intended Use:
- For visualizing and verifying stroke capture output produced by interactive data collection tools.
- Compatible with stroke files saved by custom tools like StrokeCapture.
"""
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

def select_xml_file():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a stroke XML file",
        filetypes=[("XML Files", "*.xml")]
    )
    return file_path

def parse_strokes_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    strokes = []
    for stroke in root.findall(".//Stroke"):
        points = []
        for pt in stroke.findall("Point"):
            x = float(pt.attrib["x"])
            y = float(pt.attrib["y"])
            points.append((x, y))
        strokes.append(points)
    return strokes

def plot_strokes(strokes):
    for stroke in strokes:
        if len(stroke) > 1:
            xs, ys = zip(*stroke)
            plt.plot(xs, ys)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title("Stroke Plot from XML")
    plt.show()

if __name__ == "__main__":
    xml_path = select_xml_file()
    if not xml_path or not os.path.isfile(xml_path):
        print("No valid XML file selected.")
    else:
        strokes = parse_strokes_from_xml(xml_path)
        if strokes:
            plot_strokes(strokes)
        else:
            print("No strokes found in the selected XML file.")
