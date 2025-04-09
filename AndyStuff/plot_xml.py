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
