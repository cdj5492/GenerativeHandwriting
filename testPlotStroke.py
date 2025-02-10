# import xml.etree.ElementTree as ET
# import matplotlib.pyplot as plt

# def parse_strokes(xml_file):
#     tree = ET.parse(xml_file)
#     root = tree.getroot()
#     strokes = []
    
#     for stroke in root.findall(".//Stroke"):
#         points = []
#         for point in stroke.findall("Point"):
#             x = int(point.attrib["x"])
#             y = int(point.attrib["y"])
#             points.append((x, y))
#         strokes.append(points)
    
#     return strokes

# def plot_strokes(strokes):
#     plt.figure(figsize=(10, 6))
    
#     for stroke in strokes:
#         x_coords, y_coords = zip(*stroke)
#         plt.plot(x_coords, y_coords, marker='o', linestyle='-', markersize=2)
    
#     plt.gca().invert_yaxis()  # Invert Y-axis to match typical coordinate systems
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title("Pen Strokes Visualization")
#     plt.show()

# if __name__ == "__main__":
#     xml_file = "C:/Users/Cole Johnson/OneDrive - rit.edu/DeepLearning/GenerateHandwriting/lineStrokes-all/lineStrokes/d02/d02-062/d02-062z-02.xml"
#     strokes = parse_strokes(xml_file)
#     plot_strokes(strokes)
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    strokes = []
    whiteboard_metadata = root.find(".//WhiteboardDescription")
    if whiteboard_metadata is not None:
        max_x = int(whiteboard_metadata.find("DiagonallyOppositeCoords").get("x"))
        max_y = int(whiteboard_metadata.find("DiagonallyOppositeCoords").get("y"))
        min_x = int(whiteboard_metadata.find("HorizontallyOppositeCoords").get("x"))
        min_y = int(whiteboard_metadata.find("VerticallyOppositeCoords").get("y"))
    else:
        min_x, max_x, min_y, max_y = 0, 1, 0, 1  # Fallback values
    
    for stroke in root.findall(".//Stroke"):
        points = []
        for point in stroke.findall("Point"):
            x, y = int(point.get("x")), int(point.get("y"))
            norm_x = (x - min_x) / (max_x - min_x)
            norm_y = (y - min_y) / (max_y - min_y)
            points.append((norm_x, norm_y))
        strokes.append(points)
    
    text_lines = []
    for text_line in root.findall(".//TextLine"):
        text = text_line.get("text")
        text_lines.append(text)
    
    return strokes, text_lines

def plot_strokes_with_text(strokes, text_lines):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    # ax.invert_yaxis()
    ax.invert_xaxis()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(strokes)))
    for stroke, color in zip(strokes, colors):
        x_vals, y_vals = zip(*stroke)
        ax.plot(x_vals, y_vals, color=color)
    
    for i, text in enumerate(text_lines):
        ax.text(0.05, (i + 1) * 0.05, text, fontsize=12, color='red')
    
    plt.show()

if __name__ == "__main__":
    file_path = "original-xml-part/original/a01/a01-007/strokesz.xml"
    strokes, text_lines = parse_xml(file_path)
    plot_strokes_with_text(strokes, text_lines)

