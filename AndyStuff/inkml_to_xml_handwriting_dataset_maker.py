import os
import pygame
import time
import xml.etree.ElementTree as ET
import xml.dom.minidom
from datetime import datetime
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from PIL import Image
import io


def extract_latex_from_inkml(inkml_path):
    ns = {'ink': 'http://www.w3.org/2003/InkML'}
    tree = ET.parse(inkml_path)
    root = tree.getroot()

    for ann in root.findall('ink:annotation', ns):
        if ann.attrib.get('type') == 'truth' or 'latex' in ann.text.lower():
            return ann.text.strip()

    for ann in root.findall('ink:annotationXML', ns):
        if 'latex' in ann.attrib.get('type', '').lower():
            return ann.text.strip()

    return "UNKNOWN_EXPRESSION"


class StrokeCaptureFromInkML:
    def __init__(self, inkml_dir):
        pygame.init()
        self.width, self.height = 1000, 700  # was 800x600
        self.margin = 60
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Stroke Capture for InkML Expression")
        self.font = pygame.font.SysFont(None, 28)

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        self.clock = pygame.time.Clock()
        self.strokes = []
        self.current_stroke = None
        self.drawing = False
        self.start_time = 0
        self.end_time = 0

        # Load InkML files
        self.inkml_files = sorted([
            os.path.join(inkml_dir, f)
            for f in os.listdir(inkml_dir)
            if f.lower().endswith(".inkml")
        ])
        if not self.inkml_files:
            raise ValueError("No .inkml files found in the selected directory.")

        self.current_index = 0
        self.current_latex = extract_latex_from_inkml(self.inkml_files[self.current_index])

        # Output folders
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join("output_sessions", f"session_{timestamp}")
        self.xml_dir = os.path.join(self.session_dir, "xml")
        self.text_dir = os.path.join(self.session_dir, "text")

        os.makedirs(self.xml_dir, exist_ok=True)
        os.makedirs(self.text_dir, exist_ok=True)

    def run(self):
        running = True
        self.clear_screen()

        while running and self.current_index < len(self.inkml_files):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.drawing = True
                    self.start_time = time.time()
                    self.current_stroke = []
                    x, y = event.pos
                    if y > self.margin:
                        self.current_stroke.append((x, y, time.time()))

                elif event.type == pygame.MOUSEMOTION and self.drawing:
                    x, y = event.pos
                    if y > self.margin:
                        self.current_stroke.append((x, y, time.time()))
                        self.draw_line()

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.drawing = False
                    self.end_time = time.time()
                    if self.current_stroke:
                        self.strokes.append((self.current_stroke, self.start_time, self.end_time))

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        self.save_pair()
                    elif event.key == pygame.K_c:
                        self.clear_screen()
                    elif event.key == pygame.K_q:
                        running = False

            self.clock.tick(60)

        pygame.quit()

    def draw_line(self):
        if len(self.current_stroke) < 2:
            return
        last_point = self.current_stroke[-2][:2]
        current_point = self.current_stroke[-1][:2]
        pygame.draw.line(self.screen, self.BLACK, last_point, current_point, 2)
        pygame.display.update()

    def clear_screen(self):
        self.screen.fill(self.WHITE)
        self.strokes = []

        # Draw LaTeX ASCII
        eq_surface = self.font.render(self.current_latex, True, self.BLACK)
        self.screen.blit(eq_surface, (20, 20))

        # Draw InkML visual preview
        inkml_path = self.inkml_files[self.current_index]
        stroke_preview = self.render_inkml_to_surface(inkml_path)
        self.screen.blit(stroke_preview, (self.width - 220, 10))  # was (self.width - 140, 10)

        pygame.display.flip()

    def save_pair(self):
        if not self.strokes:
            print("No strokes to save.")
            return

        idx_str = f"{self.current_index + 1:04d}"

        # Save strokes as XML
        root = ET.Element("WhiteboardCaptureSession")
        desc = ET.SubElement(root, "WhiteboardDescription")
        ET.SubElement(desc, "SensorLocation", corner="top_left")
        ET.SubElement(desc, "DiagonallyOppositeCoords", x=str(self.width), y=str(self.height))
        ET.SubElement(desc, "VerticallyOppositeCoords", x="888", y=str(self.height))
        ET.SubElement(desc, "HorizontallyOppositeCoords", x=str(self.width), y="1519")

        stroke_set = ET.SubElement(root, "StrokeSet")
        for stroke_data, start_time, end_time in self.strokes:
            stroke = ET.SubElement(stroke_set, "Stroke",
                                   colour="black",
                                   start_time=f"{start_time:.2f}",
                                   end_time=f"{end_time:.2f}")
            for x, y, t in stroke_data:
                ET.SubElement(stroke, "Point", x=str(x), y=str(y), time=f"{t:.2f}")

        xml_str = ET.tostring(root, encoding='ISO-8859-1')
        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml = "\n".join([line for line in dom.toprettyxml(indent="  ").split('\n') if line.strip()])

        xml_path = os.path.join(self.xml_dir, f"{idx_str}.xml")
        with open(xml_path, 'w') as f:
            f.write(pretty_xml)

        # Save LaTeX expression as text
        txt_path = os.path.join(self.text_dir, f"{idx_str}.txt")
        with open(txt_path, 'w') as f:
            f.write(self.current_latex)

        print(f"Saved: {idx_str}.xml and {idx_str}.txt")

        # Next sample
        self.current_index += 1
        if self.current_index < len(self.inkml_files):
            self.current_latex = extract_latex_from_inkml(self.inkml_files[self.current_index])
            self.clear_screen()
        else:
            print("All InkML expressions completed!")

    def render_inkml_to_surface(self, inkml_path):
        ns = {'ink': 'http://www.w3.org/2003/InkML'}
        tree = ET.parse(inkml_path)
        root = tree.getroot()

        strokes = []
        for trace in root.findall('ink:trace', ns):
            points = []
            for pt in trace.text.strip().split(','):
                coords = pt.strip().split()
                if len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    points.append((x, y))
            if points:
                strokes.append(points)

        # Plot and convert to Pygame Surface
        fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
        for stroke in strokes:
            xs, ys = zip(*stroke)
            ax.plot(xs, ys, color="black")
        ax.axis("off")
        ax.set_aspect("equal")
        plt.gca().invert_yaxis()

        buf = io.BytesIO()
        plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img = img.resize((200, 200))  # was (120, 120)
        mode = img.mode
        size = img.size
        data = img.tobytes()

        return pygame.image.fromstring(data, size, mode)


if __name__ == "__main__":
    Tk().withdraw()
    selected_dir = filedialog.askdirectory(title="Select Folder with InkML Files")
    if selected_dir and os.path.isdir(selected_dir):
        app = StrokeCaptureFromInkML(selected_dir)
        app.run()
    else:
        print("No valid directory selected.")
