"""
StrokeCapture Tool for LaTeX-based Handwriting Dataset Collection
-----------------------------------------------------------------

This Python script provides an interactive interface for collecting handwriting data
to build a dataset for training models that generate handwritten versions of LaTeX math expressions.

Functionality:
- Loads a list of LaTeX math expressions (one per line) from a user-selected text file.
- Renders and displays each expression as a human-readable image on a Pygame canvas.
- Captures and saves the handwritten pen strokes in an XML format (with timing and coordinates).
- Saves the corresponding LaTeX expression in a text file alongside the XML.
- All outputs are organized into a session folder with separate `xml/` and `text/` subdirectories.
- At the end of the session, it creates a CSV file summarizing all saved samples.

Controls:
- Draw using the mouse or stylus.
- Press `S` to save the current drawing and proceed to the next expression.
- Press `C` to clear the canvas and redraw the current expression.
- Press `N` to skip the current expression without saving.
- Press `Q` to quit the session early.
"""

import os
import time
import pygame
import xml.dom.minidom
import xml.etree.ElementTree as ET
from datetime import datetime
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from PIL import Image
import io

class StrokeCapture:
    def __init__(self, equation_file):
        pygame.init()
        self.width, self.height = 1200, 800
        self.margin = 80

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("LaTeX Stroke Capture")
        self.font = pygame.font.SysFont(None, 32)

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        self.strokes = []
        self.current_stroke = None
        self.drawing = False
        self.start_time = 0
        self.end_time = 0
        self.save_index = 1
        self.saved_entries = []  # holds (index, xml, txt, expr)

        self.clock = pygame.time.Clock()

        with open(equation_file, 'r', encoding='utf-8') as f:
            self.equations = [line.strip() for line in f if line.strip()]

        if not self.equations:
            raise ValueError("The selected equation file is empty or invalid.")

        self.current_index = 0
        self.current_equation = self.equations[self.current_index]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join("output_sessions", f"session_{timestamp}")
        self.xml_dir = os.path.join(self.session_dir, "xml")
        self.text_dir = os.path.join(self.session_dir, "text")

        os.makedirs(self.xml_dir, exist_ok=True)
        os.makedirs(self.text_dir, exist_ok=True)

    def sanitize_latex(self, latex_expr):
        replacements = {
            r"\lt": "<",
            r"\gt": ">",
            r"\leq": r"\le",
            r"\geq": r"\ge",
            r"\le": "<=",
            r"\ge": ">=",
            r"\neq": r"\ne",
            r"\=": "="
        }
        for old, new in replacements.items():
            latex_expr = latex_expr.replace(old, new)
        return latex_expr


    def render_latex_to_surface(self, latex_str):
        fig, ax = plt.subplots(figsize=(10.0, 3.0), dpi=100)
        ax.text(0.5, 0.5, f"${latex_str}$", fontsize=30, ha='center', va='center')
        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        img = img.resize((700, 160))
        mode = img.mode
        size = img.size
        data = img.tobytes()

        return pygame.image.fromstring(data, size, mode)

    def run(self):
        running = True
        self.clear_screen()

        while running and self.current_index < len(self.equations):
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
                    elif event.key == pygame.K_n:
                        print(f"Skipped: {self.current_index + 1:04d}")
                        self.next_equation()
                    elif event.key == pygame.K_q:
                        running = False

            self.clock.tick(60)

        pygame.quit()
        self.write_summary_files()

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

        sanitized_expr = self.sanitize_latex(self.current_equation)

        try:
            latex_surface = self.render_latex_to_surface(sanitized_expr)
            self.screen.blit(latex_surface, (20, 20))
        except Exception as e:
            print("Error rendering LaTeX:", sanitized_expr)
            print(e)
            error_surface = self.font.render("Error displaying LaTeX", True, self.BLACK)
            self.screen.blit(error_surface, (20, 20))

        counter_text = f"Expression {self.current_index + 1} of {len(self.equations)}"
        counter_surface = self.font.render(counter_text, True, self.BLACK)
        self.screen.blit(counter_surface, (20, self.height - 40))

        pygame.display.flip()

    def save_pair(self):
        if not self.strokes:
            print("No strokes to save.")
            return

        idx_str = f"{self.save_index:04d}"
        self.save_index += 1

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
            for x, y, timestamp in stroke_data:
                ET.SubElement(stroke, "Point",
                              x=str(x), y=str(y), time=f"{timestamp:.2f}")

        xml_str = ET.tostring(root, encoding='ISO-8859-1')
        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml = "\n".join([line for line in dom.toprettyxml(indent="  ").split('\n') if line.strip()])

        xml_path = os.path.join(self.xml_dir, f"{idx_str}.xml")
        txt_path = os.path.join(self.text_dir, f"{idx_str}.txt")

        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(self.current_equation)

        self.saved_entries.append((self.save_index - 1, f"{idx_str}.xml", f"{idx_str}.txt", self.current_equation))
        print(f"Saved: {idx_str}.xml + {idx_str}.txt")

        self.next_equation()

    def next_equation(self):
        self.current_index += 1
        if self.current_index < len(self.equations):
            self.current_equation = self.equations[self.current_index]
            self.clear_screen()
        else:
            print("All equations completed.")

    def write_summary_files(self):
        if not self.saved_entries:
            return

        # Write .txt file
        txt_path = os.path.join(self.session_dir, "all_expressions.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            for _, _, _, expr in self.saved_entries:
                f.write(expr + '\n')

        # Write .csv file
        csv_path = os.path.join(self.session_dir, "session_summary.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("index,xml_filename,text_filename,expression\n")
            for idx, xml_name, txt_name, expr in self.saved_entries:
                escaped = expr.replace('"', '""')  # escape quotes for CSV
                f.write(f'{idx},"{xml_name}","{txt_name}","{escaped}"\n')

        print(f"Summary written to:\n  {txt_path}\n  {csv_path}")

if __name__ == "__main__":
    Tk().withdraw()
    eq_file = filedialog.askopenfilename(
        title="Select LaTeX expression list (.txt)",
        filetypes=[("Text Files", "*.txt")]
    )

    if eq_file and os.path.isfile(eq_file):
        app = StrokeCapture(eq_file)
        app.run()
    else:
        print("No valid equation file selected.")