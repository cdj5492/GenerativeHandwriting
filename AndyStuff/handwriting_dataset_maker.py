import pygame
import pygame.locals
import time
import xml.dom.minidom
import xml.etree.ElementTree as ET
import os
from datetime import datetime
from tkinter import Tk, filedialog

class StrokeCapture:
    def __init__(self, equation_file):
        pygame.init()
        self.width, self.height = 800, 600
        self.margin = 60  # space for equation display

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Stroke Capture")
        self.font = pygame.font.SysFont(None, 32)

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        # Strokes
        self.strokes = []
        self.current_stroke = None
        self.drawing = False
        self.start_time = 0
        self.end_time = 0

        self.clock = pygame.time.Clock()

        # Load equations
        with open(equation_file, 'r') as f:
            self.equations = [line.strip() for line in f if line.strip()]

        if not self.equations:
            raise ValueError("The selected equation file is empty or invalid.")

        self.current_index = 0
        self.current_equation = self.equations[self.current_index]

        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join("output_sessions", f"session_{timestamp}")
        self.xml_dir = os.path.join(self.session_dir, "xml")
        self.text_dir = os.path.join(self.session_dir, "text")

        os.makedirs(self.xml_dir, exist_ok=True)
        os.makedirs(self.text_dir, exist_ok=True)

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
                    if y > self.margin:  # avoid drawing over equation
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

        # Draw current equation
        eq_surface = self.font.render(self.current_equation, True, self.BLACK)
        self.screen.blit(eq_surface, (20, 20))
        pygame.display.flip()

    def save_pair(self):
        if not self.strokes:
            print("No strokes to save.")
            return

        idx_str = f"{self.current_index + 1:04d}"

        # Save XML
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
                              x=str(x),
                              y=str(y),
                              time=f"{timestamp:.2f}")

        xml_str = ET.tostring(root, encoding='ISO-8859-1')
        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml = "\n".join([line for line in dom.toprettyxml(indent="  ").split('\n') if line.strip()])

        xml_path = os.path.join(self.xml_dir, f"{idx_str}.xml")
        with open(xml_path, 'w') as f:
            f.write(pretty_xml)

        # Save equation
        txt_path = os.path.join(self.text_dir, f"{idx_str}.txt")
        with open(txt_path, 'w') as f:
            f.write(self.current_equation)

        print(f"Saved pair: {idx_str}.xml + {idx_str}.txt")

        # Prepare next
        self.current_index += 1
        if self.current_index < len(self.equations):
            self.current_equation = self.equations[self.current_index]
            self.clear_screen()
        else:
            print("All equations completed!")

if __name__ == "__main__":
    Tk().withdraw()  # Hide Tkinter root window
    eq_file = filedialog.askopenfilename(
        title="Select equation list (.txt)",
        filetypes=[("Text Files", "*.txt")]
    )

    if eq_file and os.path.isfile(eq_file):
        app = StrokeCapture(eq_file)
        app.run()
    else:
        print("No valid equation file selected.")
