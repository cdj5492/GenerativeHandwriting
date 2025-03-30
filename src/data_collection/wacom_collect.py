import pygame
import pygame.locals
import time
import xml.dom.minidom
import xml.etree.ElementTree as ET
from datetime import datetime

class StrokeCapture:
    def __init__(self):
        pygame.init()
        
        # Set up the window
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Stroke Capture")
        
        # Set up colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        
        # Initialize stroke collection variables
        self.strokes = []
        self.current_stroke = None
        self.drawing = False
        self.start_time = 0
        self.end_time = 0
        
        # Set up clock for timing
        self.clock = pygame.time.Clock()
        
    def run(self):
        running = True
        self.screen.fill(self.WHITE)
        pygame.display.flip()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Handle pen down (start stroke)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.drawing = True
                    self.start_time = time.time()
                    self.current_stroke = []
                    x, y = event.pos
                    timestamp = time.time()
                    self.current_stroke.append((x, y, timestamp))
                
                # Handle pen movement
                elif event.type == pygame.MOUSEMOTION and self.drawing:
                    x, y = event.pos
                    timestamp = time.time()
                    self.current_stroke.append((x, y, timestamp))
                    self.draw_line()
                
                # Handle pen up (end stroke)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.drawing = False
                    self.end_time = time.time()
                    if self.current_stroke:
                        self.strokes.append((self.current_stroke, self.start_time, self.end_time))
                
                # Handle keyboard commands
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        self.save_strokes()
                    elif event.key == pygame.K_c:
                        self.clear_screen()
                    elif event.key == pygame.K_q:
                        running = False
            
            self.clock.tick(60)
        
        pygame.quit()

    def draw_line(self):
        """Draw the current line segment"""
        if len(self.current_stroke) < 2:
            return
        
        last_point = self.current_stroke[-2][:2]  # x, y of second-to-last point
        current_point = self.current_stroke[-1][:2]  # x, y of last point
        
        pygame.draw.line(self.screen, self.BLACK, last_point, current_point, 2)
        pygame.display.update()

    def clear_screen(self):
        """Clear the screen and reset strokes"""
        self.screen.fill(self.WHITE)
        pygame.display.flip()
        self.strokes = []

    def save_strokes(self):
        """Save strokes to XML file"""
        if not self.strokes:
            print("No strokes to save")
            return
        
        # Create the XML structure
        root = ET.Element("WhiteboardCaptureSession")
        
        # Add whiteboard description
        desc = ET.SubElement(root, "WhiteboardDescription")
        sensor = ET.SubElement(desc, "SensorLocation", corner="top_left")
        ET.SubElement(desc, "DiagonallyOppositeCoords", x=str(self.width), y=str(self.height))
        ET.SubElement(desc, "VerticallyOppositeCoords", x="888", y=str(self.height))
        ET.SubElement(desc, "HorizontallyOppositeCoords", x=str(self.width), y="1519")
        
        # Add stroke set
        stroke_set = ET.SubElement(root, "StrokeSet")
        
        # Add each stroke with its points
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
        
        # Create the XML string with proper formatting
        xml_str = ET.tostring(root, encoding='ISO-8859-1')
        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        # Remove extra newlines that minidom adds
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        pretty_xml = '\n'.join(lines)
        
        # Save to file
        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        filename = f"strokes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
        filename = os.path.join(out_dir, filename)
        with open(filename, 'w') as f:
            f.write(pretty_xml)
        
        print(f"Strokes saved to {filename}")
        
        # Clear the screen for new capture
        self.clear_screen()

if __name__ == "__main__":
    app = StrokeCapture()
    app.run()