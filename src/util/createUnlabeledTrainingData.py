import xml.etree.ElementTree as ET
from tqdm import tqdm
import glob
import torch
import numpy as np

def generate_training_data(file_list):
    """
    Processes each XML file and returns a list of torch tensors,
    each representing one handwriting sequence.
    
    Pen state is now represented as a single value:
    0 = pen down (drawing)
    1 = pen up (end of stroke)
    """
    sequences = []
    for file_path in tqdm(file_list, desc="Processing XML files"):
        tree = ET.parse(file_path)
        root = tree.getroot()
        stroke_set = root.find('StrokeSet')
        
        if stroke_set is None:
            continue  # Skip file if no StrokeSet element
            
        # Process all strokes in the file
        current_sample = []
        
        for stroke in stroke_set.findall('Stroke'):
            points = stroke.findall('Point')
            if not points:
                continue
                
            # Process points within each stroke
            prev_x = float(points[0].get('x'))
            prev_y = float(points[0].get('y'))
            
            # Add all points in the stroke with pen_down (0)
            for p in points:
                x = float(p.get('x'))
                y = float(p.get('y'))
                dx = x - prev_x
                dy = y - prev_y
                
                # Only add if it's not the first point or if there's movement
                if p != points[0] or (dx != 0 and dy != 0):
                    current_sample.append({
                        'dx': dx, 
                        'dy': dy, 
                        'pen_state': 0  # pen down (drawing)
                    })
                
                prev_x = x
                prev_y = y
            
            # Add pen-up point at the end of each stroke
            current_sample.append({
                'dx': 0.0, 
                'dy': 0.0, 
                'pen_state': 1  # pen up (end of stroke)
            })
        
        if not current_sample:
            continue  # Skip if no valid points were found
            
        # Normalize the delta values
        dx_values = [sample['dx'] for sample in current_sample]
        dy_values = [sample['dy'] for sample in current_sample]
        
        # Use robust normalization with percentiles instead of max
        dx_scale = np.percentile(np.abs(dx_values), 98) or 1.0
        dy_scale = np.percentile(np.abs(dy_values), 98) or 1.0
        
        for sample in current_sample:
            sample['dx'] /= dx_scale
            sample['dy'] /= dy_scale
            # Clip to prevent extreme values
            sample['dx'] = max(min(sample['dx'], 1.0), -1.0)
            sample['dy'] = max(min(sample['dy'], 1.0), -1.0)
        
        # Convert the list of dicts into a tensor
        # Each sample is represented as: [dx, dy, pen_state]
        data = [[s['dx'], s['dy'], s['pen_state']] for s in current_sample]
        tensor_seq = torch.tensor(data, dtype=torch.float32)
        
        # Only add sequences with reasonable length
        if len(tensor_seq) >= 5:  # Minimum length threshold
            sequences.append(tensor_seq)
    
    return sequences

if __name__ == "__main__":
    strokes_dir = "lineStrokes-all/lineStrokes"
    # recursively search through all subdirectories for xml files
    file_list = glob.glob(strokes_dir + "/**/*.xml", recursive=True)
    # for now, just do a few files
    file_list = file_list[:2]
    training_data = generate_training_data(file_list)
    print(training_data)

