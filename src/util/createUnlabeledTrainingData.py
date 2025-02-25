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
    1 = pen up 
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
        
        # Process points within each stroke
        prev_x = float(stroke_set.findall('Stroke')[0].findall('Point')[0].get('x'))
        prev_y = -float(stroke_set.findall('Stroke')[0].findall('Point')[0].get('y'))

        for stroke in stroke_set.findall('Stroke'):
            points = stroke.findall('Point')
            if not points:
                continue
            
            for i, p in enumerate(points):
                x = float(p.get('x'))
                y = -float(p.get('y'))
                
                # dx = x
                # dy = y
                dx = x - prev_x
                dy = y - prev_y
                prev_x = x
                prev_y = y
                
                # if p != points[0] or (dx != 0 and dy != 0):
                current_sample.append({
                    'dx': dx, 
                    'dy': dy, 
                    'pen_state': 1 if i == (len(points) - 1) else 0
                })
        
        if not current_sample:
            continue  # Skip if no valid points were found
            
        # Normalize the delta values
        dx_values = [sample['dx'] for sample in current_sample]
        dy_values = [sample['dy'] for sample in current_sample]
        
        # Use robust normalization with percentiles instead of max
        dx_scale = np.percentile(np.abs(dx_values), 98) or 1.0
        dy_scale = np.percentile(np.abs(dy_values), 98) or 1.0
        bigger_scale = max(dx_scale, dy_scale)
        
        for sample in current_sample:
            sample['dx'] /= bigger_scale
            sample['dy'] /= bigger_scale
            # Clip to prevent extreme values
            # sample['dx'] = max(min(sample['dx'], 1.0), -1.0)
            # sample['dy'] = max(min(sample['dy'], 1.0), -1.0)
        
        # Convert the list of dicts into a tensor
        # Each sample is represented as: [dx, dy, pen_state]
        data = [[s['dx'], s['dy'], s['pen_state']] for s in current_sample]
        tensor_seq = torch.tensor(data, dtype=torch.float32)

        # pass through torch.nan_to_num
        tensor_seq = torch.nan_to_num(tensor_seq, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Only add sequences with reasonable length
        if len(tensor_seq) >= 5:  # Minimum length threshold
            sequences.append(tensor_seq)
    
    return sequences

if __name__ == "__main__":
    from visualize import visualize_sequence_global, visualize_sequence_delta

    strokes_dir = "lineStrokes-all/lineStrokes"
    # recursively search through all subdirectories for xml files
    file_list = glob.glob(strokes_dir + "/**/*.xml", recursive=True)
    # for now, just do a few files
    file_list = file_list[:4]
    training_data = generate_training_data(file_list)
    print(training_data)

    # plot the first sequence
    visualize_sequence_global(training_data[3])
    visualize_sequence_delta(training_data[3])

