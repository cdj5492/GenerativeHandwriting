import xml.etree.ElementTree as ET
from tqdm import tqdm
import glob
import torch

def generate_training_data(file_list):
    """
    Processes each XML file and returns a list of torch tensors,
    each representing one handwriting sequence.
    """
    sequences = []

    for file_path in tqdm(file_list, desc="Processing XML files"):
        current_sample = []
        tree = ET.parse(file_path)
        root = tree.getroot()
        stroke_set = root.find('StrokeSet')
        if stroke_set is None:
            continue  # Skip file if no StrokeSet element

        for stroke in stroke_set.findall('Stroke'):
            points = stroke.findall('Point')
            if not points:
                continue

            # Insert a marker to indicate pen state change (e.g., pen-up) at the stroke's start.
            current_sample.append({'dx': 0.0, 'dy': 0.0, 'pen_state': [0, 1]})

            # Initialize with the first point.
            prev_x = float(points[0].get('x'))
            prev_y = float(points[0].get('y'))

            for p in points[1:]:
                x = float(p.get('x'))
                y = float(p.get('y'))
                dx = x - prev_x
                dy = y - prev_y
                prev_x = x
                prev_y = y

                current_sample.append({'dx': dx, 'dy': dy, 'pen_state': [0, 1]})

        # Normalize the delta values to be in [-1, 1] for the current sequence.
        max_delta = max((max(abs(sample['dx']), abs(sample['dy'])) for sample in current_sample), default=0)
        if max_delta == 0:
            max_delta = 1.0

        for sample in current_sample:
            sample['dx'] /= max_delta
            sample['dy'] /= max_delta

        # Convert the list of dicts into a tensor.
        # Each sample is represented as: [dx, dy, pen_state[0], pen_state[1]]
        data = [[s['dx'], s['dy'], s['pen_state'][0], s['pen_state'][1]] for s in current_sample]
        tensor_seq = torch.tensor(data, dtype=torch.float32)
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

