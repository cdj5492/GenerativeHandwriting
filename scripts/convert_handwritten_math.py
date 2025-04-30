#!/usr/bin/env python
import os
import csv
import xml.etree.ElementTree as ET
import numpy as np

# Normalization functions
def train_offset_normalization(data):
    """
    The co-ordinate offsets are normalized to mean 0, std. dev. 1 over the training set.
    data is expected to be a 3D numpy array of shape (n_total, max_len, 3).
    Only the relative coordinates (columns 1 and 2) are normalized.
    Returns: (mean, std, normalized_data)
    """
    mean = data[:, :, 1:].mean(axis=(0, 1))
    data[:, :, 1:] -= mean
    std = data[:, :, 1:].std(axis=(0, 1))
    data[:, :, 1:] /= std
    return mean, std, data

def valid_offset_normalization(mean, std, data):
    """
    Normalize validation data using the training set's mean and std.
    """
    data[:, :, 1:] -= mean
    data[:, :, 1:] /= std
    return data

def data_denormalization(mean, std, data):
    """
    Denormalize the data using the provided mean and std.
    """
    data[:, :, 1:] *= std
    data[:, :, 1:] += mean
    return data

def data_normalization(data):
    """
    Normalize a 2D array (skipping the flag column) to have zero mean and unit variance.
    """
    mean = data[:, 1:].mean(axis=0)
    data[:, 1:] -= mean
    std = data[:, 1:].std(axis=0)
    data[:, 1:] /= std
    return mean, std, data

def data_processing(data):
    """
    Scale the data to a range based on its min and max values, then scale up by a factor of 10.
    """
    min_xy = data[:, 1:].min(axis=0)
    data[:, 1:] -= min_xy
    max_xy = data[:, 1:].max(axis=0)
    data[:, 1:] /= (max_xy - min_xy)
    data[:, 1:] *= 10
    return data

def parse_xml_file(xml_path):
    """
    Parse an XML file containing stroke data and convert it into a relative stroke sequence.
    
    Each stroke in the XML is represented as a series of absolute (x, y) points.
    This function converts them into a sequence where each timestep is a vector [flag, dx, dy]:
      - For the very first stroke: the first point's displacement is taken as is,
        and subsequent points are differences between consecutive points.
      - For each subsequent stroke: the first point is computed as the difference between
        its absolute coordinate and the last point of the previous stroke (flagged with 1 to indicate a pen lift),
        while the other points are differences within that stroke.
        
    Returns a numpy array of shape (T, 3) with dtype np.float32, where T is the number of timesteps.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    stroke_set = root.find('StrokeSet')
    
    sequence = []
    current_last_point = None  # Holds the last absolute point from the previous stroke

    for stroke in stroke_set.findall('Stroke'):
        points = []
        for pt in stroke.findall('Point'):
            x = float(pt.get('x'))
            y = float(pt.get('y'))
            points.append((x, y))
        if not points:
            continue

        if current_last_point is None:
            for i in range(1, len(points)):
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
                sequence.append([0, dx, dy])
            current_last_point = points[-1]
        else:
            # New stroke: compute first point relative to the last point of the previous stroke.
            dx = points[0][0] - current_last_point[0]
            dy = points[0][1] - current_last_point[1]
            sequence.append([1, dx, dy])
            for i in range(1, len(points)):
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
                sequence.append([0, dx, dy])
            current_last_point = points[-1]

    return np.array(sequence, dtype=np.float32)

def pad_strokes(strokes_list):
    """
    Pads the list of stroke sequences (each a numpy array of shape (T, 3)) into a homogeneous 3D numpy array.
    Returns:
      - padded: a numpy array of shape (n_total, max_len, 3)
      - lengths: a list containing the original length of each stroke sequence.
    """
    n_total = len(strokes_list)
    max_len = max(len(s) for s in strokes_list)
    padded = np.zeros((n_total, max_len, 3), dtype=np.float32)
    lengths = []
    for i, stroke in enumerate(strokes_list):
        l = len(stroke)
        lengths.append(l)
        padded[i, :l, :] = stroke
    return padded, lengths

def main():
    # Define the base directory for the hmath dataset.
    # Expected directory structure:
    # hmath/
    #    mapping.csv
    #    xml/         -> contains the XML files with stroke data
    #    text/        -> (not used here, mapping.csv provides LaTeX expressions)
    # base_dir = 'data/raw/consolidated_math'

    # base_dir = 'AndyStuff/output_sessions/session_20250420_152042'
    # base_dir = 'data/raw/consolidated_math'
    # base_dir = 'data/raw/cleaned_up_math'
    # base_dir = 'data/raw/fractions_removed_1100'
    # base_dir = 'data/raw/simplified_data_630'
    base_dir = 'data/raw/consolidated_1523'
    base_dir = os.path.abspath(base_dir)
    mapping_path = os.path.join(base_dir, 'mapping.csv')
    xml_dir = os.path.join(base_dir, 'xml')
    
    strokes_list = []  # To collect stroke arrays for each sample
    sentences = []     # To collect corresponding LaTeX expressions
    
    # Read the mapping CSV
    with open(mapping_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        mappings = list(reader)
    
    # Sort mappings based on the 'index' field to ensure consistent order
    mappings.sort(key=lambda row: int(row['index']))

    for mapping in mappings:
        xml_filename = mapping['xml_filename']
        latex_expression = mapping['latex_expression']
        
        xml_path = os.path.join(xml_dir, xml_filename)
        stroke_data = parse_xml_file(xml_path)
        strokes_list.append(stroke_data)
        sentences.append(latex_expression)
    
    # shift every stroke to be centered
    # for i, stroke in enumerate(strokes_list):
    #     mean_x = stroke[:, 1].mean()
    #     mean_y = stroke[:, 2].mean()
    #     strokes_list[i][:, 1] -= mean_x
    #     strokes_list[i][:, 2] -= mean_y

    # normalize but with the same scaling on x and y
    for i, stroke in enumerate(strokes_list):
        max_val = stroke[:, 1:].max()
        min_val = stroke[:, 1:].min()
        strokes_list[i][:, 1:] /= (max_val - min_val)
    
    # flip x axis
    # for i, stroke in enumerate(strokes_list):
    #     strokes_list[i][:, 1] *= -1
    # flip y axis
    for i, stroke in enumerate(strokes_list):
        strokes_list[i][:, 2] *= -1

    # Now pad
    padded_data, lengths = pad_strokes(strokes_list)

    # Pad the variable-length stroke sequences to form a homogeneous array.
    # padded_data, lengths = pad_strokes(strokes_list)
    
    # Normalize the padded data using train_offset_normalization.
    # This adjusts the coordinate offsets (columns 1 and 2) so that they have zero mean and unit standard deviation.
    # mean, std, norm_data = train_offset_normalization(padded_data)

    # print(mean, std)
    
    # Convert the normalized data back into a list of variable-length arrays using the original lengths.
    # normalized_strokes_list = [norm_data[i, :lengths[i], :] for i in range(len(lengths))]
    
    # Save the normalized stroke sequences as an object array since each element might have a different shape.
    strokes_arr = np.array(strokes_list, dtype=object)
    np.save(os.path.join(base_dir, 'strokes.npy'), strokes_arr)
    
    # Save the sentences (LaTeX expressions) to sentences.txt, one per line.
    with open(os.path.join(base_dir, 'sentences.txt'), 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

if __name__ == '__main__':
    main()
