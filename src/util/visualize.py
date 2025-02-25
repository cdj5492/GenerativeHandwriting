from matplotlib import pyplot as plt
import torch

def visualize_sequence_global(sequence, title="Handwriting Sample", save_path=None):
    """
    Visualize a handwriting sequence
    
    Args:
        sequence: Tensor of shape (seq_len, 3) containing [x, y, pen_state]
        title: Title for the plot
        save_path: Path to save the visualization (if None, will display)
    """
    # Convert to numpy for easier manipulation
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.cpu().numpy()
    
    coords_x = []
    coords_y = []
    pen_states = []
    
    for x, y, pen_state in sequence:
        coords_x.append(x)
        coords_y.append(y)
        pen_states.append(pen_state)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    
    # Plot points with different colors for each stroke
    stroke_x = []
    stroke_y = []
    stroke_count = 0
    colors = plt.cm.tab10.colors  # Use a colormap for different strokes
    
    for i in range(len(coords_x)):
        if i == 0 or pen_states[i] > 0.5:
            if stroke_x:  # Only plot if we have points
                color = colors[stroke_count % len(colors)]
                ax.plot(stroke_x, stroke_y, '-', color=color, linewidth=2)
                stroke_count += 1
            stroke_x = []
            stroke_y = []
        stroke_x.append(coords_x[i])
        stroke_y.append(coords_y[i])
    
    # Plot the last stroke
    if stroke_x:
        color = colors[stroke_count % len(colors)]
        ax.plot(stroke_x, stroke_y, '-', color=color, linewidth=2)
    
    # Set axis limits with some padding
    x_min, x_max = min(coords_x), max(coords_x)
    y_min, y_max = min(coords_y), max(coords_y)
    
    padding = max((x_max - x_min), (y_max - y_min)) * 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    ax.set_title(title)
    
    # Save or display
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def visualize_sequence_delta(sequence, title="Handwriting Sample", save_path=None):
    """
    Visualize a handwriting sequence
    
    Args:
        sequence: Tensor of shape (seq_len, 3) containing [dx, dy, pen_state]
        title: Title for the plot
        save_path: Path to save the visualization (if None, will display)
    """
    # Convert to numpy for easier manipulation
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.cpu().numpy()
    
    # Accumulate coordinates
    x, y = 0, 0
    coords_x = []
    coords_y = []
    pen_states = []
    
    for dx, dy, pen_state in sequence:
        x += dx
        y += dy
        coords_x.append(x)
        coords_y.append(y)
        pen_states.append(pen_state)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    
    # Plot points with different colors for each stroke
    stroke_x = []
    stroke_y = []
    stroke_count = 0
    colors = plt.cm.tab10.colors  # Use a colormap for different strokes
    
    for i in range(len(coords_x)):
        if pen_states[i-1 if i > 0 else 0] > 0.5:
            if stroke_x:  # Only plot if we have points
                color = colors[stroke_count % len(colors)]
                ax.plot(stroke_x, stroke_y, '-', color=color, linewidth=2)
                stroke_count += 1
            stroke_x = []
            stroke_y = []
        stroke_x.append(coords_x[i])
        stroke_y.append(coords_y[i])
    
    # Plot the last stroke
    if stroke_x:
        color = colors[stroke_count % len(colors)]
        ax.plot(stroke_x, stroke_y, '-', color=color, linewidth=2)
    
    # Set axis limits with some padding
    x_min, x_max = min(coords_x), max(coords_x)
    y_min, y_max = min(coords_y), max(coords_y)
    
    padding = max((x_max - x_min), (y_max - y_min)) * 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    ax.set_title(title)
    
    # Save or display
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()