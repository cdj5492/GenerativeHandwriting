import numpy as np
from utils import plot_stroke
from utils.constants import Global
from utils.data_utils import data_denormalization

# Load the latest single stroke file
# AndyStuff/output_sessions/session_20250419_005200/strokes.npy
single_stroke_v2 = np.load("data/processed_math/strokes.npy", allow_pickle=True)


single_stroke_v2 = data_denormalization(Global.train_mean, Global.train_std, single_stroke_v2)

# Plot it
plot_stroke(single_stroke_v2[0], save_name="single_stroke.png")