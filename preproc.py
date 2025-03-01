
import h5py
import numpy as np
import pandas as pd


def load_gaze_data(h5_file, dataset='gaze'):
    """
    Load gaze data from a Titta-generated HDF5 file.
    """
    with h5py.File(h5_file, 'r') as f:
        data = pd.DataFrame(np.array(f[dataset]))
    return data

def degrees_to_pixels(degrees, screen_resolution, screen_width_cm, viewing_distance_cm):
    """
    Convert visual degrees to pixels.
    """
    pixels_per_cm = screen_resolution[0] / screen_width_cm
    cm_per_degree = 2 * viewing_distance_cm * np.tan(np.deg2rad(degrees / 2))
    return cm_per_degree * pixels_per_cm

def calculate_dispersion(x, y):
    """
    Calculate the dispersion of a set of x and y points.
    Dispersion = max(x) - min(x) + max(y) - min(y)
    """
    if len(x) < 2:
        return 0
    return (np.max(x) - np.min(x)) + (np.max(y) - np.min(y))

def dispersion_fixation_detection(gaze_data, dispersion_threshold, min_duration_ms):
    """
    I-DT dispersion-based fixation detection algorithm.

    gaze_data: DataFrame with 'time', 'x', 'y' columns.
    dispersion_threshold: Max dispersion (in pixels or degrees).
    min_duration_ms: Minimum duration for a fixation (in milliseconds).

    Returns a list of fixations, each as a (start_time, end_time, centroid_x, centroid_y).
    """
    fixations = []
    start_idx = 0
    while start_idx < len(gaze_data):
        end_idx = start_idx
        while end_idx < len(gaze_data):
            window = gaze_data.iloc[start_idx:end_idx+1]
            dispersion = calculate_dispersion(window['x'], window['y'])

            if dispersion > dispersion_threshold:
                break

            end_idx += 1

        duration = gaze_data.iloc[end_idx-1]['time'] - gaze_data.iloc[start_idx]['time']

        if duration >= min_duration_ms:
            fixation_x = window['x'].mean()
            fixation_y = window['y'].mean()
            fixations.append((gaze_data.iloc[start_idx]['time'], gaze_data.iloc[end_idx-1]['time'], fixation_x, fixation_y))

        start_idx = end_idx

    return fixations


if __name__ == '__main__':
    gaze_data = load_gaze_data('data.h5')
    fixations = dispersion_fixation_detection(gaze_data, dispersion_threshold=degrees_to_pixels(1, (1920, 1080), 53.13, 60), min_duration_ms=100)
    print(fixations)