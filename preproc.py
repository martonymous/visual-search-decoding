import h5py
import numpy as np
import pandas as pd


def load_gaze_data(h5_file: h5py.Dataset, output='pandas'):
    """
    Load gaze data from a Titta-generated HDF5 file.
    """
    with h5py.File(h5_file, 'r') as f:
        labels = np.array(f['gaze']['axis0'])

        categorical_data  = np.array(f['gaze']['block0_values'])
        categorical_labels = labels[28:]

        measurement_data = np.array(f['gaze']['block1_values'])
        measurement_labels = labels[2:28]

        timestamps = np.array(f['gaze']['block2_values'])
        timestamp_labels = labels[:2]

    if output == 'pandas':
        categorical_data = pd.DataFrame(categorical_data, columns=categorical_labels)
        measurement_data = pd.DataFrame(measurement_data, columns=measurement_labels)
        timestamps = pd.DataFrame(timestamps, columns=timestamp_labels)

        return categorical_data, measurement_data, timestamps

    return (categorical_data, categorical_labels), (measurement_data, measurement_labels), (timestamps, timestamp_labels)

def preprocess_raw(df: pd.DatFrame, filter: bool=False):
    """
    Preprocess raw gaze data.
    """
    # there might also be some bad values, out of range, filter those by setting to NaN, then interpolate
    if filter:
        for col in df.columns:
            if col.startswith('right_gaze_point_on_display_area') or col.startswith('left_gaze_point_on_display_area'):
                df[col] = df[col].apply(lambda x: x if 0 <= x <= 1 else np.nan)        

    df['rx'] = df[b'right_gaze_point_on_display_area_x'].interpolate()
    df['ry'] = df[b'right_gaze_point_on_display_area_y'].interpolate()
    df['lx'] = df[b'left_gaze_point_on_display_area_x'].interpolate()
    df['ly'] = df[b'left_gaze_point_on_display_area_y'].interpolate()

    df['x'] = (df['rx'] + df['lx']) / 2
    df['y'] = (df['ry'] + df['ly']) / 2

    return df

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
    cat, con, time = load_gaze_data('test.h5')
    df = pd.concat([con, time])
    fixations = dispersion_fixation_detection(df, dispersion_threshold=1, min_duration_ms=100)
    # print(fixations)