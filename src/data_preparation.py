import pandas as pd
from sklearn import preprocessing
from scipy import stats
import numpy as np


def load_data(file_path):
    """
    Load the dataset from a CSV file.

    :param file_path: str, path to the CSV file
    :return: pd.DataFrame
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df, label_name):
    """
    Preprocess the dataset: encode labels and prepare features.

    :param df: pd.DataFrame, raw data
    :param label_name: str, the name of the label column
    :return: pd.DataFrame
    """
    le = preprocessing.LabelEncoder()
    df[label_name] = le.fit_transform(df[label_name].values.ravel())
    return df, le


def create_segments_and_labels(df, time_steps, step, label_name):
    """
    Create segments and labels for time series data.

    :param df: pd.DataFrame, preprocessed data
    :param time_steps: int, number of steps in each segment
    :param step: int, step size for the sliding window
    :param label_name: str, the name of the label column
    :return: tuple of np.ndarray, segmented data and labels
    """
    N_FEATURES = 6
    segments = []
    labels = []

    for i in range(0, len(df) - time_steps, step):
        wx = df['wx'].values[i: i + time_steps]
        wy = df['wy'].values[i: i + time_steps]
        wz = df['wz'].values[i: i + time_steps]
        nx = df['nx'].values[i: i + time_steps]
        ny = df['ny'].values[i: i + time_steps]
        nz = df['nz'].values[i: i + time_steps]
        label_slice = df[label_name][i: i + time_steps]
        label_mode = stats.mode(label_slice)

        if isinstance(label_mode.mode, np.ndarray):
            label = label_mode.mode[0]
        else:
            label = label_mode.mode

        segments.append([wx, wy, wz, nx, ny, nz])
        labels.append(label)

    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels
