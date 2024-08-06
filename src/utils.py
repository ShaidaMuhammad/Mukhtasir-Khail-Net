import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def load_data(file_path):
    """
    Load dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(file_path)


def preprocess_data(df, label_name):
    """
    Encode labels and prepare feature data.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        label_name (str): The name of the label column.

    Returns:
        pd.DataFrame, np.ndarray: Preprocessed data and encoded labels.
    """
    le = LabelEncoder()
    df[label_name] = le.fit_transform(df[label_name].values.ravel())
    return df, le.classes_


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot the confusion matrix.

    Args:
        cm (array): Confusion matrix.
        classes (list): List of class names.
        normalize (bool): Whether to normalize the confusion matrix.
        title (str): Title of the plot.
        cmap (Colormap): Colormap instance.
    """
    import itertools
    import numpy as np

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_classification_report(report, title='Classification Report', cmap='Greens'):
    """
    Plot the classification report as a heatmap.

    Args:
        report (dict): Classification report.
        title (str): Title of the plot.
        cmap (str): Colormap name.
    """
    import seaborn as sns
    import pandas as pd

    report_df = pd.DataFrame(report).iloc[:-1, :].T
    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df, annot=True, cmap=cmap)
    plt.title(title)
    plt.show()
