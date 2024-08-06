import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

from data_preparation import load_data, preprocess_data, create_segments_and_labels
from model import create_model


def train_and_evaluate(data_path, time_periods, step_distance, label_name, num_splits=2):
    """
    Train and evaluate the model using K-Fold cross-validation.

    :param data_path: str, path to the data file
    :param time_periods: int, number of time periods for segmentation
    :param step_distance: int, step size for the sliding window
    :param label_name: str, the name of the label column
    :param num_splits: int, number of K-Folds
    """
    df = load_data(data_path)
    df, le = preprocess_data(df, label_name)
    x_data, y_data = create_segments_and_labels(df, time_periods, step_distance, label_name)

    num_classes = le.classes_.size
    labels = list(le.classes_)

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    best_model = None
    best_accuracy = 0.0
    all_true = []
    all_pred = []

    for train_index, test_index in kf.split(x_data):
        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

        model = create_model(X_train.shape[1:], num_classes)
        checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True,
                                     mode='max')
        history = model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0, validation_data=(X_test, y_test),
                            callbacks=[checkpoint])

        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        all_true.extend(y_true_classes)
        all_pred.extend(y_pred_classes)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            print("Best Model Updated")

    best_model.save('trained-Model.keras')
    plot_confusion_matrix(all_true, all_pred, labels)
    plot_classification_report(all_true, all_pred, labels)


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    df_cm = pd.DataFrame(cm, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(6, 6))
    sns.heatmap(df_cm, annot=True, square=True, cmap="Blues", linewidths=.2, cbar_kws={"shrink": 0.8})
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def plot_classification_report(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    plt.figure(figsize=(4, 4))
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, cmap="Greens", annot=True)
    plt.show()


if __name__ == "__main__":
    data_path = '../data/combined_AU-SD.csv'
    TIME_PERIODS = 256
    STEP_DISTANCE = 16
    LABEL = 'Act'

    train_and_evaluate(data_path, TIME_PERIODS, STEP_DISTANCE, LABEL)
