import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from utils import plot_confusion_matrix, plot_classification_report


def evaluate_model(model, X_test, y_test, classes):
    """
    Evaluate the model on the test set and plot results.

    Args:
        model (keras.Model): Trained Keras model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        classes (list): List of class names.
    """
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plot_confusion_matrix(cm, classes, title='Confusion Matrix')

    # Classification report
    report = classification_report(y_true_classes, y_pred_classes, target_names=classes, output_dict=True)
    plot_classification_report(report, title='Classification Report')

    return cm, report
