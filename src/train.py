import os
import numpy as np
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from data_preparation import create_segments_and_labels
from model import create_model
from utils import load_data, preprocess_data
from evaluate import evaluate_model

# Constants
TIME_PERIODS = 256
STEP_DISTANCE = 16
LABEL = 'Act'
NUM_SPLITS = 2
DATA_PATH = 'data/combined_AU-SD.csv'


def train_and_evaluate(data_path, time_periods, step_distance, label_name, num_splits=2):
    # Load and preprocess data
    df = load_data(data_path)
    df, classes = preprocess_data(df, label_name)

    # Create segments and labels
    x_data, y_data = create_segments_and_labels(df, time_periods, step_distance, label_name)
    num_classes = len(classes)

    # K-Fold cross-validation
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    all_true, all_pred = [], []
    best_model, best_accuracy = None, 0.0

    for train_index, test_index in kf.split(x_data):
        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

        # Create and compile model
        model = create_model(X_train.shape[1:], num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train model with checkpoint
        checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max',
                                     verbose=1)
        history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test),
                            callbacks=[checkpoint], verbose=0)

        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        # Load best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

        # Predict and store results
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        all_true.extend(y_true_classes)
        all_pred.extend(y_pred_classes)

    # Evaluate best model
    cm, report = evaluate_model(best_model, x_data, to_categorical(y_data, num_classes), classes)

    # Save final model
    best_model.save('trained-Model.keras')
    print("Model training and evaluation completed.")


if __name__ == '__main__':
    train_and_evaluate(DATA_PATH, TIME_PERIODS, STEP_DISTANCE, LABEL, NUM_SPLITS)
