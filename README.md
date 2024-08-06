# Activity Recognition Model

This project is for training a deep learning model to recognize different activities using time series data. The project uses a dataset containing sensor data for different activities and applies a deep learning model to classify the activities.

## Structure

- `data/`: Contains the dataset.
  - `combined_AU-SD.csv`: The dataset file.
- `src/`: Contains the source code.
  - `data_preparation.py`: Functions for loading and preprocessing data.
  - `model.py`: Defines the deep learning model architecture.
  - `train.py`: Script to train and evaluate the model.
  - `evaluate.py`: Functions for evaluating the model performance.
  - `utils.py`: Utility functions.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.

## Setup

### Prerequisites

Ensure you have Python 3.10.14 installed on your machine.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/activity-recognition-model.git
    cd activity-recognition-model
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Data

Ensure you have the `combined_AU-SD.csv` file in the `data/` directory.

## Running the Project

1. Ensure you have the `combined_AU-SD.csv` file in the `data` directory.
2. Run the training script using:

    ```bash
    python src/train.py
    ```

## Code Description

### `data_preparation.py`

This module contains functions for loading and preprocessing the data, including:

- `load_data(file_path)`: Loads the dataset from a CSV file.
- `preprocess_data(df, label_name)`: Encodes labels and prepares features.
- `create_segments_and_labels(df, time_steps, step, label_name)`: Creates segments and labels for time series data.

### `model.py`

This module defines the deep learning model architecture:

- `create_model(input_shape, num_classes)`: Creates and compiles the deep learning model.

### `train.py`

This script handles the training process, including cross-validation:

- `train_and_evaluate(data_path, time_periods, step_distance, label_name, num_splits=2)`: Trains and evaluates the model using K-Fold cross-validation.
- `plot_confusion_matrix(y_true, y_pred, classes)`: Plots the confusion matrix.
- `plot_classification_report(y_true, y_pred, classes)`: Plots the classification report.

### `evaluate.py`

This module contains functions for evaluating the model performance (to be filled with relevant evaluation functions if needed).

### `utils.py`

This module contains utility functions (to be filled with relevant utility functions if needed).

## Model Training and Evaluation

The `train.py` script trains the model using K-Fold cross-validation and evaluates its performance. The best model is saved as `trained-Model.keras`.

### Output

- The script will output the confusion matrix and classification report for the model's performance.
- The best model will be saved as `trained-Model.keras`.

## Notes

- Ensure that you have the dataset file in the correct directory before running the training script.
- The model architecture and training parameters can be adjusted in the respective scripts as needed.

## Contributing

If you wish to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

By organizing the project in this manner, the code is more maintainable, understandable, and ready to be pushed to GitHub.