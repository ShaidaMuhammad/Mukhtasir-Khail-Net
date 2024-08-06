# src/model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, add, Dense, GlobalAveragePooling1D


def set_memory_growth():
    """
    Configure TensorFlow to use GPU memory growth if available.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} GPU(s) found and memory growth set.")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
    else:
        print("No GPU found. Using CPU.")


def create_model(input_shape, num_classes):
    """
    Create and compile the deep learning model.

    :param input_shape: tuple, shape of the input data
    :param num_classes: int, number of output classes
    :return: tf.keras.Model
    """
    signal_input = Input(shape=input_shape, name='data')

    # Module 1
    layer_1_a = Conv1D(filters=10, kernel_size=1, padding='same', activation='relu', name='1x1_a_3')(signal_input)
    layer_2_a = Conv1D(filters=10, kernel_size=1, padding='same', activation='relu', name='1x1_a_5')(signal_input)
    layer_2_a = Conv1D(filters=10, kernel_size=3, padding='same', activation='relu', name='1x5_a')(layer_2_a)
    mid_1_a = add([layer_1_a, layer_2_a])

    # Concatenated Module 1 with input
    mid_1_a = Conv1D(filters=9, kernel_size=1, padding='same', activation='relu', name='a_1x1_size_reduce')(mid_1_a)

    # Ending network
    before_flat = Conv1D(filters=6, kernel_size=1, padding='same', activation='relu', name='before_2_1x1_size_reduce')(
        mid_1_a)
    globelAverage = GlobalAveragePooling1D(data_format='channels_last')(before_flat)

    out = Dense(num_classes, activation='softmax', name='predictions')(globelAverage)
    model = Model(inputs=signal_input, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    set_memory_growth()
    # Example usage
    input_shape = (256, 6)  # Example input shape
    num_classes = 6  # Example number of classes
    model = create_model(input_shape, num_classes)
    model.summary()
