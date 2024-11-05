import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

from custom_loss_functions import SCCE, SCCEInverse, SCCEMinMaxBin, SCCEMaxBin, SCCEDifference
from custom_layers import DefaultDense, RowWiseQuantized, RowWiseQuantizedSTE, ColumnWiseQuantized, ColumnWiseQuantizedSTE
from custom_callbacks import ScaleTrackingCallback, AccuracyTrackingCallBack, LossTrackingCallbackNew
from utils import print_model_structure, count_unique_values, count_unique_values_and_plot_histograms, count_unique_values_2

from datetime import datetime

from plot_scripts import plot_loss

# Prepare the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.expand_dims(x_train, -1)  # CNNs typically expect input data to be 4D
x_test = np.expand_dims(x_test, -1)


def initialize_quantized_model(input_shape = (28, 28, 1)):
    input_layer = Input(shape=input_shape)
    flatten_layer = Flatten()(input_layer)
    quantized_dense_layer = RowWiseQuantized(128)(flatten_layer)
    quantized_dense_layer_2 = tf.keras.activations.relu(quantized_dense_layer)
    output_layer = RowWiseQuantized(10)(quantized_dense_layer_2)
    output_layer_2 = tf.keras.activations.softmax(output_layer)
    quantized_model = Model(inputs=input_layer, outputs=output_layer_2)

    return quantized_model


def prepare_model_dir(model, penalty_rate, run_timestamp):
    log_dir = f'logs/{run_timestamp}_pr_{penalty_rate}'
    os.makedirs(log_dir)
    print_model_structure(model, log_dir, filename="quantized_model_structure.txt")
    return log_dir


def initialize_loss_function(model, penalty_rate, log_dir, loss_func = SCCEMaxBin):
    # Initialize your custom loss function
    loss_function = loss_func(
        layers=[
            model.get_layer(index=2),
            model.get_layer(index=3)
        ],
        penalty_rate=penalty_rate,
        row_wise=1,
        log_dir = log_dir  # 1 = True = scale factor values are applied row-wise, must match the used custom layer (RowWiseQuantized)
                # 0 = False = scale factor values are applied column-wise, must match the used custom layer (ColumnWiseQuantized)
    )
    return loss_function


def compile_model(model, learning_rate, loss_function):
    # Compile your models
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_function.compute_total_loss,
        metrics=['accuracy']
    )

def initialize_callbacks(model, loss_function, log_dir, validation_data):
    # Initialize your callbacks
    scale_tracking_callback_first_dense_layer = ScaleTrackingCallback(model.get_layer(index=2), log_dir)
    scale_tracking_callback_second_dense_layer = ScaleTrackingCallback(model.get_layer(index=3), log_dir)
    penalty_callback = LossTrackingCallbackNew(loss_function=loss_function, validation_data=validation_data, interval=val_interval, log_dir=log_dir)
    accuracy_callback = AccuracyTrackingCallBack(model.get_layer(index=3), log_dir)

    # Return callbacks as a dictionary
    callbacks = {
        'scale_tracking_callback_first_dense_layer': scale_tracking_callback_first_dense_layer,
        'scale_tracking_callback_second_dense_layer': scale_tracking_callback_second_dense_layer,
        'penalty_callback': penalty_callback,
        'accuracy_callback': accuracy_callback
    }

    return callbacks

def train_model(model, epochs, validation_data, batch_size, **callbacks):
    # Train your model
    model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=list(callbacks.values()),  
        batch_size=batch_size
    )

def evaluate_model(model):
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Quantized Model Test Accuracy: {accuracy}')
    return accuracy