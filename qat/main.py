import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

from custom_loss_functions import SCCE, SCCEInverse, SCCEMinMaxBin, SCCEMaxBin, SCCEDifference
from custom_layers import DefaultDense, RowWiseQuantized, RowWiseQuantizedSTE, ColumnWiseQuantized, ColumnWiseQuantizedSTE
from custom_callbacks import ScaleTrackingCallback, LossTrackingCallback, AccuracyTrackingCallBack
from utils import print_model_structure, count_unique_values

# Prepare the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.expand_dims(x_train, -1)  # CNNs typically expect input data to be 4D
x_test = np.expand_dims(x_test, -1)

input_layer = Input(shape=(28, 28, 1))
flatten_layer = Flatten()(input_layer)
quantized_dense_layer = RowWiseQuantizedSTE(128, activation='relu')
dense_output = quantized_dense_layer(flatten_layer)
output_layer = RowWiseQuantizedSTE(10, activation='softmax')(dense_output)
quantized_model = Model(inputs=input_layer, outputs=output_layer)

print_model_structure(quantized_model)

# Initialize your custom loss function
loss_function = SCCEMinMaxBin(
    layers=[
        quantized_model.get_layer(index=2),
        quantized_model.get_layer(index=3)
    ],
    penalty_rate=0.1,
    row_wise=1 # 1 = True = row-wise, 0 = False = column-wise
)

# Compile your model
quantized_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=loss_function.compute_total_loss,
    metrics=['accuracy']
)

# Initialize your callbacks
scale_tracking_callback_first_dense_layer = ScaleTrackingCallback(quantized_model.get_layer(index=2))
scale_tracking_callback_second_dense_layer = ScaleTrackingCallback(quantized_model.get_layer(index=3))
penalty_callback = LossTrackingCallback(loss_function=loss_function)
accuracy_callback = AccuracyTrackingCallBack(quantized_model.get_layer(index=3))

# Train your model
quantized_model.fit(
    x_train, y_train,
    epochs=2,
    validation_data=(x_test, y_test),
    callbacks=[
        scale_tracking_callback_first_dense_layer,
        scale_tracking_callback_second_dense_layer,
        penalty_callback,
        accuracy_callback
    ]
)

# Evaluate
loss, accuracy = quantized_model.evaluate(x_test, y_test)
print(f'Quantized Model Test Accuracy: {accuracy}')

# Plot
scale_tracking_callback_first_dense_layer.plot_scale_values(layer_name="Quantized Dense Layer 1")
scale_tracking_callback_second_dense_layer.plot_scale_values(layer_name="Quantized Dense Layer 2")
penalty_callback.plot_loss()
accuracy_callback.plot_accuracy()

# Count number of unique values for w and b before and after quantization for each custom layer
count_unique_values(quantized_model)