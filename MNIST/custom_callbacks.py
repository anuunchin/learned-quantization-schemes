import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_average_loss_epoch
import os


eps_float32 = np.finfo(np.float32).eps


class QuantizedDataTrackingCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer, log_dir):
        super(QuantizedDataTrackingCallback, self).__init__()
        self.layer = layer
        self.log_file_path_weights = f"{log_dir}/{layer.__class__.__name__}_scale_{layer.scale.shape}.log"

    def on_epoch_end(self, epoch, logs=None):
        weights = self.layer.scale.numpy().flatten()

        with open(self.log_file_path_weights, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in weights:
                file.write(f"{value}\n")


class NestedScaleTrackingCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer, log_dir):
        super(NestedScaleTrackingCallback, self).__init__()
        self.layer = layer
        self.log_file_path_weights = f"{log_dir}/{layer.W.__class__.__name__}_scale_{layer.W.shape}.log"
        self.log_file_path_biases = f"{log_dir}/{layer.b.__class__.__name__}_scale_{layer.b.shape}.log"

    def on_epoch_end(self, epoch, logs=None):
        weights = self.layer.W.numpy().flatten()

        with open(self.log_file_path_weights, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in weights:
                file.write(f"{value}\n")

        biases = self.layer.b.numpy().flatten()

        with open(self.log_file_path_biases, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in biases:
                file.write(f"{value}\n")

class NestedScaleTrackingCallbackNew(tf.keras.callbacks.Callback):
    def __init__(self, layer, log_dir):
        super(NestedScaleTrackingCallbackNew, self).__init__()
        self.layer = layer
        self.qw_layer = layer.nested_q_w_layer
        self.qb_layer = layer.nested_q_b_layer
        self.w_scale = layer.nested_q_w_layer.scale
        self.b_scale = layer.nested_q_b_layer.scale

        self.log_file_path_weights = f"{log_dir}/{layer.W.name}_{layer.W.shape}.log"
        self.log_file_path_biases = f"{log_dir}/{layer.b.name}_{layer.b.shape}.log"

        self.log_file_path_w_scale = f"{log_dir}/{layer.W.name}_{layer.W.shape}_{self.w_scale.name}_{self.w_scale.shape}.log"
        self.log_file_path_b_scale = f"{log_dir}/{layer.b.name}_{layer.b.shape}_{self.b_scale.name}_{self.b_scale.shape}.log"

        self.log_file_path_wq = f"{log_dir}/Quantized_{layer.W.name}_{layer.W.shape}_{self.w_scale.name}_{self.w_scale.shape}.log"
        self.log_file_path_bq = f"{log_dir}/Quantized_{layer.b.name}_{layer.b.shape}_{self.b_scale.name}_{self.b_scale.shape}.log"

    def on_epoch_end(self, epoch, logs=None):
        weights = self.layer.W.numpy().flatten()

        with open(self.log_file_path_weights, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in weights:
                file.write(f"{value}\n")

        biases = self.layer.b.numpy().flatten()

        with open(self.log_file_path_biases, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in biases:
                file.write(f"{value}\n")

        weight_scales = self.w_scale.numpy().flatten()

        with open(self.log_file_path_w_scale, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in weight_scales:
                file.write(f"{value}\n")

        bias_scales = self.b_scale.numpy().flatten()

        with open(self.log_file_path_b_scale, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in bias_scales:
                file.write(f"{value}\n")

    def on_train_end(self,  logs=None):

        inputs_quantized_nonrounded = self.layer.W / self.w_scale  # changes in the scale affect the output inversely

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_scaled_back = inputs_quantized_rounded.numpy().flatten()

        with open(self.log_file_path_wq, 'a') as file:
            for value in inputs_quantized_scaled_back:
                file.write(f"{value}\n")

        inputs_quantized_nonrounded = self.layer.b / self.b_scale  # changes in the scale affect the output inversely

        inputs_quantized_rounded = tf.floor(inputs_quantized_nonrounded)

        inputs_quantized_scaled_back = inputs_quantized_rounded.numpy().flatten()

        with open(self.log_file_path_bq, 'a') as file:
            for value in inputs_quantized_scaled_back:
                file.write(f"{value}\n")


class AccuracyLossTrackingCallBack(tf.keras.callbacks.Callback):
    def __init__(self, layer, log_dir, accuracy_file = "accuracy.log", loss_file = "loss.log"):
        super(AccuracyLossTrackingCallBack, self).__init__()
        self.accuracy_log_file_path = f"{log_dir}/{accuracy_file}"
        self.loss_log_file_path = f"{log_dir}/{loss_file}"
        self.val_accuracy_log_file_path = f"{log_dir}/val_{accuracy_file}"
        self.val_loss_log_file_path = f"{log_dir}/val_{loss_file}"

    def on_epoch_end(self, epoch, logs=None):
        with open(self.val_accuracy_log_file_path, 'a') as file:
            file.write(f"{logs['val_accuracy']}\n")

        with open(self.val_loss_log_file_path, 'a') as file:
            file.write(f"{logs['val_loss']}\n")


class LossTrackingCallbackNew(tf.keras.callbacks.Callback):
    def __init__(self, loss_function, validation_data, interval, log_dir):
        self.validation_data = validation_data
        self.interval = interval
        self.batch_count = 0
        self.validation_loss_file = f"{log_dir}/validation_loss.log"
        self.log_dir = log_dir

        self.accuracy_log_file_path = f"{log_dir}/accuracy.log"
        self.loss_log_file_path = f"{log_dir}/loss.log"
        self.train_loss = []
        self.train_accuracy = []


    def on_train_batch_begin(self, batch, logs=None):
        with open(f"{self.log_dir}/total_loss_log.log", 'a') as file:
            file.write(f"Train batch {batch + 1}: ")

        with open(f"{self.log_dir}/scale_loss_log.log", 'a') as file:
            file.write(f"Train batch {batch + 1}: ")

    def on_train_batch_end(self, batch, logs=None):
        self.train_loss.append(logs["loss"])
        self.train_accuracy.append(logs["accuracy"])

        # trigger a validation every interval batches
        self.batch_count += 1
        if self.batch_count % self.interval == 0:
            val_loss, val_accuracy = self.model.evaluate(*self.validation_data, verbose=0)
            with open(self.validation_loss_file, 'a') as file:
                file.write(f'{val_loss}\n')
    
    def on_epoch_end(self, epoch, logs=None):
        with open(self.accuracy_log_file_path, 'a') as file:
            file.write(f"{np.mean(self.train_accuracy)}\n")
        
        self.train_accuracy = []

        with open(self.loss_log_file_path, 'a') as file:
            file.write(f"{np.mean(self.train_loss)}\n")

        self.train_loss = []

    def on_epoch_begin(self, epoch, logs=None):
        with open(f"{self.log_dir}/total_loss_log.log", 'a') as file:
            file.write(f"Epoch {epoch + 1}:\n")

        with open(f"{self.log_dir}/scale_loss_log.log", 'a') as file:
            file.write(f"Epoch {epoch + 1}:\n")

        # Check if the bins_w log file exists
        bins_w_path = f"{self.log_dir}/bins_w.log"
        if os.path.exists(bins_w_path):
            with open(bins_w_path, 'a') as file:
                file.write(f"Epoch {epoch + 1}:\n")

        # Check if the bins_b log file exists
        bins_b_path = f"{self.log_dir}/bins_b.log"
        if os.path.exists(bins_b_path):
            with open(bins_b_path, 'a') as file:
                file.write(f"Epoch {epoch + 1}:\n")

        # Check if the bins_average log file exists
        bins_b_path = f"{self.log_dir}/bins_average.log"
        if os.path.exists(bins_b_path):
            with open(bins_b_path, 'a') as file:
                file.write(f"Epoch {epoch + 1}:\n")

    def on_test_batch_begin(self, batch, logs=None):
        with open(f"{self.log_dir}/total_loss_log.log", 'a') as file:
            file.write(f"Test batch {batch + 1}: ")

        with open(f"{self.log_dir}/scale_loss_log.log", 'a') as file:
            file.write(f"Test batch {batch + 1}: ")

