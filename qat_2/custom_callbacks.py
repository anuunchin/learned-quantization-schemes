import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_average_loss_epoch
import os


eps_float32 = np.finfo(np.float32).eps


class ScaleTrackingCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer, log_dir):
        super(ScaleTrackingCallback, self).__init__()
        self.layer = layer
        self.log_file_path_scale_w = f"{log_dir}/{layer.__class__.__name__}_scale_w_{layer.scale_w.shape}.log"
        self.log_file_path_scale_b = f"{log_dir}/{layer.__class__.__name__}_scale_w_{layer.scale_w.shape}_scale_b_{layer.scale_b.shape}.log"
        self.log_file_path_w_rounded = f"{log_dir}/unique_{layer.__class__.__name__}_w_{layer.w.shape}.log"
        self.log_file_path_b_rounded = f"{log_dir}/unique_{layer.__class__.__name__}_b_{layer.b.shape}.log"
        self.log_file_path_w_b_rounded = f"{log_dir}/unique_combined_{layer.__class__.__name__}_w_b.log"

    def on_epoch_end(self, epoch, logs=None):
        scale_values_w = self.layer.scale_w.numpy().flatten()

        with open(self.log_file_path_scale_w, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in scale_values_w:
                file.write(f"{value}\n")

        scale_values_b = self.layer.scale_b.numpy().flatten()

        with open(self.log_file_path_scale_b, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in scale_values_b:
                file.write(f"{value}\n")

        w_quantized_rounded_unique = np.unique(tf.floor(self.layer.w / self.layer.scale_w).numpy())
        with open(self.log_file_path_w_rounded, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in w_quantized_rounded_unique:
                file.write(f"{value}\n")

        b_quantized_rounded_unique = np.unique(tf.floor(self.layer.b / self.layer.scale_b).numpy())
        with open(self.log_file_path_b_rounded, 'a') as file:
            file.write(f"Epoch {epoch}\n")
            for value in b_quantized_rounded_unique:
                file.write(f"{value}\n")

        combined_quantized = np.union1d(w_quantized_rounded_unique, b_quantized_rounded_unique)

        log_entry = f"Epoch {epoch}\n" + " ".join(map(str, combined_quantized)) + "\n"

        with open(self.log_file_path_w_b_rounded, 'a') as file:
            file.write(log_entry)


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

