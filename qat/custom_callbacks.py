import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_average_loss_epoch

eps_float32 = np.finfo(np.float32).eps


class ScaleTrackingCallback(tf.keras.callbacks.Callback):
    """
    This callback is used to track and plot the scale values (scale_w and scale_b) of a custom layer's weights and biases
    after each epoch during training.
    """
    def __init__(self, layer):
        super(ScaleTrackingCallback, self).__init__()
        self.layer = layer
        self.scale_values_per_epoch_w = []
        self.scale_values_per_epoch_b = []

    def on_epoch_end(self, epoch, logs=None):
        scale_values_w = self.layer.scale_w.numpy().flatten()
        self.scale_values_per_epoch_w.append(scale_values_w)

        scale_values_b = self.layer.scale_b.numpy().flatten()
        self.scale_values_per_epoch_b.append(scale_values_b)

    def plot_scale_values(self, layer_name):
        plt.figure(figsize=(12, 8))

        # Plot each scale value trajectory
        for i in range(len(self.scale_values_per_epoch_w[0])):
            scale_trajectory = [epoch[i] for epoch in self.scale_values_per_epoch_w]
            plt.plot(range(1, len(self.scale_values_per_epoch_w) + 1), scale_trajectory, linestyle='-', marker='o')

        plt.xlabel('Epochs')
        plt.ylabel('Scale Values')
        plt.title('Scale Values of w per Epoch - ' + layer_name)
        plt.xticks(range(1, len(self.scale_values_per_epoch_w) + 1))  
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 8))

        # Plot each scale value trajectory
        for i in range(len(self.scale_values_per_epoch_b[0])):
            scale_trajectory = [epoch[i] for epoch in self.scale_values_per_epoch_b]
            plt.plot(range(1, len(self.scale_values_per_epoch_b) + 1), scale_trajectory, linestyle='-', marker='o')

        plt.xlabel('Epochs')
        plt.ylabel('Scale Values')
        plt.title('Scale Values of b per Epoch - ' + layer_name)
        plt.xticks(range(1, len(self.scale_values_per_epoch_b) + 1))  
        plt.legend()
        plt.show()


class AccuracyTrackingCallBack(tf.keras.callbacks.Callback):
    """
    This callback is used to track and plot the accuracy of the model after each epoch during training.
    """
    def __init__(self, layer):
        super(AccuracyTrackingCallBack, self).__init__()
        self.epoch_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_accuracy.append(logs['accuracy'])

    def plot_accuracy(self):
        epochs = range(1, len(self.epoch_accuracy) + 1)

        plt.figure(figsize=(12, 8))

        plt.plot(epochs, self.epoch_accuracy, 'r-', label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Epoch')
        plt.xticks(range(1, len(self.epoch_accuracy) + 1))  
        plt.legend()

        plt.tight_layout()
        plt.show()

# Something is going wrong when the penalty loss is being calculated
class LossTrackingCallback(tf.keras.callbacks.Callback):
    """
    This callback is used to track and plot the components of the loss (total loss and custom added loss)
    after each epoch during training. Note that, since we can't directly extract the custom added loss from the model's function
    that calculates the total loss, we are REcalculating the custom loss component separately.
    """
    def __init__(self, loss_function):
        self.penalty_rate = loss_function.penalty_rate
        self.loss_function_name = loss_function.get_name()

        self.epoch_scc_loss = []
        self.scale_penalty_loss = []

    def on_epoch_end(self, epoch, logs=None):
        # Path to the file
        total_loss_file = 'total_loss_log.txt'
        scale_loss_file = 'scale_loss_log.txt'

        start_line = epoch * (1875 + 313) + 1
        end_line = start_line + 1875 - 1


        scale_loss = calculate_average_loss_epoch(scale_loss_file, start_line=start_line, end_line=end_line)
        total_loss = calculate_average_loss_epoch(total_loss_file, start_line=start_line, end_line=end_line)
        print("\n", total_loss, "Lines: ", start_line, end_line)


        self.scale_penalty_loss.append(float(scale_loss))
        self.epoch_scc_loss.append(logs['loss'])

    def plot_loss(self):
        epochs = range(1, len(self.epoch_scc_loss) + 1)

        plt.figure(figsize=(12, 8))

        plt.fill_between(epochs, 0, self.epoch_scc_loss, color='grey', alpha=0.5, label='Total Loss')

        plt.plot(epochs, self.scale_penalty_loss, 'r-', label='Custom Added Loss', linewidth=2)
        plt.fill_between(epochs, 0, self.scale_penalty_loss, color='none', edgecolor='red', hatch='///', linewidth=0)

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss structure per epoch with penalty rate {self.penalty_rate} and loss function {self.loss_function_name}')
        plt.xticks(range(1, len(self.epoch_scc_loss) + 1))
        plt.legend()

        plt.tight_layout()
        plt.show()
        