import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_average_loss_epoch
import os

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

    def plot_scale_values(self, layer_name, folder_name):
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
        plt.savefig(os.path.join(folder_name, 'Scale Values of w per Epoch - ' + layer_name + '.png'))

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
        plt.savefig(os.path.join(folder_name, 'Scale Values of b per Epoch - ' + layer_name + '.png'))

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

    def plot_accuracy(self, folder_name):
        epochs = range(1, len(self.epoch_accuracy) + 1)

        plt.figure(figsize=(12, 8))

        plt.plot(epochs, self.epoch_accuracy, 'r-', label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Epoch')
        plt.xticks(range(1, len(self.epoch_accuracy) + 1))  
        plt.legend()
        plt.savefig(os.path.join(folder_name, 'Accuracy per Epoch.png'))

        plt.tight_layout()
        plt.show()


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
        total_loss_file = 'logs/total_loss_log.txt'
        scale_loss_file = 'logs/scale_loss_log.txt'

        start_line = epoch * (1875 + 313) + 1
        end_line = start_line + 1875 - 1

        scale_loss = calculate_average_loss_epoch(scale_loss_file, start_line=start_line, end_line=end_line)
        total_loss = calculate_average_loss_epoch(total_loss_file, start_line=start_line, end_line=end_line)

        tolerate = 0.0002

        if total_loss is not None and abs(total_loss - logs['loss']) > tolerate:
            print(f"\nSomething wrong in loss calculation: calculated total_loss = {total_loss}, expected_loss = {logs['loss']}")
#            raise ValueError(f"Something wrong in loss calculation: calculated total_loss = {total_loss}, expected_loss = {logs['loss']}")

        self.scale_penalty_loss.append(float(scale_loss))
        self.epoch_scc_loss.append(logs['loss'])

    def plot_loss(self, folder_name):
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
        plt.savefig(os.path.join(folder_name, f'Loss structure per epoch with penalty rate {self.penalty_rate} and loss function {self.loss_function_name}.png'))

        plt.tight_layout()
        plt.show()


class LossTrackingCallbackNew(tf.keras.callbacks.Callback):
    """
    This callback is used to track and plot the components of the loss (total loss and custom added loss)
    after each epoch during training. Note that, since we can't directly extract the custom added loss from the model's function
    that calculates the total loss, we are REcalculating the custom loss component separately.
    """
    def __init__(self, loss_function, validation_data, interval):
        self.penalty_rate = loss_function.penalty_rate
        self.loss_function_name = loss_function.get_name()
        self.validation_data = validation_data
        self.interval = interval

        self.epoch_scc_loss = []
        self.scale_penalty_loss = []
        
        self.batch_scc_loss = []
        self.batch_scale_penalty_loss = []
        self.validation_loss_per_interval = {}

        self.current_epoch = None
        self.batch_count = 0


    def on_train_batch_end(self, batch, logs=None):
        # Log batch losses
        total_loss_file = 'logs/total_loss_log.txt'
        scale_loss_file = 'logs/scale_loss_log.txt'

        batch_line = self.current_epoch * (1875 + 313) + (int(batch / self.interval) * 313) + batch + 1

        batch_total_loss = calculate_average_loss_epoch(total_loss_file, start_line=batch_line, end_line=batch_line)
        batch_scale_loss = calculate_average_loss_epoch(scale_loss_file, start_line=batch_line, end_line=batch_line)


        # Note that the below part is commented out because the logs['loss'] has values averaged between the batches
        """
#        tolerate = 0.0002

#        if batch_total_loss is not None and abs(batch_total_loss - logs['loss']) > tolerate:
#            print(f"\nBATCH: Something wrong in loss calculation: calculated total_loss = {batch_total_loss}, expected_loss = {logs['loss']}")
#            raise ValueError(f"Something wrong in loss calculation: calculated total_loss = {total_loss}, expected_loss = {logs['loss']}")
"""
        self.batch_scale_penalty_loss.append(float(batch_scale_loss))
        self.batch_scc_loss.append(batch_total_loss)

        # trigger a validation every interval batches
        self.batch_count += 1
        if self.batch_count % self.interval == 0:
            val_loss, val_accuracy = self.model.evaluate(*self.validation_data, verbose=0)
            #print(f'Custom validation loss after {self.batch_count} batches: {val_loss:.4f}')
            #print(f'Custom validation accuracy after {self.batch_count} batches: {val_accuracy:.4f}')
            self.validation_loss_per_interval[self.batch_count] = val_loss

    def on_train_begin(self, logs=None):
        # Clean / create files for logging
        # TODO modify the batch loss calculation so that the plot function reads from a file, otherwise it fills memory
        log_file_paths = {
                    "batch_total_loss_logs" : "logs/batch_total_loss.txt",
                    "batch_scale_loss_logs" : "logs/batch_scale_loss.txt",
                    "validation_loss" : "logs/interval_validation_loss.txt",
        }

        for _, path in log_file_paths.items():
            if os.path.exists(path):
                with open(path, 'w') as f:
                    f.write('')
            else:
                open(path, 'w').close()


    def on_epoch_begin(self, epoch, logs=None):
        # Update current epoch
        self.current_epoch = epoch        


    def on_epoch_end(self, epoch, logs=None):
        # This doesn't properly working if custom validation interval is set!!
        total_loss_file = 'logs/total_loss_log.txt'
        scale_loss_file = 'logs/scale_loss_log.txt'

        start_line = epoch * (1875 + 313) + 1
        end_line = start_line + 1875 - 1

        scale_loss = calculate_average_loss_epoch(scale_loss_file, start_line=start_line, end_line=end_line)
        total_loss = calculate_average_loss_epoch(total_loss_file, start_line=start_line, end_line=end_line)

        tolerate = 0.0002

        if total_loss is not None and abs(total_loss - logs['loss']) > tolerate:
            print(f"\nSomething wrong in loss calculation: calculated total_loss = {total_loss}, expected_loss = {logs['loss']}")
#            raise ValueError(f"Something wrong in loss calculation: calculated total_loss = {total_loss}, expected_loss = {logs['loss']}")

        self.scale_penalty_loss.append(float(scale_loss))
        self.epoch_scc_loss.append(logs['loss'])


    def plot_loss(self, folder_name):
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
        plt.savefig(os.path.join(folder_name, f'Loss structure per epoch with penalty rate {self.penalty_rate} and loss function {self.loss_function_name}.png'))

        plt.tight_layout()
        plt.show()

    def plot_loss_per_batch(self, folder_name, n = 20):
        batches = range(1 + n, len(self.batch_scc_loss) + 1)

        plt.figure(figsize=(30,30))  # Adjust height if necessary

        plt.fill_between(batches, 0, self.batch_scc_loss[n:], color='grey', alpha=0.5, label='Total loss per batch')
        
        plt.plot(batches, self.batch_scc_loss[n:], 'r-', label="Custom Added Loss", linewidth=2)
        plt.fill_between(batches, 0, self.batch_scale_penalty_loss[n:], color='none', edgecolor='red', hatch='///', linewidth=0)


        # Plot the validation loss at intervals
        validation_batches = list(self.validation_loss_per_interval.keys())
        validation_losses = list(self.validation_loss_per_interval.values())
        plt.scatter(validation_batches, validation_losses, color='blue', label='Validation Loss', zorder=3)  # Use zorder to make sure it's on top
        plt.plot(validation_batches, validation_losses, 'b--', linewidth=2)  # Optionally connect the points with a dashed line


        plt.xlabel('Batches', labelpad=40)
        plt.ylabel('Loss')
        plt.title(f'Loss structure per batch with penalty rate {self.penalty_rate} and loss function {self.loss_function_name}')

        # Set x-ticks to every 1875th batch and label them by epoch
        num_batches_per_epoch = 1875
        num_epochs = len(self.batch_scc_loss) // num_batches_per_epoch
        xticks = []
        xtick_labels = []
        epoch_ticks = []

        for i in range(num_epochs + 1):
            # Add batch number tick
            xticks.append(i * num_batches_per_epoch)
            xtick_labels.append(str(i * num_batches_per_epoch))
            
            # Add epoch label in the middle
            if i > 0:
                epoch_tick = i * num_batches_per_epoch - num_batches_per_epoch // 2
                epoch_ticks.append(epoch_tick)
                xticks.append(epoch_tick)
                xtick_labels.append(f'Epoch {i}')

        plt.xticks(xticks, labels=[''] * len(xticks))  # Set the x-ticks but without labels

        ax = plt.gca()  # Get the current axis

        # Remove ticks for epoch labels
        ax.tick_params(axis='x', which='both', length=0)  # Hide all ticks first
        ax.set_xticks([tick for tick in xticks if tick not in epoch_ticks])  # Reset ticks for batch numbers only

        for i, tick in enumerate(xticks):
            if tick in epoch_ticks:
                ax.text(tick, ax.get_ylim()[0] - 0.02 * ax.get_ylim()[1], xtick_labels[i], ha='center', va='top', rotation=0, color='blue')
            else:
                ax.text(tick, ax.get_ylim()[0] - 0.02 * ax.get_ylim()[1], xtick_labels[i], ha='center', va='top', rotation=0)

        plt.legend()
        plt.savefig(os.path.join(folder_name, f'Loss structure per batch with penalty rate {self.penalty_rate} and loss function {self.loss_function_name}.png'))
