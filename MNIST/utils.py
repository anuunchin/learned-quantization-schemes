import numpy as np
import tensorflow as tf
import os


def print_model_structure(model, folder_name, filename="model_structure.txt"):
    """
    Prints the structure of the given model, including each layer's input and output shapes.
    If a layer has scale weights and biases, their shapes are also printed.
    """
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, filename)

    with open(file_path, 'w') as f:
        # Redirect print output to the file
        original_stdout = os.sys.stdout
        os.sys.stdout = f

        print("\n" + "-" * 80)  # Add lines of dashes before

        print("MODEL STRUCTURE")

        for i, layer in enumerate(model.layers):
            print(f"\nLAYER {i}: {layer}")

#            attributes = dir(layer)
#            for attr in attributes:
#                try:
                    # Print the attribute and its value
#                    print(f"  - {attr}: {getattr(layer, attr)}")
#                except Exception as e:
#                    print(f"  - {attr}: (Could not retrieve: {str(e)})")


            print(f"  - Input Shape: {layer.input}")
            print(f"  - Output Shape: {layer.output}")
            if hasattr(layer, 'get_scale_w') and layer.get_scale_w() is not None:
                print(f"  - Scale Shape of w: {layer.get_scale_w().shape}")
            if hasattr(layer, 'get_scale_b') and layer.get_scale_b() is not None:
                print(f"  - Scale Shape of b: {layer.get_scale_b().shape}")

        print("-" * 80)  # Add lines of dashes after

        # Restore original stdout
        os.sys.stdout = original_stdout

    # Read the file and print its contents to the terminal
    with open(file_path, 'r') as f:
        file_contents = f.read()
        print(file_contents)


def count_unique_values(model, folder_name, filename="unique_values.txt"):
    """
    Counts and prints the number of unique values for weights and biases of each custom layer in the model.
    Also prints the number of unique values for the quantized weights and biases.
    Saves the output to a text file and then prints the file contents to the terminal.
    """
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, filename)
    
    with open(file_path, 'w') as f:
        # Redirect print output to the file
        original_stdout = os.sys.stdout
        os.sys.stdout = f

        print("\n" + "-" * 80)  # Add lines of dashes before
        print("NUMBER OF UNIQUE VALUES FOR W AND B OF EACH CUSTOM LAYER\n")

        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'get_scale_w') and layer.get_scale_w() is not None:
                w = layer.w.numpy()
                b = layer.b.numpy()
                scale_w = layer.get_scale_w().numpy()
                scale_b = layer.get_scale_b().numpy()

                w_unique_values = len(np.unique(w))
                b_unique_values = len(np.unique(b))

                w_quantized_unique_values = len(np.unique(tf.floor(w / scale_w).numpy()))
                w_quantized_min_value = tf.reduce_min(tf.abs(tf.floor(w / scale_w))).numpy()
                w_quantized_max_value = tf.reduce_max(tf.abs(tf.floor(w / scale_w))).numpy()



                b_quantized_unique_values = len(np.unique(tf.floor(b / scale_b).numpy()))
                b_quantized_min_value = tf.reduce_min(tf.abs(tf.floor(b / scale_b))).numpy()
                b_quantized_max_value = tf.reduce_max(tf.abs(tf.floor(b / scale_b))).numpy()

                print("LAYER WITH ID:", i)
                print("Unique values in w: ", w_unique_values)
                print("Unique values in quantized w: ", w_quantized_unique_values)
                print("Min abs value in quantized w: ", w_quantized_min_value)
                print("Max abs value in quantized w: ", w_quantized_max_value)
                print("The values: ", np.unique(tf.floor(w / scale_w).numpy()), "\n")

                print("Unique values in b: ", b_unique_values)
                print("Unique values in quantized b: ", b_quantized_unique_values)
                print("Min abs value in quantized b: ", b_quantized_min_value)
                print("Max abs value in quantized b: ", b_quantized_max_value)
                print("The values: ", np.unique(tf.floor(b / scale_b).numpy()), "\n")

            else:
                print("LAYER WITH ID:", i, "DOESN'T HAVE SCALE FACTOR VALUES OR MEANINGFUL ONES\n")

        print("\n" + "-" * 80)  # Add lines of dashes before

        # Restore original stdout
        os.sys.stdout = original_stdout

    # Read the file and print its contents to the terminal
    with open(file_path, 'r') as f:
        file_contents = f.read()
        print(file_contents)
        


def plot_scatter(unique_values, counts, log_dir, layer_idx, variable):
    plt.figure(figsize=(10, 6))
    plt.scatter(unique_values, counts, color='grey')
    plt.xlabel('Unique Values')
    plt.ylabel('Occurrences')
    plt.title(f'Scatter Plot of Occurrences of Unique Values in {variable} in Layer {layer_idx}')
    plt.grid(True)

    plots_dir = os.path.join(log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'Scatter Plot of Occurrences of Unique Values in {variable} in Layer {layer_idx}'))


    plt.show()


def plot_values(unique_values, counts, log_dir, layer_idx, variable):
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(unique_values, counts, width=0.5, color='grey', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Unique Values')
    plt.ylabel('Occurrences')
    plt.title(f'Histogram of Unique Values in {variable}')

    # Display the plot
    plt.show()


def count_unique_values_2(model, folder_name, filename="unique_values.txt"):
    """
    Counts and prints the number of unique values for weights and biases of each custom layer in the model.
    Also prints the number of unique values for the quantized weights and biases.
    Saves the output to a text file and then prints the file contents to the terminal.
    """
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, filename)
    
    values = []

    with open(file_path, 'w') as f:
        # Redirect print output to the file
        original_stdout = os.sys.stdout
        os.sys.stdout = f

        print("\n" + "-" * 80)  # Add lines of dashes before
        print("NUMBER OF UNIQUE VALUES FOR W AND B OF EACH CUSTOM LAYER\n")

        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'get_scale_w') and layer.get_scale_w() is not None:
                
                w = layer.w.numpy()
                unique_values, counts = np.unique(w, return_counts=True)
                plot_scatter(unique_values, counts, folder_name, i, variable=w)

                b = layer.b.numpy()
                unique_values, counts = np.unique(b, return_counts=True)
                plot_scatter(unique_values, counts, folder_name, i, variable=b)

                scale_w = layer.get_scale_w().numpy()
                scale_b = layer.get_scale_b().numpy()

                w_unique_values = len(np.unique(w))
                b_unique_values = len(np.unique(b))

                w_quantized_values = tf.floor(w / scale_w).numpy()
                unique_values, counts = np.unique(w_quantized_values, return_counts=True)
                plot_scatter(unique_values, counts, folder_name, i, variable=w_quantized_values)

                b_quantized_values = tf.floor(b / scale_b).numpy()
                unique_values, counts = np.unique(b_quantized_values, return_counts=True)
                plot_scatter(unique_values, counts, folder_name, i, variable=b_quantized_values)

                w_quantized_unique_values = len(np.unique(w_quantized_values))
                b_quantized_unique_values = len(np.unique(b_quantized_values))

                print("LAYER WITH ID:", i)
                print("Unique values in w: ", w_unique_values)
                print("Unique values in quantized w: ", w_quantized_unique_values)
                print("Unique values in b: ", b_unique_values)
                print("Unique values in quantized b: ", b_quantized_unique_values, "\n")
            else:
                print("LAYER WITH ID:", i, "DOESN'T HAVE SCALE FACTOR VALUES OR MEANINGFUL ONES\n")

        print("\n" + "-" * 80)  # Add lines of dashes before

        # Restore original stdout
        os.sys.stdout = original_stdout

    # Read the file and print its contents to the terminal
    with open(file_path, 'r') as f:
        file_contents = f.read()
        print(file_contents)
        


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

def count_unique_values_and_plot_histograms(model, folder_name, filename="unique_values.txt"):
    """
    Counts and prints the number of unique values for weights and biases of each custom layer in the model.
    Also prints the number of unique values for the quantized weights and biases.
    Saves the output to a text file and then prints the file contents to the terminal.
    Additionally, creates histograms of the frequency of each unique value for quantized and non-quantized weights.
    """
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, filename)
    
    all_w_values = []
    all_w_quantized_values = []
    
    with open(file_path, 'w') as f:
        # Redirect print output to the file
        original_stdout = os.sys.stdout
        os.sys.stdout = f

        print("\n" + "-" * 80)  # Add lines of dashes before
        print("NUMBER OF UNIQUE VALUES FOR W AND B OF EACH CUSTOM LAYER\n")

        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'get_scale_w') and layer.get_scale_w() is not None:
                w = layer.w.numpy().flatten()  # Flatten the weight matrix to count unique values
                scale_w = layer.get_scale_w().numpy()

                w_quantized = tf.floor(w / scale_w).numpy().flatten()

                all_w_values.extend(w)
                all_w_quantized_values.extend(w_quantized)

                print("LAYER WITH ID:", i)
                print("Unique values in w: ", len(np.unique(w)))
                print("Unique values in quantized w: ", len(np.unique(w_quantized)), "\n")
            else:
                print("LAYER WITH ID:", i, "DOESN'T HAVE SCALE FACTOR VALUES OR MEANINGFUL ONES\n")

        print("\n" + "-" * 80)  # Add lines of dashes before

        # Restore original stdout
        os.sys.stdout = original_stdout

    # Read the file and print its contents to the terminal
    with open(file_path, 'r') as f:
        file_contents = f.read()
        print(file_contents)
    
    # Count the frequency of each unique value
    w_value_counts = Counter(all_w_values)
    w_quantized_value_counts = Counter(all_w_quantized_values)

    # Plot histograms
    plt.figure(figsize=(12, 6))

    # Histogram for the frequency of each unique value in w
    plt.subplot(1, 2, 1)
    plt.bar(w_value_counts.keys(), w_value_counts.values(), color='blue', alpha=0.7)
    plt.title('Histogram of Frequencies of Unique Values in W')
    plt.xlabel('Unique Values')
    plt.ylabel('Frequency')

    # Histogram for the frequency of each unique value in quantized w
    plt.subplot(1, 2, 2)
    plt.bar(w_quantized_value_counts.keys(), w_quantized_value_counts.values(), color='green', alpha=0.7)
    plt.title('Histogram of Frequencies of Unique Values in Quantized W')
    plt.xlabel('Unique Values')
    plt.ylabel('Frequency')

    # Show the plots
    plt.tight_layout()
    plt.show()

# Example usage:
# count_unique_values_and_plot_histograms(model, 'output_folder')



def calculate_average_loss_epoch(file_path, start_line, end_line):
    """
    Calculate the average loss over a specified range of lines in a log file.
    """
    loss_values = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i + 1 < start_line:
                continue
            if i + 1 > end_line:
                break
            # Extract the loss value from the line
            if line.startswith("Loss"):
                _, value = line.split(':')
                loss_values.append(float(value.strip()))

    if loss_values:
        average_loss = sum(loss_values) / len(loss_values)
        return average_loss
    else:
        return None
    