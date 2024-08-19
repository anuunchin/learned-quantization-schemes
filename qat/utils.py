import numpy as np
import tensorflow as tf
import os


def print_model_structure(model):
    """
    Prints the structure of the given model, including each layer's input and output shapes.
    If a layer has scale weights and biases, their shapes are also printed.
    """
    print("\n" + "-" * 80)  # Add lines of dashes before

    print("MODEL STRUCTURE")

    for i, layer in enumerate(model.layers):
        print(f"\nLAYER {i}: {layer}")
        print(f"  - Input Shape: {layer.input_shape}")
        print(f"  - Output Shape: {layer.output_shape}")
        if hasattr(layer, 'get_scale_w') and layer.get_scale_w() is not None:
            print(f"  - Scale Shape of w: {layer.get_scale_w().shape}")
        if hasattr(layer, 'get_scale_b') and layer.get_scale_b() is not None:
            print(f"  - Scale Shape of b: {layer.get_scale_b().shape}")

    print("-" * 80)  # Add lines of dashes after


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
                b_quantized_unique_values = len(np.unique(tf.floor(b / scale_b).numpy()))

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
    




def calculate_average_loss_batch(file_path, start_line, end_line):
    """
    This function will be used in the Losstracking callback, meaning it will be called on each epoch end
    """