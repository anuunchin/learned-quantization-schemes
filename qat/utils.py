import numpy as np
import tensorflow as tf


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


def count_unique_values(model):
    """
    Counts and prints the number of unique values for weights and biases of each custom layer in the model.
    Also prints the number of unique values for the quantized weights and biases.
    """
    print("\n" + "-" * 80)  # Add lines of dashes before

    print("NUMBER OF UNIQUE VALUES FOR W AND B OF EACH CUSTOM LAYER")

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
    
            print("\nLAYER WITH ID:", i)
            print("Unique values in w: ", w_unique_values)
            print("Unique values in quantized w: ", w_quantized_unique_values)
            print("Unique values in b: ", b_unique_values)
            print("Unique values in quantized b: ", b_quantized_unique_values)
        else:
            print("LAYER WITH IF: ", i, "DOESN'T HAVE SCALE FACTOR VALUES OR MEANINGFUL ONES")

    print("\n" + "-" * 80)  # Add lines of dashes before


def calculate_average_loss_epoch(file_path, start_line, end_line):
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
