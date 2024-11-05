import matplotlib.pyplot as plt
import os
import math
import numpy as np

def plot_scale_values(layer, log_dir):

    # Define paths to the log files
    log_file_path_w = f"{log_dir}/{layer.__class__.__name__}_scale_w_{layer.scale_w.shape}.log"
    log_file_path_b = f"{log_dir}/{layer.__class__.__name__}_scale_w_{layer.scale_w.shape}_scale_b_{layer.scale_b.shape}.log"

    def process_log_file(file_path):
        all_values = []
        epochs = []
        current_values = []
        
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                if line.startswith('Epoch'):
                    current_epoch = int(line.split()[1])
                    epochs.append(current_epoch)

                    if current_epoch != 0:
                        all_values.append(current_values)
                        current_values = []
                else:
                    value = float(line)
                    current_values.append(value)

        all_values.append(current_values)
        return all_values

    # Process both log files
    all_values_w = process_log_file(log_file_path_w)
    all_values_b = process_log_file(log_file_path_b)

    # Plot for scale_w
    plt.figure(figsize=(12, 8))
    for i in range(len(all_values_w[0])):
        scale_trajectory = [epoch[i] for epoch in all_values_w]
        plt.plot(range(1, len(all_values_w) + 1), scale_trajectory, linestyle='-', marker='o', label=f'scale_w_{i}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Scale Values')
    plt.title(f'scale_w of shape {layer.scale_w.shape} of layer {layer.__class__.__name__}')
    plt.xticks(range(1, len(all_values_w) + 1))
    
    plots_dir = os.path.join(log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'scale_w_of_shape_{layer.scale_w.shape}_of_layer_{layer.__class__.__name__}.png'))
    
    plt.show()

    # Plot for scale_b
    plt.figure(figsize=(12, 8))
    for i in range(len(all_values_b[0])):
        scale_trajectory = [epoch[i] for epoch in all_values_b]
        plt.plot(range(1, len(all_values_b) + 1), scale_trajectory, linestyle='-', marker='o', label=f'scale_b_{i}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Scale Values')
    plt.title(f'scale_b of shape {layer.scale_b.shape} of layer {layer.__class__.__name__}')
    plt.xticks(range(1, len(all_values_b) + 1))
    
    plt.savefig(os.path.join(plots_dir, f'scale_b_of_shape_{layer.scale_b.shape}_of_layer_{layer.__class__.__name__}_with_scale_w_of_shape_{layer.scale_w.shape}.png'))
    
    plt.show()


def validation_losses(log_dir):
    val_loss_file = f"{log_dir}/validation_loss.log"

    val_loss = []

    with open(val_loss_file, 'r') as file:
        for line in file:
            line = line.strip()
            val_loss.append(float(line))

    return val_loss


def get_training_loss(log_dir, type):
    total_loss_file = f"{log_dir}/{type}_loss_log.log"

    train_list = []

    with open(total_loss_file, 'r') as file:
        for line in file:
            if line.startswith("Train batch"):
                parts = line.split(": ")
                loss_value = float(parts[1].strip())
                train_list.append(loss_value)

    return train_list


def average_chunks(lst, n):
    averaged_list = []
    
    # Process each chunk
    for i in range(0, len(lst), n):
        chunk = lst[i:i+n]
        averaged_list.append(sum(chunk) / len(chunk))
    
    return averaged_list

def plot_training_validation_loss(log_dir, interval, epochs, x_train_size, batch_size):

    train_list = get_training_loss(log_dir = log_dir, type = "total")

    skip = 25
    batch_loss = train_list[skip:]

    chunk_size = 25
    chunked_train_list = average_chunks(batch_loss, chunk_size)

    batch_loss_range = range(skip, len(train_list), chunk_size)

    val_loss = validation_losses(log_dir)
    val_loss_range = range(1, len(val_loss) * interval + 1, interval)


    total_batches = len(train_list)
    batches_per_epoch = int(total_batches / epochs)
    epoch_ticks = list(range(int(batches_per_epoch / 2), total_batches + 1, batches_per_epoch))
    epoch_labels = [f'Epoch {i + 1}' for i in range(epochs)]


    # Plot the averages
    plt.figure(figsize=(8, 8))
    plt.plot(batch_loss_range, chunked_train_list, linestyle='-', color='grey', label='Training Loss')
    plt.plot(val_loss_range, val_loss, linestyle='-', color='red', label='Validation Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()  # Add a legend to distinguish between the two lines

    plt.xticks(epoch_ticks, epoch_labels, rotation=90)

    # Add descriptive text
    description_text = "Full validation is conducted at specified batch intervals.\n" \
                    "Batch losses are averaged to smooth out fluctuations in the graph.\n" \
                    "The first few batch losses are not represented in the plot.\n" \
                    "Epoch markers on the x-axis represent the midpoint of each epoch."

    plt.text(0.2, 0.95, description_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment="left", bbox=dict(facecolor='white', alpha=0.5))


    plots_dir = os.path.join(log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'training_and_validation_loss.png'))

    plt.show()


def plot_loss(log_dir, x_train_size, batch_size, penalty_rate):
    total = get_training_loss(log_dir, "total")
    scale = get_training_loss(log_dir, "scale")

    def average_every_n_entries(lst, n):
        return [sum(lst[i:i+n]) / n for i in range(0, len(lst), n)]

    n = int(x_train_size / batch_size)
    total_avg = average_every_n_entries(total, n)
    scale_avg = average_every_n_entries(scale, n)

    loss_dir = os.path.join(log_dir, 'losses')
    os.makedirs(loss_dir, exist_ok=True)
    
    total_avg_log_path = os.path.join(loss_dir, 'total_avg_log.txt')
    scale_avg_log_path = os.path.join(loss_dir, 'scale_avg_log.txt')

    with open(total_avg_log_path, 'w') as total_avg_file:
        for entry in total_avg:
            total_avg_file.write(f"{entry}\n")

    with open(scale_avg_log_path, 'w') as scale_avg_file:
        for entry in scale_avg:
            scale_avg_file.write(f"{entry}\n")


    epochs = range(1, len(total_avg) + 1)

    plt.figure(figsize=(12, 8))

    plt.fill_between(epochs, 0, total_avg, color='grey', alpha=0.5, label='Total Loss')

    plt.plot(epochs, scale_avg, 'r-', label='Custom Added Loss', linewidth=2)
    plt.fill_between(epochs, 0, scale_avg, color='none', edgecolor='red', hatch='///', linewidth=0)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss structure per epoch with penalty rate {penalty_rate}')
    plt.xticks(range(1, len(scale_avg) + 1))
    plt.legend()

    plots_dir = os.path.join(log_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'Loss structure per epoch with penalty rate {penalty_rate}.png'))


    plt.tight_layout()
    plt.show()


import matplotlib.ticker as mtick

def plot_accuracy(log_dir):
    # Paths to the log files
    accuracy_log_path = os.path.join(log_dir, 'accuracy.log')
    vanilla_accuracy_log_path = os.path.join(log_dir, 'vanilla_accuracy.log')
    
    # Read the accuracy values from the log files
    accuracies = []
    vanilla_accuracies = []

    # Check if accuracy.log exists and read the accuracy values
    if os.path.exists(accuracy_log_path):
        with open(accuracy_log_path, 'r') as f:
            for line in f:
                accuracies.append(float(line.strip()))
    else:
        print(f"Warning: {accuracy_log_path} not found.")

    # Check if vanilla_accuracy.log exists and read the accuracy values
    if os.path.exists(vanilla_accuracy_log_path):
        with open(vanilla_accuracy_log_path, 'r') as f:
            for line in f:
                vanilla_accuracies.append(float(line.strip()))
    else:
        print(f"Warning: {vanilla_accuracy_log_path} not found.")

    # Plotting
    plt.figure(figsize=(12, 8))
    
    if accuracies:
        plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-', color='red', label='Quantized Model Accuracy')
    
    if vanilla_accuracies:
        plt.plot(range(1, len(accuracies) + 1), vanilla_accuracies, marker='o', linestyle='-', color='red', label='Vanilla Model Accuracy')
    
    if accuracies or vanilla_accuracies:
        plt.xticks(range(1, len(accuracies) + 1))  # Set x-axis ticks to match the number of epochs
        
        # Convert y-axis to percentage
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))  # Assuming accuracy is between 0 and 1
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Accuracy over Epochs')
        plt.legend()

        # Save the plot
        plot_path = os.path.join(log_dir, 'plots/accuracy_plot.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        
        plt.show()
    else:
        print("No data available to plot.")



def plot_pr_accuracy(log_dirs, penalty_rates):

    def plot(accuracies, penalty_rates):
        plt.figure(figsize=(12, 8))
        penalty_indices = list(range(len(penalty_rates)))  # Use indices for plotting

        for epoch, acc_values in accuracies.items():
            plt.plot(acc_values, penalty_indices, marker='o', label=f'Epoch {epoch}')  # Plot with indices

        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))  # Assuming accuracy is between 0 and 1
        plt.ylabel('Penalty Rate')
        plt.xlabel('Accuracy (%)')
        plt.title('Penalty Rate vs Accuracy for Each Epoch')
        
        # Set custom y-axis tick labels to the actual penalty rate values
        plt.yticks(ticks=penalty_indices, labels=penalty_rates)
        
        plt.legend()
        plt.grid(True)
        plt.show()

    accuracies = {}

    for log_dir in log_dirs:
        accuracy_log_path = os.path.join(log_dir, 'accuracy.log')
        with open(accuracy_log_path, 'r') as f:
            for index, line in enumerate(f):
                accuracies.setdefault(index, []).append(float(line.strip()))

    plot(accuracies, penalty_rates)

def plot_pr_accuracy_2(log_dirs, penalty_rates):

    def plot(accuracies, penalty_rates):
        plt.figure(figsize=(12, 8))
        penalty_indices = list(range(len(penalty_rates)))  # Use indices for plotting

        for epoch, acc_values in accuracies.items():
            if epoch == 9:
                plt.plot(acc_values, penalty_indices, marker='o', label=f'Epoch {epoch}')  # Plot with indices

        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))  # Assuming accuracy is between 0 and 1
        plt.ylabel('Penalty Rate')
        plt.xlabel('Accuracy (%)')
        plt.title('Penalty Rate vs Accuracy for Each Epoch')
        
        # Set custom y-axis tick labels to the actual penalty rate values
        plt.yticks(ticks=penalty_indices, labels=penalty_rates)
        
        plt.legend()
        plt.grid(True)
        plt.show()

    accuracies = {}

    for log_dir in log_dirs:
        accuracy_log_path = os.path.join(log_dir, 'accuracy.log')
        with open(accuracy_log_path, 'r') as f:
            for index, line in enumerate(f):
                accuracies.setdefault(index, []).append(float(line.strip()))

    plot(accuracies, penalty_rates)

def plot_loss_accuracy(log_dirs, penalty_rates):

    def plot(total_losses, scale_losses, penalty_rates):
        plt.figure(figsize=(12,8))

        
        return

    total_losses = {} # keys are epochs, list of values containts
    scale_losses = {} # keys are epochs

    for log_dir in log_dirs:
        total_loss_log_path = os.path.join(log_dir, 'losses/total_avg_log.txt')
        scale_loss_log_path = os.path.join(log_dir, 'losses/scale_avg_log.log')
    
        with open(total_loss_log_path, 'r') as total_f:
            for index, line in enumerate(total_f):
                total_losses.setdefault(index, []).append(float(line.strip()))
    
        with open(scale_loss_log_path, 'r') as scale_f:
                for index, line in enumerate(scale_f):
                    scale_losses.setdefault(index, []).append(float(line.strip()))
    
    plot(total_losses, scale_losses, penalty_rates)


def plot_pareto(log_dirs, penalty_rates, exclude=0):
    
    total_losses = [] # keys are epochs, list of values containts
    scale_losses = [] # keys are epochs

    for log_dir in log_dirs:
        total_loss_log_path = os.path.join(log_dir, 'losses/total_avg_log.txt')
        scale_loss_log_path = os.path.join(log_dir, 'losses/scale_avg_log.txt')

        with open(total_loss_log_path, 'r') as total_f:
            for index, line in enumerate(total_f):
                if index == 6:
                    total_losses.append(float(line.strip()))
    #                total_losses.setdefault(index, []).append(float(line.strip()))

        with open(scale_loss_log_path, 'r') as scale_f:
            for index, line in enumerate(scale_f):
                if index == 9:
                    scale_losses.append(float(line.strip()))
    #                scale_losses.setdefault(index, []).append(float(line.strip()))

    print(total_losses[0])
    print(scale_losses[0])
    scce_loss = [total - scale for total, scale in zip(total_losses, scale_losses)]

    labels = penalty_rates[:len(penalty_rates)-exclude]
    cut_scce = scce_loss[:len(penalty_rates)-exclude]
    cut_total = total_losses[:len(penalty_rates)-exclude]
    cut_scale = scale_losses[:len(penalty_rates)-exclude]

    plt.figure(figsize=(12, 8))
    plt.plot(cut_scce, cut_scale, marker='o', label=f'Test')  
    # Annotating each point with its label
    for i, label in enumerate(labels):
        plt.text(cut_scce[i], cut_scale[i], label, fontsize=12, ha='right')

    plt.ylabel('SCCE loss')
    plt.xlabel('Scale loss')
    plt.title('SCCE loss VS Scale loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_unique_values_accuracy(log_dirs, penalty_rates):

    def plot(unique_vals, accs, prs):
        plt.figure(figsize=(12, 12))
        plt.plot(accs, unique_vals, marker='o', label="test")
        
        for i, pr in enumerate(prs):
            plt.text(accs[i]+0.04, unique_vals[i] + 0.8, f"pr={pr}", fontsize=8, ha='right', va='bottom')
        
        plt.xticks(np.arange(0.0, 1.0, 0.1))
        plt.yticks(np.arange(min(unique_vals), max(unique_vals) + 2, 2))

        plt.xlabel('Accuracy')
        plt.ylabel('Number of unique values')
        plt.title('Accuracy and number of unique values for different penalty rates')
        plt.grid(True)
        plt.show()

    unique_values_per_pr = []
    accuracies = []

    for log_dir, pr in zip(log_dirs, penalty_rates):
        unique_values_los_path = os.path.join(log_dir, 'unique_values.txt')
        unique_values = 0
        with open(unique_values_los_path, 'r') as f:
            for index, line in enumerate(f):
                if line.startswith("Unique values in quantized"):
                    num = int(line.split()[-1])
                    unique_values += num
        unique_values_per_pr.append(unique_values)

        accuracy_log_path = os.path.join(log_dir, 'accuracy.log')
        last_line = None
        with open(accuracy_log_path, 'r') as f:
            for index, line in enumerate(f):
                last_line = line
        accuracies.append(float(last_line.strip()))

    plot(unique_values_per_pr, accuracies, penalty_rates)


import matplotlib.ticker as mtick

def plot_accuracies(log_dirs):
    plt.figure(figsize=(12, 8))

    for log_dir in log_dirs:
        accuracy_log_path = os.path.join(log_dir, 'accuracy.log')
        vanilla_accuracy_log_path = os.path.join(log_dir, 'vanilla_accuracy.log')

        accuracies = []
        vanilla_accuracies = []

        # Check if accuracy.log exists and read the accuracy values
        if os.path.exists(accuracy_log_path):
            with open(accuracy_log_path, 'r') as f:
                for line in f:
                    accuracies.append(float(line.strip()))
        else:
            print(f"Warning: {accuracy_log_path} not found in {log_dir}.")

        # Check if vanilla_accuracy.log exists and read the accuracy values
        if os.path.exists(vanilla_accuracy_log_path):
            with open(vanilla_accuracy_log_path, 'r') as f:
                for line in f:
                    vanilla_accuracies.append(float(line.strip()))
        else:
            print(f"Warning: {vanilla_accuracy_log_path} not found in {log_dir}.")

        # Plotting
        if accuracies:
            plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-', label=f'Quantized Model Accuracy ({os.path.basename(log_dir)})')

        if vanilla_accuracies:
            plt.plot(range(1, len(vanilla_accuracies) + 1), vanilla_accuracies, marker='o', linestyle='--', label=f'Vanilla Model Accuracy ({os.path.basename(log_dir)})')

    if any([os.path.exists(os.path.join(log_dir, 'accuracy.log')) or os.path.exists(os.path.join(log_dir, 'vanilla_accuracy.log')) for log_dir in log_dirs]):
        plt.xticks(range(1, len(accuracies) + 1))  # Set x-axis ticks to match the number of epochs
        
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))  # Assuming accuracy is between 0 and 1
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Accuracy over Epochs')
        plt.legend()

        plt.show()
    else:
        print("No data available to plot.")



def count_unique_quantized_w(file_paths):
    quantized_w_sums = []
    file_names = []

    # Process each file
    for file_path in file_paths:
        quantized_w_sum = 0  # Initialize sum for each file
        file_name = os.path.basename(file_path)
        parent_dir = os.path.basename(os.path.dirname(file_path)) 
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('Unique values in quantized w:'):
                        try:
                            value = int(line.split(':')[-1].strip())
                            quantized_w_sum += value
                        except ValueError:
                            print(f"Warning: Could not parse value in file {file_name} on line: {line.strip()}")
            if quantized_w_sum > 0:
                label = f"{parent_dir}"  # Use the parent directory as part of the label
                print(f"File: {label}, Sum of Unique Quantized W Values: {quantized_w_sum}")  
                quantized_w_sums.append(quantized_w_sum)
                file_names.append(label)
        else:
            print(f"Warning: {file_path} not found.")

    if quantized_w_sums:
        plt.figure(figsize=(20, 6))
        bars = plt.barh(file_names, quantized_w_sums, color='skyblue')
        plt.xlabel('Unique Quantized W Values')
        plt.ylabel('Files')
        plt.title('Unique Quantized W Values for Each Loss Function')
        
        # Add numeric value labels to each bar
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,  
                     f'{bar.get_width():.0f}',  
                     va='center', 
                     ha='left')  # Horizontally align to the left of the bar
        
        plt.tight_layout()
        plt.show()
    else:
        print("No data available to plot.")

        

def count_unique_quantized_b(file_paths):
    quantized_w_sums = []
    file_names = []

    # Process each file
    for file_path in file_paths:
        quantized_w_sum = 0  # Initialize sum for each file
        file_name = os.path.basename(file_path)
        parent_dir = os.path.basename(os.path.dirname(file_path)) 
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('Unique values in quantized b:'):
                        try:
                            value = int(line.split(':')[-1].strip())
                            quantized_w_sum += value
                        except ValueError:
                            print(f"Warning: Could not parse value in file {file_name} on line: {line.strip()}")
            if quantized_w_sum > 0:
                label = f"{parent_dir}"  # Use the parent directory as part of the label
                print(f"File: {label}, Sum of Unique Quantized B Values: {quantized_w_sum}")  
                quantized_w_sums.append(quantized_w_sum)
                file_names.append(label)
        else:
            print(f"Warning: {file_path} not found.")

    if quantized_w_sums:
        plt.figure(figsize=(20, 6))
        bars = plt.barh(file_names, quantized_w_sums, color='skyblue')
        plt.xlabel('Unique Quantized B Values')
        plt.ylabel('Files')
        plt.title('Unique Quantized B Values for Each Loss Function')
        
        # Add numeric value labels to each bar
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,  # Position text at the end of each bar
                     f'{bar.get_width():.0f}',  # Format to remove decimals
                     va='center',  # Vertically center the text with respect to the bar
                     ha='left')  # Horizontally align to the left of the bar

        plt.tight_layout()
        plt.show()
    else:
        print("No data available to plot.")



def plot_pareto_accuracy(accuracy_log_path, quantized_log_path, save_dir=None):
    """
    Plots Pareto optimality from accuracy and unique quantized values logs.
    """
    
    # Step 1: Load accuracy values from accuracy.log
    with open(accuracy_log_path, 'r') as acc_file:
        accuracy = [float(line.strip()) for line in acc_file if line.strip()]  # Ignore empty lines
        print(len(accuracy))

    # Step 2: Load and count unique values from unique_combined_RowWiseQuantized_w_b.log
    unique_values_per_epoch = {}
    with open(quantized_log_path, 'r') as quant_file:
        lines = quant_file.readlines()
        for line in lines:
            if line.startswith("Epoch"):
                epoch = line.split()[1]
                unique_values_per_epoch.setdefault(epoch, [])
            else:
                values = list(map(float, line.split()))
                unique_values_per_epoch[epoch] += values
    

    unique_values_per_epoch = [len(np.unique(values)) for key, values in unique_values_per_epoch.items()]

    if len(accuracy) != len(unique_values_per_epoch):
        raise ValueError("Mismatch between number of epochs in accuracy log and quantized values log.")
    
    
    plt.figure(figsize=(10, 8))
    plt.plot(accuracy, unique_values_per_epoch, 'o-', color='b')
    plt.ylabel('Number of Unique Quantized Values')
    plt.xlabel('Accuracy')
    plt.title('# of Unique Quantized Values vs Accuracy over Epochs')
    plt.grid(True)
    plt.gca().invert_xaxis()  # Invert the y-axis to start from large to small
    plt.legend()

    for i in range(len(accuracy)):
        plt.text(accuracy[i], unique_values_per_epoch[i] + 6, f'{i + 1}', fontsize=9, ha='right')

    if save_dir:
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        # Define the full file path
        file_path = os.path.join(save_dir, 'pareto_optimality_accuracy_plot.png')
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Plot saved to {file_path}")
    
    plt.show()


def plot_pareto_loss(loss_log_path, quantized_log_path, save_dir=None):
    """
    Plots Pareto optimality from loss and unique quantized values logs.
    """
    
    # Step 1: Load loss values from loss.log
    with open(loss_log_path, 'r') as acc_file:
        loss = [float(line.strip()) for line in acc_file if line.strip()]  # Ignore empty lines
        print(len(loss))

    # Step 2: Load and count unique values from unique_combined_RowWiseQuantized_w_b.log
    unique_values_per_epoch = {}
    with open(quantized_log_path, 'r') as quant_file:
        lines = quant_file.readlines()
        for line in lines:
            if line.startswith("Epoch"):
                epoch = line.split()[1]
                unique_values_per_epoch.setdefault(epoch, [])
            else:
                values = list(map(float, line.split()))
                unique_values_per_epoch[epoch] += values
    

    unique_values_per_epoch = [len(np.unique(values)) for key, values in unique_values_per_epoch.items()]

    if len(loss) != len(unique_values_per_epoch):
        raise ValueError("Mismatch between number of epochs in loss log and quantized values log.")
    
    plt.figure(figsize=(10, 8))
    plt.plot(loss, unique_values_per_epoch, 'o-', color='b')
    plt.ylabel('Number of Unique Quantized Values')
    plt.xlabel('Loss')
    plt.title('# of Unique Quantized Values vs Loss over Epochs')
    plt.grid(True)
    #plt.gca().invert_yaxis()  # Invert the y-axis to start from large to small
    plt.legend()

    for i in range(len(loss)):
        plt.text(loss[i], unique_values_per_epoch[i] + 6, f'{i + 1}', fontsize=9, ha='right')

    if save_dir:
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        # Define the full file path
        file_path = os.path.join(save_dir, 'pareto_optimality_loss_plot.png')
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Plot saved to {file_path}")
    
    plt.show()

def plot_accuracy_loss(log_dir):
    """
    Reads accuracy and loss log files from the given directory, plots training and validation accuracy and loss,
    and saves the plots in the same directory.
    
    Parameters:
    log_dir (str): The directory containing accuracy and loss log files.
    """
    
    # File paths
    accuracy_file = os.path.join(log_dir, 'accuracy.log')
    loss_file = os.path.join(log_dir, 'loss.log')
    val_accuracy_file = os.path.join(log_dir, 'val_accuracy.log')
    val_loss_file = os.path.join(log_dir, 'val_loss.log')

    # Load data from log files
    def load_log_file(filepath):
        with open(filepath, 'r') as file:
            data = [float(line.strip()) for line in file if line.strip()]
        return data

    # Load data from the files
    accuracy = load_log_file(accuracy_file)
    loss = load_log_file(loss_file)
    val_accuracy = load_log_file(val_accuracy_file)
    val_loss = load_log_file(val_loss_file)

    # Plot Accuracy
    plt.figure(figsize=(10, 10))
    plt.plot(accuracy, label='Training Accuracy', color='blue', linestyle='-')
    plt.plot(val_accuracy, label='Validation Accuracy', color='red', linestyle='-')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    accuracy_plot_path = os.path.join(log_dir, 'accuracy_plot.png')
    plt.savefig(accuracy_plot_path)
    print(f"Accuracy plot saved to: {accuracy_plot_path}")

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss, label='Training Loss', color='blue', linestyle='-')
    plt.plot(val_loss, label='Validation Loss', color='red', linestyle='-')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(log_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to: {loss_plot_path}")


def plot_pareto_losses(paths, skip_epochs=0, skip_last_epochs=0):
    """
    Plots Pareto optimality from loss and unique quantized values logs of different training scenarios in terms of penalty rate.
    Additionally, it connects corresponding points across paths for each epoch with different colors.
    
    Args:
        paths: List of tuples where each tuple contains:
            - loss_log_path: Path to the loss log file
            - quantized_log_path: Path to the quantized values log file
            - save_dir: Directory where the plots will be saved
        skip_epochs: Number of initial epochs to skip (default is 0, meaning no epochs are skipped)
        skip_last_epochs: Number of last epochs to skip (default is 0, meaning no last epochs are skipped)
    """
    plt.figure(figsize=(20, 20))

    # Colormap to assign a unique color to each line for different scenarios
    colors = plt.cm.get_cmap('tab20', len(paths))
    
    # Colormap to assign different colors for the connecting lines for each epoch
    epoch_colors = plt.cm.get_cmap('tab20', 100)  # 100 epoch hardcoded 

    save_dirs = []
    
    all_losses = []
    all_unique_values = []

    for idx, path in enumerate(paths):
        loss_log_path = path[0]
        quantized_log_path = path[1]
        save_dir = path[2]
        save_dirs.append(save_dir)

        # Get penalty rate from dir name
        pr = [loss_log_path.split("_")[i+1] for i, item in enumerate(loss_log_path.split("_")) if loss_log_path.split("_")[i] == "pr"][0]

        # Step 1: Load loss values from loss.log
        with open(loss_log_path, 'r') as acc_file:
            loss = [float(line.strip()) for line in acc_file if line.strip()]  # Ignore empty lines
            print(f"Loss length: {len(loss)}")

        # Step 2: Load and count unique values from unique_combined_RowWiseQuantized_w_b.log
        unique_values_per_epoch = {}
        with open(quantized_log_path, 'r') as quant_file:
            lines = quant_file.readlines()
            for line in lines:
                if line.startswith("Epoch"):
                    epoch = line.split()[1]
                    unique_values_per_epoch.setdefault(epoch, [])
                else:
                    values = list(map(float, line.split()))
                    unique_values_per_epoch[epoch] += values

        unique_values_per_epoch = [len(np.unique(values)) for key, values in unique_values_per_epoch.items()]

        if len(loss) != len(unique_values_per_epoch):
            raise ValueError("Mismatch between number of epochs in loss log and quantized values log.")

        # Skip the first 'skip_epochs' epochs and the last 'skip_last_epochs' epochs
        if skip_last_epochs > 0:
            loss = loss[skip_epochs:-skip_last_epochs]
            unique_values_per_epoch = unique_values_per_epoch[skip_epochs:-skip_last_epochs]
        else:
            loss = loss[skip_epochs:]
            unique_values_per_epoch = unique_values_per_epoch[skip_epochs:]

        # Store losses and unique values for this path
        all_losses.append(loss)
        all_unique_values.append(unique_values_per_epoch)

        # Plot each line with a unique color using the colormap
        plt.plot(loss, unique_values_per_epoch, 'o-', color=colors(idx), label=f'Scenario with pr {pr}')

    # Now plot lines connecting corresponding entries (same epoch) across different paths with different colors
    num_epochs = len(all_losses[0])  # Assuming all paths have the same number of epochs after skipping
    for epoch_idx in range(num_epochs):
        epoch_losses = [all_losses[path_idx][epoch_idx] for path_idx in range(len(paths))]
        epoch_unique_values = [all_unique_values[path_idx][epoch_idx] for path_idx in range(len(paths))]

        # Plot lines connecting corresponding points for the current epoch across different paths
        # Assign different color for each epoch connection
        plt.plot(epoch_losses, epoch_unique_values, '--', color=epoch_colors(epoch_idx), alpha=0.5)  # Dashed colored line to connect points

    plt.ylabel('Number of Unique Quantized Values')
    plt.xlabel('Loss')
    plt.title('# of Unique Quantized Values vs Loss over Epochs for different penalty rates')

    plt.xscale('log', base=2)  # Logarithmic scale for Loss (x-axis)
    plt.yscale('log', base=2)  # Logarithmic scale for Unique Quantized Values (y-axis)

    plt.grid(True)
    plt.legend(loc='best')

    for save_dir in save_dirs:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'pareto_optimality_losses_plot.png')
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Plot saved to {file_path}")

    plt.show()


def plot_pareto_accuracies(paths, skip_epochs=0, skip_last_epochs=0):
    plt.figure(figsize=(20, 20))

    # Colormap to assign a unique color to each line for different scenarios
    colors = plt.cm.get_cmap('tab20', len(paths))
    
    # Colormap to assign different colors for the connecting lines for each epoch
    epoch_colors = plt.cm.get_cmap('tab20', 100)  # 100 epoch hardcoded 

    save_dirs = []
    
    all_losses = []
    all_unique_values = []

    for idx, path in enumerate(paths):
        loss_log_path = path[0]
        quantized_log_path = path[1]
        save_dir = path[2]
        save_dirs.append(save_dir)

        # Get penalty rate from dir name
        pr = [loss_log_path.split("_")[i+1] for i, item in enumerate(loss_log_path.split("_")) if loss_log_path.split("_")[i] == "pr"][0]

        # Step 1: Load loss values from loss.log
        with open(loss_log_path, 'r') as acc_file:
            loss = [float(line.strip()) for line in acc_file if line.strip()]  # Ignore empty lines
            print(f"Accuracy length: {len(loss)}")

        # Step 2: Load and count unique values from unique_combined_RowWiseQuantized_w_b.log
        unique_values_per_epoch = {}
        with open(quantized_log_path, 'r') as quant_file:
            lines = quant_file.readlines()
            for line in lines:
                if line.startswith("Epoch"):
                    epoch = line.split()[1]
                    unique_values_per_epoch.setdefault(epoch, [])
                else:
                    values = list(map(float, line.split()))
                    unique_values_per_epoch[epoch] += values

        unique_values_per_epoch = [len(np.unique(values)) for key, values in unique_values_per_epoch.items()]

        if len(loss) != len(unique_values_per_epoch):
            raise ValueError("Mismatch between number of epochs in accuracy log and quantized values log.")

        # Skip the first 'skip_epochs' epochs and the last 'skip_last_epochs' epochs
        if skip_last_epochs > 0:
            loss = loss[skip_epochs:-skip_last_epochs]
            unique_values_per_epoch = unique_values_per_epoch[skip_epochs:-skip_last_epochs]
        else:
            loss = loss[skip_epochs:]
            unique_values_per_epoch = unique_values_per_epoch[skip_epochs:]

        # Store losses and unique values for this path
        all_losses.append(loss)
        all_unique_values.append(unique_values_per_epoch)

        # Plot each line with a unique color using the colormap
        plt.plot(loss, unique_values_per_epoch, 'o-', color=colors(idx), label=f'Scenario with pr {pr}')

    # Now plot lines connecting corresponding entries (same epoch) across different paths with different colors
    num_epochs = len(all_losses[0])  # Assuming all paths have the same number of epochs after skipping
    for epoch_idx in range(num_epochs):
        epoch_losses = [all_losses[path_idx][epoch_idx] for path_idx in range(len(paths))]
        epoch_unique_values = [all_unique_values[path_idx][epoch_idx] for path_idx in range(len(paths))]

        # Plot lines connecting corresponding points for the current epoch across different paths
        # Assign different color for each epoch connection
        plt.plot(epoch_losses, epoch_unique_values, '--', color=epoch_colors(epoch_idx), alpha=0.5)  # Dashed colored line to connect points

    plt.ylabel('Number of Unique Quantized Values')
    plt.xlabel('Accuracy')
    plt.title('# of Unique Quantized Values vs Accuracy over Epochs for different penalty rates')

#    plt.xscale('log', base=2)  # Logarithmic scale for Loss (x-axis)
    plt.yscale('log', base=2)  # Logarithmic scale for Unique Quantized Values (y-axis)

    plt.grid(True)
    plt.legend(loc='best')

    for save_dir in save_dirs:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'pareto_optimality_accuracies_plot.png')
        plt.savefig(file_path, bbox_inches='tight')
        print(f"Plot saved to {file_path}")

    plt.show()