
def test(current_epoch, batch, interval):
    batch_line = current_epoch * (1875 + 313) + (int(batch / interval) * 313) + batch + 1
    print(batch_line, batch)


def calculate_average_loss_epoch_2(file_path, start_line, end_line, n):
    """
    Calculate the average loss over a specified range of lines in a log file,
    with a given interval 'n' and skipping 313 lines after each interval.
    """
    loss_values = []
    with open(file_path, 'r') as file:
        i = 0
        while True:
            # Check if the current line is within the start and end range
            if start_line <= i + 1 <= end_line:
                for _ in range(n):
                    if start_line <= i + 1 <= end_line:
                        line = file.readline()
                        if not line:  # End of file reached
                            break
                        i += 1
                        # Extract the loss value from the line
                        if line.startswith("Loss"):
                            _, value = line.split(':')
                            loss_values.append(float(value.strip()))
                    else:
                        break

                # Skip the next 313 lines
                for _ in range(313):
                    if start_line <= i + 1 <= end_line:
                        line = file.readline()
                        if not line:  # End of file reached
                            break
                        i += 1
                    else:
                        break
            else:
                break

    if loss_values:
        print(len(loss_values))
        average_loss = sum(loss_values) / len(loss_values)
        return average_loss
    else:
        return None


if __name__ == "__main__":
    total_loss_file = 'qat/logs/total_loss_log.txt'
    scale_loss_file = 'logs/scale_loss_log.txt'

    print(calculate_average_loss_epoch_2(total_loss_file, 1, 9700, 75))