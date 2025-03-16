import re
import argparse

def extract_hit_ratios_and_times(filename):
    # Regex to match the hit ratios from the file
    hit_ratio_pattern = re.compile(r"System Hit ratio:\s*([\d.]+)\s*GPU hit ratio:\s*([\d.]+)\s*CPU hit ratio:\s*([\d.]+)")
    
    # Regex to match the time-related values
    request_time_pattern = re.compile(r"Request Time:\s*([\d.]+)")
    request_kernel_time_pattern = re.compile(r"Request Kernel Time:\s*([\d.]+)")
    kernel_time_pattern = re.compile(r"Kernel Time:\s*([\d.]+)")
    epoch_time_pattern = re.compile(r"epoch time:\s*([\d.]+)")

    system_ratios = []
    gpu_ratios = []
    cpu_ratios = []
    request_times = []
    request_kernel_times = []
    kernel_times = []
    epoch_times = []

    with open(filename, 'r') as file:
        for line in file:
            # Extract hit ratios
            match_hit = hit_ratio_pattern.search(line)
            if match_hit:
                system_ratios.append(float(match_hit.group(1)))
                gpu_ratios.append(float(match_hit.group(2)))
                cpu_ratios.append(float(match_hit.group(3)))
            
            # Extract time-related values
            match_request_time = request_time_pattern.search(line)
            if match_request_time:
                request_times.append(float(match_request_time.group(1)))

            match_request_kernel_time = request_kernel_time_pattern.search(line)
            if match_request_kernel_time:
                request_kernel_times.append(float(match_request_kernel_time.group(1)))

            match_kernel_time = kernel_time_pattern.search(line)
            if match_kernel_time:
                kernel_times.append(float(match_kernel_time.group(1)))

            match_epoch_time = epoch_time_pattern.search(line)
            if match_epoch_time:
                epoch_times.append(float(match_epoch_time.group(1)))

    return system_ratios, gpu_ratios, cpu_ratios, request_times, request_kernel_times, kernel_times, epoch_times

def calculate_average_for_batches(ratios, batch_size):
    # Split ratios into batches of batch_size and calculate the average for each batch
    averages = []
    for i in range(0, len(ratios), batch_size):
        batch = ratios[i:i + batch_size]
        if batch:  # Ensure there's something in the batch
            avg = sum(batch) / len(batch)
            averages.append(avg)
    return averages

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Extract and calculate hit ratio and time averages.")
    parser.add_argument('filename', type=str, help='File containing hit ratio and time data')

    args = parser.parse_args()
    filename = args.filename

    (system_ratios, gpu_ratios, cpu_ratios, request_times, request_kernel_times, 
     kernel_times, epoch_times) = extract_hit_ratios_and_times(filename)

    # Calculate averages for the first 16 and the next 16 hit ratios
    batch_size = 16

    if len(system_ratios) >= 32:
        print("Averages for the first 16 and next 16 occurrences:")

        # System hit ratios
        system_avg_batches = calculate_average_for_batches(system_ratios, batch_size)
        print(f"System Hit Ratio: First 16: {system_avg_batches[0]}, Next 16: {system_avg_batches[1]}")

        # GPU hit ratios
        gpu_avg_batches = calculate_average_for_batches(gpu_ratios, batch_size)
        print(f"GPU Hit Ratio: First 16: {gpu_avg_batches[0]}, Next 16: {gpu_avg_batches[1]}")

        # CPU hit ratios
        cpu_avg_batches = calculate_average_for_batches(cpu_ratios, batch_size)
        print(f"CPU Hit Ratio: First 16: {cpu_avg_batches[0]}, Next 16: {cpu_avg_batches[1]}")

        # Request times
        request_avg_batches = calculate_average_for_batches(request_times, batch_size)
        print(f"Request Time: First 16: {request_avg_batches[0]}, Next 16: {request_avg_batches[1]}")

        # Request kernel times
        request_kernel_avg_batches = calculate_average_for_batches(request_kernel_times, batch_size)
        print(f"Request Kernel Time: First 16: {request_kernel_avg_batches[0]}, Next 16: {request_kernel_avg_batches[1]}")

        # Kernel times
        kernel_avg_batches = calculate_average_for_batches(kernel_times, batch_size)
        print(f"Kernel Time: First 16: {kernel_avg_batches[0]}, Next 16: {kernel_avg_batches[1]}")

        # Epoch times
        epoch_avg_batches = calculate_average_for_batches(epoch_times, batch_size)
        print(f"Epoch Time: First 16: {epoch_avg_batches[0]}, Next 16: {epoch_avg_batches[1]}")
    else:
        print("Not enough data to calculate averages for 32 occurrences.")

if __name__ == "__main__":
    main()

