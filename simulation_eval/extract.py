import re
import argparse

def extract_hit_ratios(filename):
    # Regex to match the hit ratios from the file
    hit_ratio_pattern = re.compile(r"System Hit ratio:\s*([\d.]+)\s*GPU hit ratio:\s*([\d.]+)\s*CPU hit ratio:\s*([\d.]+)")

    system_ratios = []
    gpu_ratios = []
    cpu_ratios = []

    with open(filename, 'r') as file:
        for line in file:
            match = hit_ratio_pattern.search(line)
            if match:
                system_ratios.append(float(match.group(1)))
                gpu_ratios.append(float(match.group(2)))
                cpu_ratios.append(float(match.group(3)))

    return system_ratios, gpu_ratios, cpu_ratios

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
    parser = argparse.ArgumentParser(description="Extract and calculate hit ratio averages.")
    parser.add_argument('filename', type=str, help='File containing hit ratio data')

    args = parser.parse_args()
    filename = args.filename

    system_ratios, gpu_ratios, cpu_ratios = extract_hit_ratios(filename)

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
    else:
        print("Not enough data to calculate averages for 32 occurrences.")

if __name__ == "__main__":
    main()

