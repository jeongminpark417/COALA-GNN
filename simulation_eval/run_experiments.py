import os
import subprocess

# Define the command template with placeholders for varying arguments
command_template = (
    "python Graph_Color_dist_emulate.py --data OGB --path /mnt/nvme16/ --batch_size 1024 --fan_out '10,5,5' "
    "--color_path /mnt/nvme16/ogbn_papers100M/ --dist {dist} --num_parts {num_parts} "
    "--cache_percent {cache_percent} --reset_time {reset_time}"
)

# Define the values for different parameters
num_parts_values = [8]
#dist_values = ["baseline", "graph_color", "cache_meta"]
dist_values = ["graph_color"]
dist_values = ["cache_meta_affinity"]

cache_percent_values = [1.0, 0.5, 0.2, 0.1, 0.05]
#cache_percent_values = [0.1]
reset_time_values = [10]


# Define the output file for storing results
output_file = "experiment_results.txt"

# Open the output file in write mode
with open(output_file, "w") as f:
    # Write header for results
    f.write("num_parts,dist,cache_percent,hit_ratio\n")

    # Iterate over each combination of parameters
    for num_parts in num_parts_values:
        for dist in dist_values:
            for reset_time in reset_time_values:
                for cache_percent in cache_percent_values:
                    # Format the command with the current set of parameters
                    command = command_template.format(
                        num_parts=num_parts,
                        dist=dist,
                        cache_percent=cache_percent,
                        reset_time=reset_time
                    )

                    # Print the command being executed
                    print(f"Running command: {command}")

                    # Run the command and capture the output
                    result = subprocess.run(command, shell=True, capture_output=True, text=True)

                    # Get the output from stdout
                    output = result.stdout

                    # Find the hit ratio in the output
                    for line in output.splitlines():
                        if "Hit ratio:" in line:
                            # Extract the hit ratio value
                            hit_ratio = line.split("Hit ratio:")[1].strip()
                            
                            # Write the configuration and result to the file
                            f.write(f"{num_parts},{dist},{reset_time},{cache_percent},{hit_ratio}\n")
                            print(f"Recorded: num_parts={num_parts}, dist={dist}, reset_time={reset_time}, cache_percent={cache_percent}, hit_ratio={hit_ratio}")

print(f"Results have been saved to {output_file}.")

