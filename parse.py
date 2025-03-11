import os
import re
import csv
import numpy as np

# Sample data dump folder path
data_folder = "./out"

def parse_data(file_path):
    with open(file_path, 'r') as f:
        data = f.read()

    # Extract number of GPUs
    gpu_match = re.search(r'-n (\d+)', data)
    num_gpus = int(gpu_match.group(1)) if gpu_match else None

    # Extract model name
    model_match = re.search(r'configs/(\d+B)/', data)
    model_name = model_match.group(1) if model_match else None

    # Extract uni_dist flags
    use_uni_dist = "use_uni_dist = True" in data
    all_gathers = "uni_dist_low_latency_all_gathers = True" in data
    reduce_scatters = "uni_dist_low_latency_reduce_scatters = True" in data

    # Extract iteration times (excluding first iteration)
    time_matches = re.findall(r'iter \d+: loss .*?, time ([\d.]+)ms', data)
    times = list(map(float, time_matches[1:]))  # Exclude first iteration

    # Compute mean and standard deviation
    mean_time = np.mean(times) if times else None
    std_time = np.std(times) if times else None

    return [file_path, model_name, num_gpus, use_uni_dist, all_gathers, reduce_scatters, mean_time, std_time]

def aggregate_data(folder_path):
    results_by_model = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".out"):
            file_path = os.path.join(folder_path, filename)
            parsed_data = parse_data(file_path)

            model_name = parsed_data[1]
            if model_name:
                if model_name not in results_by_model:
                    results_by_model[model_name] = []
                results_by_model[model_name].append(parsed_data)

    # Write each model's data to a separate CSV file
    for model, results in results_by_model.items():
        output_csv = f"{model}.csv"
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["File", "Model", "GPUs", "use_uni_dist", "all_gathers", "reduce_scatters", "Mean Time (ms)", "Std Time (ms)"])
            writer.writerows(results)

# Run the parser
aggregate_data(data_folder)
