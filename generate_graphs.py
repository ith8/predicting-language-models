from collections import defaultdict
import os
import json
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def list_directories(path):
    dirs = []
    jsonl_files = set()
    for root, subdirs, files in os.walk(path):
        for subdir in subdirs:
            if subdir not in ["google", "Qwen"]:
                dirs.append(os.path.relpath(os.path.join(root, subdir), start=path))

        for file in files:
            if file.endswith(".jsonl"):
                jsonl_files.add(file)
    return dirs, jsonl_files


def load_file_contents(file_path):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def compute_correlation(list_dicts1, list_dicts2):
    probs1 = [extract_prob(data) for data in list_dicts1]
    probs2 = [extract_prob(data) for data in list_dicts2]
    correlation, _ = pearsonr(probs1, probs2)
    return correlation


def extract_prob(data):
    probs = data["probs"]
    total_prob = sum(probs.values())
    normalized_probs = {key: val / total_prob for key, val in probs.items()}
    return normalized_probs[data["answer_matching_behavior"].strip()]


def compute_all_pairs_correlations():
    model_dirs, jsonl_files = list_directories("models/")
    persona_correlations = {}

    for persona_data_file_name in jsonl_files:
        model_persona_data = {}
        for model_dir in model_dirs:
            file_path = os.path.join("models/", model_dir, persona_data_file_name)
            model_persona_data[model_dir] = load_file_contents(file_path)

        correlations = {}
        model_names = list(model_persona_data.keys())
        for i, model_name1 in enumerate(model_names):
            for model_name2 in model_names[i + 1 :]:
                list_dicts1 = model_persona_data[model_name1]
                list_dicts2 = model_persona_data[model_name2]
                correlation = compute_correlation(list_dicts1, list_dicts2)
                correlations[
                    (model_name1.split("/")[-1], model_name2.split("/")[-1])
                ] = correlation

        persona_correlations[persona_data_file_name] = correlations

    return persona_correlations


def n_sample_averages(model_persona_data, n):
    num_questions = len(model_persona_data)
    average_prob_list, model_prob_list = [], []
    for i in range(num_questions):
        indices = list(range(num_questions))
        indices.remove(i)

        sampled_indices = np.random.choice(indices, n, replace=False)
        sampled_probs = [
            extract_prob(model_persona_data[idx]) for idx in sampled_indices
        ]
        avg_prob = np.mean(sampled_probs)
        average_prob_list.append(avg_prob)
        model_prob_list.append(extract_prob(model_persona_data[i]))

    return model_prob_list, average_prob_list


def compute_correlations_with_sampled_averages():
    model_dirs, jsonl_files = list_directories("models/")
    persona_correlations = {}

    for persona_data_file_name in jsonl_files:
        correlations = {}
        for model_name in model_dirs:
            file_path = os.path.join("models/", model_name, persona_data_file_name)
            model_persona_data = load_file_contents(file_path)

            for num_samples in [10, 20, 30]:
                model_probs_list, avg_probs_list = n_sample_averages(
                    model_persona_data, num_samples
                )
                correlations[model_name.split("/")[-1], f"{num_samples}_samples"], _ = (
                    pearsonr(model_probs_list, avg_probs_list)
                )

        persona_correlations[persona_data_file_name] = correlations

    return persona_correlations


def generate_all_pairs_heatmaps(all_correlations, save_path):
    for persona_data_file_name, correlations in all_correlations.items():
        persona = persona_data_file_name.replace(".jsonl", "")
        graph_heatmap_and_save(correlations, persona, save_path)


def generate_average_heatmap(all_correlations, save_path):
    aggregated_correlations = defaultdict(list)
    count_files = 0
    for file_correlations in all_correlations.values():
        for model_pair, correlation in file_correlations.items():
            aggregated_correlations[model_pair].append(correlation)
        count_files += 1

    average_correlations = {
        pair: sum(values) / len(values)
        for pair, values in aggregated_correlations.items()
    }

    graph_heatmap_and_save(average_correlations, "average_persona", save_path)


def graph_heatmap_and_save(correlations, persona, save_path):
    graph_averages = "sample" in save_path
    os.makedirs(save_path, exist_ok=True)

    columns = sorted(set([pair[1] for pair in correlations.keys()]))
    index = [
        "flan-t5-small",
        "flan-t5-large",
        "flan-t5-xl",
        "Qwen1.5-1.8B-Chat",
        "Qwen1.5-7B-Chat",
    ]

    corr_matrix = pd.DataFrame(
        index=index, columns=columns if graph_averages else index, data=np.nan
    )

    for (model1, model2), value in correlations.items():
        corr_matrix.at[model1, model2] = value
        if not graph_averages:
            corr_matrix.at[model2, model1] = value

    plt.figure(figsize=(12, 10))
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=cmap,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        linecolor="grey",
        linewidths=0.5,
        cbar=True,
    )
    plt.title(f"{persona} model correlation")

    file_path = os.path.join(save_path, f"{persona}_heatmap.png")
    plt.savefig(file_path)
    plt.close()


generate_all_pairs_heatmaps(
    compute_all_pairs_correlations(), save_path="assets/all_pairs/"
)
generate_average_heatmap(
    compute_all_pairs_correlations(), save_path="assets/all_pairs/"
)

generate_all_pairs_heatmaps(
    compute_correlations_with_sampled_averages(), save_path="assets/n_sample_averages/"
)
generate_average_heatmap(
    compute_correlations_with_sampled_averages(), save_path="assets/n_sample_averages/"
)
