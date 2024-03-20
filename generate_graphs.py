from collections import defaultdict
from scipy.stats import pearsonr, t
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers.util import cos_sim

MODEL_ORDER = [
    "gpt-3.5-turbo",
    "gpt-4-1106-preview",
    "flan-t5-small",
    "flan-t5-large",
    "flan-t5-xl",
    "flan-t5-xxl",
    "Qwen1.5-1.8B-Chat",
    "Qwen1.5-7B-Chat",
    "Qwen1.5-14B-Chat",
]
TARGET_DIR = "static/images"

plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 12


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


def abbreviate_name(name):
    return " ".join([x for x in name.split("_") if x != "layer"]).replace(
        "last token", "lt"
    )


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


def split_by_answer_matching_behavior(data):
    yes_matching_queries = [
        d for d in data if d["answer_matching_behavior"].strip() == "Yes"
    ]
    no_matching_queries = [
        d for d in data if d["answer_matching_behavior"].strip() == "No"
    ]
    return yes_matching_queries, no_matching_queries


def extract_activations(data, activation_layer):
    return torch.tensor(data.get("activations")[activation_layer])


def sample_response_probs(model_persona_data, num_samples):
    num_questions = len(model_persona_data)
    predicted_prob_list, model_prob_list = [], []
    np.random.shuffle(model_persona_data)
    for i in range(0, num_questions - num_samples, num_samples):
        sampled_indices = list(range(i + 1, i + num_samples))
        avg_prob = np.mean(
            [extract_prob(model_persona_data[idx]) for idx in sampled_indices]
        )
        predicted_prob_list.append(avg_prob)
        model_prob_list.append(extract_prob(model_persona_data[i]))
        model_persona_data[i]["predicted_avg"] = avg_prob

    return model_prob_list, predicted_prob_list


def compute_average_activation_vectors(data, activation_layer):
    activations_vectors = {}
    yes_matching_queries, no_matching_queries = split_by_answer_matching_behavior(data)
    activations_vectors["Yes"] = np.mean(
        [extract_activations(d, activation_layer) for d in yes_matching_queries], axis=0
    )
    activations_vectors["No"] = np.mean(
        [extract_activations(d, activation_layer) for d in no_matching_queries], axis=0
    )
    return activations_vectors


def compute_cosine_similarity(average_activation_vectors, query_data, activation_layer):
    activation_vector = extract_activations(query_data, activation_layer)
    similarity = cos_sim(
        activation_vector,
        torch.tensor(
            average_activation_vectors[query_data["answer_matching_behavior"].strip()]
        ),
    )
    return similarity.item()


def predicted_probs_from_activation(model_persona_data, activation_layer):
    predicted_prob_list, model_prob_list = [], []
    eval_set, prediction_set = model_persona_data[:40], model_persona_data[40:]
    average_activation_vectors = compute_average_activation_vectors(
        prediction_set, activation_layer
    )
    for i, query_data in enumerate(eval_set):
        predicted_prob_list.append(
            compute_cosine_similarity(
                average_activation_vectors, query_data, activation_layer
            )
        )
        model_prob_list.append(extract_prob(model_persona_data[i]))

    return model_prob_list, predicted_prob_list


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


def compute_correlations_with_sampled_responses():
    model_dirs, jsonl_files = list_directories("models/")
    persona_correlations = {}

    for persona_data_file_name in jsonl_files:
        correlations = {}
        for model_name in model_dirs:
            file_path = os.path.join("models/", model_name, persona_data_file_name)
            model_persona_data = load_file_contents(file_path)

            for num_samples in [3, 5, 7]:
                model_probs_list, avg_probs_list = sample_response_probs(
                    model_persona_data, num_samples
                )
                correlations[model_name.split("/")[-1], num_samples], _ = pearsonr(
                    model_probs_list, avg_probs_list
                )

        persona_correlations[persona_data_file_name] = correlations

    return persona_correlations


def compute_correlations_with_prediction_from_activations():
    model_dirs, jsonl_files = list_directories("models/")
    persona_correlations = {}

    for persona_data_file_name in jsonl_files:
        correlations = {}
        for model_name in model_dirs:
            if not (model_name.startswith("google") or model_name.startswith("Qwen")):
                continue
            file_path = os.path.join("models/", model_name, persona_data_file_name)
            model_persona_data = load_file_contents(file_path)
            for hidden_layer in model_persona_data[0]["activations"].keys():
                if hidden_layer == "embedding_layer_last_token":
                    continue
                model_probs_list, predicted_probs_list = (
                    predicted_probs_from_activation(model_persona_data, hidden_layer)
                )
                activation_name = abbreviate_name(hidden_layer)
                if model_name.startswith("Qwen"):
                    activation_name = "q " + activation_name
                correlations[model_name.split("/")[-1], activation_name], _ = pearsonr(
                    model_probs_list, predicted_probs_list
                )

        persona_correlations[persona_data_file_name] = correlations

    return persona_correlations


def generate_all_pairs_heatmaps(all_correlations, save_path):
    for persona_data_file_name, correlations in all_correlations.items():
        persona = persona_data_file_name.replace(".jsonl", "")
        plot_heatmap(correlations, persona, save_path)
    generate_aggregate_heatmap(all_correlations, save_path)


def generate_aggregate_heatmap(all_correlations, save_path):
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

    plot_heatmap(average_correlations, "average-across-persona", save_path)


def plot_heatmap(correlations, persona, save_path):
    graph_averages = "all_pairs" not in save_path
    os.makedirs(save_path, exist_ok=True)

    columns = sorted(set([pair[1] for pair in correlations.keys()]))
    index = MODEL_ORDER
    if "activation" in save_path:
        index = [i for i in index if "gpt" not in i]

    corr_matrix = pd.DataFrame(
        index=index, columns=columns if graph_averages else index, data=np.nan
    )

    for (model1, model2), value in correlations.items():
        corr_matrix.at[model1, model2] = value
        if not graph_averages:
            corr_matrix.at[model2, model1] = value

    plt.figure(figsize=(16, 14))
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
    plt.xticks(rotation=35)
    plt.yticks(rotation=35)
    plt.title(persona, fontsize=25, fontweight="bold", pad=30)

    file_path = os.path.join(save_path, f"{persona.split(',')[0]}_heatmap.png")
    plt.savefig(file_path)
    plt.close()


def plot_persona_results(base_path="models/"):
    model_dirs, jsonl_files = list_directories(base_path)
    persona_averages = []

    for model_dir in model_dirs:
        for jsonl_file in jsonl_files:
            file_path = os.path.join(base_path, model_dir, jsonl_file)
            if os.path.exists(file_path):
                questions_data = load_file_contents(file_path)
                prob_list = [extract_prob(data) for data in questions_data]
                avg_prob = np.mean(prob_list)
                std_error = np.std(prob_list) / np.sqrt(len(prob_list))
                margin_of_error = std_error * t.ppf(0.975, len(prob_list) - 1)

                persona_averages.append(
                    {
                        "Model": model_dir.split("/")[-1],
                        "Persona": os.path.splitext(jsonl_file)[0],
                        "Average Probability": avg_prob,
                        "Margin of Error": margin_of_error,
                    }
                )

    df = pd.DataFrame(persona_averages).sort_values(by="Persona")

    palette = {
        "flan-t5-large": "blue",
        "flan-t5-small": "skyblue",
        "flan-t5-xl": "blueviolet",
        "flan-t5-xxl": "darkviolet",
        "Qwen1.5-1.8B-Chat": "lightgreen",
        "Qwen1.5-7B-Chat": "green",
        "Qwen1.5-14B-Chat": "darkgreen",
        "gpt-3.5-turbo": "orange",
        "gpt-4-1106-preview": "red",
    }

    plt.figure(figsize=(12, 10))
    ax = sns.scatterplot(
        data=df,
        x="Average Probability",
        y="Persona",
        hue="Model",
        hue_order=MODEL_ORDER,
        palette=palette,
        s=200,
        alpha=0.6,
    )
    for _, row in df.iterrows():
        ax.errorbar(
            x=row["Average Probability"],
            y=row["Persona"],
            xerr=row["Margin of Error"],
            fmt="none",
            color="black",
            capsize=2,
            alpha=0.5,
        )
    plt.axvline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xlim(0, 1)
    plt.xlabel("% Answers Matching Behavior")
    plt.ylabel("Behavior")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Model")
    plt.tight_layout()
    plt.tight_layout()

    plt.savefig(f"{TARGET_DIR}/persona_results.png")
    plt.close()


if __name__ == "__main__":
    # plot_persona_results()

    # generate_all_pairs_heatmaps(
    #     compute_all_pairs_correlations(), save_path=f"{TARGET_DIR}/all_pairs/"
    # )

    generate_all_pairs_heatmaps(
        compute_correlations_with_prediction_from_activations(),
        save_path=f"{TARGET_DIR}/activation_correlations/",
    )

    # generate_all_pairs_heatmaps(
    #     compute_correlations_with_sampled_responses(),
    #     save_path=f"{TARGET_DIR}/n_sample_averages/",
    # )
