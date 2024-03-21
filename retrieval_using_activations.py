import os
from generate_graphs import (
    TARGET_DIR,
    generate_all_pairs_heatmaps,
    list_directories,
    load_file_contents,
    extract_prob,
)
from scipy.stats import pearsonr
from sentence_transformers.util import cos_sim
import torch
import jsonlines


def queried_response_probs(model_persona_data, top_k, embeddings_data_file):
    predicted_prob_list, model_prob_list = [], []
    top_cos_sims = []
    for i, line in enumerate(model_persona_data):
        top_k_results = get_top_k_similar_responses(
            i, model_persona_data, embeddings_data_file, top_k
        )
        predicted_prob_list.append(
            weighted_average_prob(top_k_results, model_persona_data)
        )
        model_prob_list.append(extract_prob(line))
        top_cos_sims.append(top_k_results[0][1])

    return model_prob_list, predicted_prob_list, sum(top_cos_sims) / len(top_cos_sims)


def get_top_k_similar_responses(
    index, model_persona_data, embeddings_data_file, top_k, max_similarity=1
):
    cos_sims = torch.tensor(embeddings_data_file[index]["cos_sims"])
    sorted_indices = reversed(cos_sims.argsort())
    result = []
    for i in sorted_indices:
        if (
            i != index
            and model_persona_data[i]["answer_matching_behavior"]
            == embeddings_data_file[i]["answer_matching_behavior"]
            and cos_sims[i] < max_similarity
        ):
            result.append((i, cos_sims[i]))
            if len(result) == top_k:
                break
    return result


def weighted_average_prob(top_k_results, model_persona_data):
    prob = 0
    for i, cos_sim in top_k_results:
        prob += extract_prob(model_persona_data[i]) * cos_sim
    return prob / sum(cos_sim for _, cos_sim in top_k_results)


def compute_cos_sims(embeddings_data):
    embeddings = [
        embedding["activations"]["embedding_layer"] for embedding in embeddings_data
    ]
    for i, embedding in enumerate(embeddings):
        cos_sims = cos_sim(embedding, embeddings)[0]
        embeddings_data[i]["cos_sims"] = cos_sims.tolist()
    return embeddings_data


def add_cos_sims_to_file():
    model_dirs, jsonl_files = list_directories("models/")

    for embeddings_data_file_name in jsonl_files:
        for embeddings_model_name in model_dirs:
            if "gpt" in embeddings_model_name:
                continue
            file_path = os.path.join(
                "models/", embeddings_model_name, embeddings_data_file_name
            )
            persona_embeddings_data = load_file_contents(file_path)
            persona_embeddings_data = compute_cos_sims(persona_embeddings_data)
            with jsonlines.open(file_path, "w") as writer:
                for line in persona_embeddings_data:
                    writer.write(line)


def compute_correlations_with_queried_responses():
    model_dirs, jsonl_files = list_directories("models/")
    persona_correlations = {}

    for persona_data_file_name in jsonl_files:
        correlations = {}
        print(persona_data_file_name)
        for model_name in model_dirs:
            if "gpt" in model_name:
                continue
            file_path = os.path.join("models/", model_name, persona_data_file_name)
            model_persona_data = load_file_contents(file_path)

            for top_k in [1, 3, 5]:
                model_probs_list, avg_probs_list, mean_similarity = (
                    queried_response_probs(
                        model_persona_data, top_k, model_persona_data
                    )
                )

                correlations[model_name.split("/")[-1], top_k], _ = pearsonr(
                    model_probs_list, avg_probs_list
                )
            mean_similarity = round(mean_similarity.item(), 3)
            print(model_name.split("/")[-1], mean_similarity)

        persona_correlations[persona_data_file_name] = correlations

    return persona_correlations


if __name__ == "__main__":
    # add_cos_sims_to_file()

    generate_all_pairs_heatmaps(
        compute_correlations_with_queried_responses(),
        save_path=f"{TARGET_DIR}/correlations_with_retrieved_responses_from_activation/",
    )
