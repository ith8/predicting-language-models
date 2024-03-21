import os
from generate_graphs import (
    TARGET_DIR,
    generate_all_pairs_heatmaps,
    list_directories,
    load_file_contents,
    extract_prob,
)
from retrieval import get_top_k_similar_responses, weighted_average_prob
from scipy.stats import pearsonr
from sentence_transformers.util import cos_sim
import torch
import jsonlines


def queried_response_probs_lookup(
    model_persona_data, lookup_persona_data, top_k, embeddings_data_file
):
    predicted_prob_list, model_prob_list = [], []
    top_cos_sims = []
    for i, line in enumerate(model_persona_data):
        top_k_results = get_top_k_similar_responses(
            i, model_persona_data, embeddings_data_file, top_k
        )
        predicted_prob_list.append(
            weighted_average_prob(top_k_results, lookup_persona_data)
        )
        model_prob_list.append(extract_prob(line))
        top_cos_sims.append(top_k_results[0][1])

    return model_prob_list, predicted_prob_list, sum(top_cos_sims) / len(top_cos_sims)


def compute_cos_sim_lookup(embeddings_data, lookup_embeddings_data):
    embeddings = [embedding["embedding"] for embedding in embeddings_data]
    lookup_embeddings = [embedding["embedding"] for embedding in lookup_embeddings_data]
    for i, embedding in enumerate(embeddings):
        cos_sims = cos_sim(embedding, lookup_embeddings)[0]
        embeddings_data[i]["cos_sims"] = cos_sims.tolist()
    return embeddings_data


def add_cos_sim_lookup_to_file():
    model_dirs, jsonl_files = list_directories("embeddings_hitler/")
    for embeddings_data_file_name in jsonl_files:
        for embeddings_model_name in model_dirs:
            file_path = os.path.join(
                "embeddings_hitler/", embeddings_model_name, embeddings_data_file_name
            )
            lookup_file_path = os.path.join(
                "embeddings/", embeddings_model_name, embeddings_data_file_name
            )
            persona_embeddings_data = load_file_contents(file_path)
            lookup_embeddings_data = load_file_contents(lookup_file_path)
            persona_embeddings_data = compute_cos_sim_lookup(
                persona_embeddings_data, lookup_embeddings_data
            )
            with jsonlines.open(file_path, "w") as writer:
                for line in persona_embeddings_data:
                    writer.write(line)


def compute_correlations_with_queried_responses_lookup(embeddings_model):
    model_dirs, jsonl_files = list_directories("models_hitler/")
    persona_correlations = {}

    for persona_data_file_name in jsonl_files:
        correlations = {}
        embeddings_data_file = load_file_contents(
            os.path.join("embeddings_hitler/", embeddings_model, persona_data_file_name)
        )
        for model_name in model_dirs:
            file_path = os.path.join(
                "models_hitler/", model_name, persona_data_file_name
            )
            lookup_file_path = os.path.join(
                "models/", model_name, persona_data_file_name
            )
            model_persona_data = load_file_contents(file_path)
            lookup_persona_data = load_file_contents(lookup_file_path)

            for top_k in [1, 3, 5]:
                model_probs_list, avg_probs_list, mean_similarity = (
                    queried_response_probs_lookup(
                        model_persona_data,
                        lookup_persona_data,
                        top_k,
                        embeddings_data_file,
                    )
                )
                correlations[model_name.split("/")[-1], top_k], _ = pearsonr(
                    model_probs_list, avg_probs_list
                )
        mean_similarity = round(mean_similarity.item(), 3)

        persona_correlations[
            f"{persona_data_file_name}, mean similarity {mean_similarity}"
        ] = correlations

    return persona_correlations


if __name__ == "__main__":
    # embeddings_data_file = load_file_contents(file)
    # embeddings = [line["embedding"] for line in embeddings_data_file]
    # cos_sims = torch.tensor(embeddings_data_file[1]["cos_sims"])
    # top_5_indices = cos_sims.argsort()[-5:]
    # print(embeddings_data_file[1]["statement"])
    # top_5_statement = [(embeddings_data_file[i]["statement"], cos_sims[i]) for i in top_5_indices]
    # print(top_5_statement)
    embedding_model_names, _ = list_directories("embeddings_hitler/")

    for embeddings_model in embedding_model_names:

        generate_all_pairs_heatmaps(
            compute_correlations_with_queried_responses_lookup(embeddings_model),
            save_path=f"{TARGET_DIR}/hitler/correlations_with_retrieved_responses/{embeddings_model}",
        )