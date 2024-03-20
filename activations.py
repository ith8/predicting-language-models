import os
from app import list_directories
from generate_graphs import (
    TARGET_DIR,
    compute_average_activation_vectors,
    compute_cosine_similarity,
    extract_prob,
    generate_all_pairs_heatmaps,
    load_file_contents,
    predicted_probs_from_activation,
)
from scipy.stats import pearsonr


def prefix_match(hidden_layer_eval, hidden_layer_stored):
    return (
        hidden_layer_eval.split("_")[0] == hidden_layer_stored.split("_")[0]
        and hidden_layer_eval.split("_")[1] == hidden_layer_stored.split("_")[1]
    )


def predicted_probs_from_activation(
    model_persona_data, activation_data, activation_layer_eval, activation_layer_stored
):
    predicted_prob_list, model_prob_list = [], []
    average_activation_vectors = compute_average_activation_vectors(
        activation_data, activation_layer_stored
    )
    for i, query_data in enumerate(model_persona_data):
        predicted_prob_list.append(
            compute_cosine_similarity(
                average_activation_vectors, query_data, activation_layer_eval
            )
        )
        model_prob_list.append(extract_prob(model_persona_data[i]))

    return model_prob_list, predicted_prob_list


comparison_keys = {
    "google": [
        ("encoder_middle_layer", "encoder_middle_layer_without_last_token"),
        ("encoder_middle_layer_last_token", "encoder_middle_layer_next_to_last_token"),
        ("encoder_last_layer", "encoder_last_layer_without_last_token"),
        ("encoder_last_layer_last_token", "encoder_last_layer_next_to_last_token"),
    ],
    "Qwen": [
        ("last_layer", "last_layer"),
        ("middle_layer", "middle_layer"),
        # ("embedding_layer", "embedding_layer_without_last_token"),
        ("last_layer_last_token", "last_layer_next_to_last_token"),
        ("middle_layer_last_token", "middle_layer_next_to_last_token"),
        ("embedding_layer_last_token", "embedding_layer_next_to_last_token"),
    ],
}


def get_initials(name):
    return "".join([x[0] for x in name.split("_")])


def compute_correlations_with_prediction_from_activations():
    activations_dirs, jsonl_files = list_directories("models_activations/")
    persona_correlations = {}

    for persona_data_file_name in jsonl_files:
        correlations = {}
        for model_name in activations_dirs:
            activations_file_path = os.path.join(
                "models_activations/", model_name, persona_data_file_name
            )
            file_path = os.path.join("models/", model_name, persona_data_file_name)
            model_persona_data = load_file_contents(file_path)
            model_activations_data = load_file_contents(activations_file_path)

            for hidden_layer_eval, hidden_layer_stored in comparison_keys[
                model_name.split("/")[0]
            ]:
                model_probs_list, predicted_probs_list = (
                    predicted_probs_from_activation(
                        model_persona_data,
                        model_activations_data,
                        hidden_layer_eval,
                        hidden_layer_stored,
                    )
                )

                activation_name = get_initials(hidden_layer_eval)
                if model_name.startswith("Qwen"):
                    activation_name = "qwen " + activation_name
                    print(predicted_probs_list[:5])

                correlations[model_name.split("/")[-1], activation_name], _ = pearsonr(
                    model_probs_list, predicted_probs_list
                )

        persona_correlations[persona_data_file_name] = correlations
    return persona_correlations


if __name__ == "__main__":
    generate_all_pairs_heatmaps(
        compute_correlations_with_prediction_from_activations(),
        save_path=f"{TARGET_DIR}/activation_correlations/",
    )
