import argparse
from collections import defaultdict
import os
from generate_graphs import list_directories
from sentence_transformers import SentenceTransformer
import jsonlines
import torch
import openai
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

openai.api_key = os.environ.get("OPENAI_API_KEY")


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def load_t5_model_and_tokenizer(model_name):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return model, tokenizer


def compute_logits(question, model, tokenizer, t5=False):
    probs_list, activations = {}, {}
    input_text = question + "\n"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    with torch.no_grad():
        if t5:
            decoder_start_token = tokenizer.pad_token_id
            decoder_input_ids = torch.tensor([[decoder_start_token]])
            outputs = model(
                input_ids,
                output_hidden_states=True,
                decoder_input_ids=decoder_input_ids,
            )
            activations["embedding_layer"] = (
                outputs.encoder_hidden_states[0].squeeze(0).mean(dim=0).tolist()
            )
            activations["embedding_layer_last_token"] = (
                outputs.encoder_hidden_states[0].squeeze(0)[-1].tolist()
            )

            encoder_middle_layer = len(outputs.encoder_hidden_states) // 2
            activations["encoder_middle_layer"] = (
                outputs.encoder_hidden_states[encoder_middle_layer]
                .squeeze(0)
                .mean(dim=0)
                .tolist()
            )
            activations["encoder_middle_layer_last_token"] = (
                outputs.encoder_hidden_states[encoder_middle_layer]
                .squeeze(0)[-1]
                .tolist()
            )

            activations["encoder_last_layer"] = (
                outputs.encoder_last_hidden_state.squeeze(0).mean(dim=0).tolist()
            )
            activations["encoder_last_layer_last_token"] = (
                outputs.encoder_last_hidden_state.squeeze(0)[-1].tolist()
            )

        else:
            outputs = model(input_ids, output_hidden_states=True)
            activations["last_layer"] = (
                outputs.hidden_states[-1].squeeze(0).mean(dim=0).tolist()
            )
            middel_hidden_state = len(outputs.hidden_states) // 2
            activations["middle_layer"] = (
                outputs.hidden_states[middel_hidden_state]
                .squeeze(0)
                .mean(dim=0)
                .tolist()
            )
            activations["embedding_layer"] = (
                outputs.hidden_states[0].squeeze(0).mean(dim=0).tolist()
            )

            activations["last_layer_last_token"] = (
                outputs.hidden_states[-1].squeeze(0)[-1].tolist()
            )
            activations["middle_layer_last_token"] = (
                outputs.hidden_states[middel_hidden_state].squeeze(0)[-1].tolist()
            )
            activations["embedding_layer_last_token"] = (
                outputs.hidden_states[0].squeeze(0)[-1].tolist()
            )

        logits = outputs.logits

    last_token_logits = logits[0, -1, :]
    most_likely_token_id = torch.argmax(last_token_logits, dim=-1).item()

    most_likely_token = tokenizer.decode([most_likely_token_id])
    print(f"Most likely token: {most_likely_token}")

    probs = torch.softmax(last_token_logits, dim=0)
    prob = probs[most_likely_token_id].item()
    print(f"Probability of the most likely token: {prob}")

    for answer in ["Yes", "No"]:
        probs_list[answer] = torch.softmax(last_token_logits, dim=0)[
            tokenizer.encode(answer)[0]
        ].item()
        probs_list[answer] += torch.softmax(last_token_logits, dim=0)[
            tokenizer.encode(answer.lower())[0]
        ].item()
    return probs_list, activations


def compute_logits_openaiAPI(question, model_name):
    probs_list = {}
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": question}],
        max_tokens=1,
        logprobs=True,
        top_logprobs=5,
    )

    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    top_completion = {
        log_prob.token: np.exp(log_prob.logprob) for log_prob in top_logprobs
    }

    for answer in ["Yes", "No"]:
        probs_list[answer] = top_completion.get(answer, 0) + top_completion.get(
            answer.lower(), 0
        )
        if probs_list[answer] == 0:
            probs_list[answer] = min(top_completion.values()) / 2
    if probs_list["Yes"] == probs_list["No"]:
        return None

    print(top_completion)
    return probs_list


def generate_data(
    model_name,
    questions_file,
    max_questions=200,
    t5=False,
    openai=False,
):
    if t5:
        model, tokenizer = load_t5_model_and_tokenizer(model_name)
    elif not openai:
        model, tokenizer = load_model_and_tokenizer(model_name)

    result_file_name = f"models/{model_name}/{os.path.basename(questions_file)}"
    os.makedirs(os.path.dirname(result_file_name), exist_ok=True)
    with jsonlines.open(questions_file) as reader:
        with jsonlines.open(result_file_name, mode="w") as writer:
            for obj in list(reader)[:max_questions]:

                question = obj["question"]
                print(question)
                if not t5:
                    question = "Answer with either yes or no only: " + question
                if openai:
                    probs = compute_logits_openaiAPI(question, model_name)
                else:
                    probs, activations = compute_logits(
                        question, model, tokenizer, t5=t5
                    )
                    obj["activations"] = activations

                del obj["question"]
                del obj["answer_not_matching_behavior"]
                obj["probs"] = probs
                writer.write(obj)


def compute_embeddings(question, model):
    embeddings = model.encode(question)
    return embeddings.tolist()


def compute_embeddings_openAI(question, model_name):
    client = openai.OpenAI()
    response = client.embeddings.create(input=question, model=model_name)
    return response.data[0].embedding


def generate_embedding(
    model_name,
    questions_file,
    max_questions=200,
    openai=False,
):
    model = SentenceTransformer(model_name)
    result_file_name = f"embeddings/{model_name}/{os.path.basename(questions_file)}"
    os.makedirs(os.path.dirname(result_file_name), exist_ok=True)
    with jsonlines.open(questions_file) as reader:
        with jsonlines.open(result_file_name, mode="w") as writer:
            for obj in list(reader)[:max_questions]:
                question = obj["question"]
                print(question)
                if openai:
                    embedding = compute_embeddings_openAI(question, model_name)
                else:
                    embedding = compute_embeddings(question, model)
                obj["embedding"] = embedding
                del obj["question"]
                del obj["answer_not_matching_behavior"]
                writer.write(obj)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate data for a model")
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--questions_file", type=str, help="questions file")
    parser.add_argument("--max_questions", default=200, type=int, help="max questions")
    parser.add_argument("--t5", type=bool, default=False, help="t5 model")
    parser.add_argument("--all", type=bool, default=False, help="all personas")
    parser.add_argument("--openai", type=bool, default=False, help="openai model")
    parser.add_argument(
        "--embedding", type=bool, default=False, help="generate embeddings"
    )
    parser.add_argument(
        "--activations", type=bool, default=False, help="generate activations"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.embedding:
        if args.all:
            _, jsonl_files = list_directories("models/")
            for jsonl_file in jsonl_files:
                generate_embedding(
                    args.model_name,
                    os.path.join("evals/persona", jsonl_file),
                    args.max_questions,
                    args.openai,
                )
        else:
            generate_embedding(
                args.model_name,
                args.questions_file,
                args.max_questions,
                args.openai,
            )
    else:
        if args.all:
            _, jsonl_files = list_directories("models/")
            for jsonl_file in jsonl_files:
                generate_data(
                    args.model_name,
                    os.path.join("evals/persona", jsonl_file),
                    args.max_questions,
                    args.t5,
                    args.openai,
                )
        else:
            generate_data(
                args.model_name,
                args.questions_file,
                args.max_questions,
                args.t5,
                args.openai,
            )
