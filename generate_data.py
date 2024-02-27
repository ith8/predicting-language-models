import argparse
import os
from generate_graphs import list_directories
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
    probs_list = {}
    input_text = question + "\n"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    with torch.no_grad():
        if t5:
            decoder_start_token = tokenizer.pad_token_id
            decoder_input_ids = torch.tensor([[decoder_start_token]])
            outputs = model(
                input_ids,
                output_hidden_states=False,
                decoder_input_ids=decoder_input_ids,
            )
            # generated_ids = model.generate(input_ids, max_length=10, decoder_input_ids=decoder_input_ids)
        else:
            outputs = model(input_ids, output_hidden_states=False)
            # generated_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
        # generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # print(f"Generated text: {generated_text}")

        logits = outputs.logits

    last_token_logits = logits[0, -1, :]
    most_likely_token_id = torch.argmax(last_token_logits, dim=-1).item()

    most_likely_token = tokenizer.decode([most_likely_token_id])
    print(f"Most likely token: {most_likely_token}")

    probs = torch.softmax(last_token_logits, dim=0)
    prob = probs[most_likely_token_id].item()
    print(f"Probability of the most likely token: {prob}")
    # print(f"Logits of the most likely token: {last_token_logits[most_likely_token_id]}")

    for answer in ["Yes", "No"]:
        probs_list[answer] = torch.softmax(last_token_logits, dim=0)[
            tokenizer.encode(answer)[0]
        ].item()
        probs_list[answer] += torch.softmax(last_token_logits, dim=0)[
            tokenizer.encode(answer.lower())[0]
        ].item()
    return probs_list


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
    model_name, questions_file, max_questions=200, t5=False, openai=False
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
                    probs = compute_logits(question, model, tokenizer, t5=t5)
                del obj["question"]
                del obj["answer_not_matching_behavior"]
                obj["probs"] = probs
                writer.write(obj)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate data for a model")
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument("--questions_file", type=str, help="questions file")
    parser.add_argument("--max_questions", default=200, type=int, help="max questions")
    parser.add_argument("--t5", type=bool, default=False, help="t5 model")
    parser.add_argument("--all", type=bool, default=False, help="all personas")
    parser.add_argument("--openai", type=bool, default=False, help="openai model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all:
        _, jsonl_files = list_directories("models/")
        for jsonl_file in jsonl_files:
            if "agreeableness" in jsonl_file:
                continue
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
