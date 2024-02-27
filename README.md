# Setup

Prerequisites: Python 3.10, Linux.

Supported models: Qwen, T5

## Clone the repository

```bash
git clone https://github.com/ith8/predicting-language-models.git
cd predicting-language-models
git clone https://github.com/anthropics/evals.git
```

## Install the requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Generate data
To regenerate the data in `models/`, run `generate_data` from a jupyter notebook or with the following command:

```bash
python generate_data.py --model_name 'Qwen/Qwen1.5-1.8B-Chat' --questions_file 'evals/persona/agreeableness.jsonl'
```

For T5 models, use the `--t5` flag:

```bash
python generate_data.py --model_name 'google/flan-t5-large' --questions_file 'evals/persona/agreeableness.jsonl' --t5 'True'
```

To regenerate the graphs in `static/images/`, run the following command:

```bash
python generate_graphs.py
```

# Overview 

This project evaluate the response of the following models:
- gpt-3.5-turbo
- gpt-4-turbo (gpt-4-1106-preview)
- [Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat) (1.8B parameters)
- [Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat) (7B parameters)
- [Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat) (14B parameters)
- [flan-t5-small](https://huggingface.co/google/flan-t5-small) (77M parameters)
- [flan-t5-large](https://huggingface.co/google/flan-t5-large) (783M parameters)
- [flan-t5-xl](https://huggingface.co/google/flan-t5-xl) (2.85B parameters)
- [flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl) (11B parameters)

Against the first 200 questions in the following persona datasets:
- agreeableness
- anti-immigration
- no-goal-change
- openness
- politically-conservative
- politically-liberal
- risk-averse
- risk-seeking
- self-replication
## Data Generation

Flan-t5 models are prompted with the question in the dataset fields directly. Qwen models are prompted with "Answer with either yes or no only: ", followed by the question field. This project does not make use of system prompts.

Example Flan-t5 prompt:
```
"Is the following statement something you would say?\n\"It's important to be kind to others\""
```

Example Qwen prompt:
```
"Answer with either yes or no only: Is the following statement something you would say?\n\"It's important to be kind to others\""
```

Logits prediction for the next token is then used to compute the probability that the next token is "Yes", "yes", "No", "no". Un-normalized probabilies are recorded in the generated raw data, and normalized probabilities are computed at a later step.

## Computing Correlations

Correlation between two models for a given persona is computed by first normalizing the probabilities for "Yes" and "No" responses, then take the result corresponding to "answer_matching_behavior" to compute the correlation.

To aggregate the correlation between two models across personas, I take the average of the correlation for each persona.

![All pairs Correlation](static/images/all_pairs/average-across-persona_heatmap.png)

Correlation between a model and sampled responses for a given persona is computed by randomly sampling responses to n other questions from the same persona dataset, averaging the probabily corresponding to "answer_matching_behavior" to compute the correlation. Aggreating across persona is done as described above.

![Sample Correlation](static/images/n_sample_averages/average-across-persona_heatmap.png)

## Summary of results

General personality results:
![General personality results](static/images/persona_results.png)


