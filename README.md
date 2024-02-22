# Setup

Prerequisites: Python 3.10, Mac or Linux.

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