# Setup

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
To regenerate the data in models/, run `generate_data` from a jupyter notebook or with the following command:

```bash
python generate_data.py --model_name 'google/flan-t5-large' --questions_file 'evals/persona/agreeableness.jsonl' --t5 'True'
```
