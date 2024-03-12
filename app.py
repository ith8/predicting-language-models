from flask import Flask, render_template, jsonify, request
import os
import json


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

MODELS_PATH = (
    "models" if not os.environ.get("MODELS_PATH") else os.environ.get("MODELS_PATH")
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/raw-data")
def raw_data():
    return render_template("raw_data.html")


@app.route("/models")
def get_models():
    models, _ = list_directories(MODELS_PATH)

    models = [model for model in models if model not in ["google", "Qwen"]]
    return jsonify(models)


@app.route("/models/<path:model>/files")
def get_files(model):
    files_path = os.path.join(MODELS_PATH, model)
    files = [
        f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))
    ]
    return jsonify(files)


@app.route("/models/<path:model>/files/<filename>")
def get_file_content(model, filename):
    file_path = os.path.join(MODELS_PATH, model, filename)
    if os.path.exists(file_path):
        content = []
        with open(file_path, "r") as file:
            for line in file:
                try:
                    json_line = json.loads(line)

                    json_line["Yes"] = json_line["probs"].get("Yes", None)
                    json_line["No"] = json_line["probs"].get("No", None)

                    del json_line["probs"]
                    if "activations" in json_line:
                        del json_line["activations"]

                    content.append(json_line)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from line in {filename}: {e}")

        return jsonify(content)
    return jsonify({"error": "File not found"}), 404


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


if __name__ == "__main__":
    app.run(debug=True, port=8080)
