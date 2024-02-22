from flask import Flask, render_template, jsonify, request
import os
import json

from generate_graphs import list_directories

app = Flask(
    __name__,
    template_folder="public_html/predicting-language-models/templates",
    static_folder="public_html/predicting-language-models/static",
)


@app.route("/predicting-language-models")
def home():
    return render_template("index.html")


@app.route("/predicting-language-models/raw-data")
def raw_data():
    return render_template("raw_data.html")


@app.route("/models")
def get_models():
    models_path = "models"
    models, _ = list_directories(models_path)

    models = [model for model in models if model not in ["google", "Qwen"]]
    return jsonify(models)


@app.route("/models/<path:model>/files")
def get_files(model):
    files_path = os.path.join("models", model)
    files = [
        f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))
    ]
    return jsonify(files)


@app.route("/models/<path:model>/files/<filename>")
def get_file_content(model, filename):
    file_path = os.path.join("models", model, filename)
    if os.path.exists(file_path):
        content = []
        with open(file_path, "r") as file:
            for line in file:
                try:
                    json_line = json.loads(line)

                    json_line["Yes"] = json_line["probs"].get("Yes", None)
                    json_line["No"] = json_line["probs"].get("No", None)

                    del json_line["probs"]

                    content.append(json_line)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from line in {filename}: {e}")

        return jsonify(content)
    return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    app.run(debug=True, port=8080)
