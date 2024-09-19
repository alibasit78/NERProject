import os
from pathlib import Path

PROJECT_NAME = "ner"

list_of_files = [
    f"{PROJECT_NAME}/__init__.py",
    f"{PROJECT_NAME}/components/__init__.py",
    f"{PROJECT_NAME}/components/data_ingestion.py",
    f"{PROJECT_NAME}/components/data_validation.py",
    f"{PROJECT_NAME}/components/data_transformation.py",
    f"{PROJECT_NAME}/components/model_trainer.py",
    f"{PROJECT_NAME}/components/model_evaluation.py",
    f"{PROJECT_NAME}/components/model_pusher.py",
    f"{PROJECT_NAME}/configuration/__init__.py",
    f"{PROJECT_NAME}/constants/__init__.py",
    f"{PROJECT_NAME}/entity/__init__.py",
    f"{PROJECT_NAME}/entity/config_entity.py",
    f"{PROJECT_NAME}/entity/artifact_entity.py",
    f"{PROJECT_NAME}/exception/__init__.py",
    f"{PROJECT_NAME}/logger/__init__.py",
    f"{PROJECT_NAME}/pipline/__init__.py",
    f"{PROJECT_NAME}/pipline/training_pipeline.py",
    f"{PROJECT_NAME}/pipline/prediction_pipeline.py",
    f"{PROJECT_NAME}/utils/__init__.py",
    f"{PROJECT_NAME}/utils/main_utils.py",
    "notebook/README.MD",
    "app.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "demo.py",
    "setup.py",
    "config/model.yaml",
    "config/schema.yaml",
]

if __name__ == "__main__":
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)
        if filedir != "":
            print(filedir)
            os.makedirs(filedir, exist_ok=True)
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w", encoding="utf-8") as f:
                pass
        else:
            print(f"file is already present at: {filepath}")
