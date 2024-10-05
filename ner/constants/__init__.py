# AWS Credentials
AWS_ACCESS_KEY_ID_ENV_KEY = ""
AWS_SECRET_ACCESS_KEY_ENV_KEY = ""
REGION_NAME = ""

DATA_DIR = "data"
SEED = 42  # seed for reproducing data
FILE_NAME = "ner_datasetreference.csv"
CSV_FILE_NAME = "ner.csv"
LABEL_FILE_NAME = "label_names.pkl"


# Pipeline
PIPELINE_NAME: str = "NER_PIPELINE"
ARTIFACT_DIR: str = "artifact"
# MODEL_FILE_NAME: str = "model.bin"

# Data ingestion constants
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"

# Data transformation constants
DATA_TRANSFORMATION_DIR: str = "data_transformation"
DATASET_DICT_DIR: str = "dataset_dict"
LABEL_NAMES_FILE_NAME: str = "ordered_label_names.pkl"
LABEL_TO_ID_FILE_NAME: str = "label_to_id.pkl"
ID_TO_LABEL_FILE_NAME: str = "id_to_label.pkl"

# Model Training constants
MODEL_TRAINING_ARTIFACTS_DIR: str = "model_training"
TOKENIZER_FILE_NAME: str = ""
PRE_TRAINED_MODEL_CHECKPOINT_NAME: str = "bert-base-cased"
SAVED_MODEL_NAME: str = "model.bin"
ACCUMULATE_STEPS: int = 8
BATCH_SIZE: int = 8
LEARNING_RATE: float = 1e-5
WEIGHT_DECAY: float = 0.01
EPOCHS: int = 1
SAVE_TOTAL_LIMIT: int = 1

# Model Evaluation constants
MODEL_EVAL_DIR: str = "model_evaluation"
MODEL_EVAL_THR_SCORE: float = 0.0

DATA_INGESTED_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTED_TRAIN_VAL_SPLIT_RATIO: float = 0.1

# Model Pusher constants
MODEL_BUCKET_NAME: str = "ner-data-20"
S3_MODEL_SAVED_DIR: str = "saved_model_dir"

# Saved Model Names
MODEL_NAME: str = "model.safetensors"
MODEL_CONFIG_NAME: str = "config.json"
TOKENIZER_NAME: str = "tokenizer.json"
# app configuration
APP_HOST = "0.0.0.0"
APP_PORT = 8080
