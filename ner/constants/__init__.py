# AWS Credentials
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

DATA_DIR = "data"
SEED = 42  # seed for reproducing data
FILE_NAME = "ner_datasetreference.csv"
CSV_FILE_NAME = "ner.csv"
LABEL_FILE_NAME = "label_names.pkl"


# Pipeline
PIPELINE_NAME: str = "NER_PIPELINE"
ARTIFACT_DIR: str = "artifact"
MODEL_FILE_NAME: str = "model.bin"

# Data ingestion constants
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"

# Data transformation constants
DATA_TRANSFORMATION_DIR: str = "data_transformation"
DATASET_DICT_DIR: str = "dataset_dict"
LABEL_NAMES_FILE_NAME: str = "ordered_label_names.pkl"
LABEL_TO_ID_FILE_NAME: str = "label_to_id.pkl"
ID_TO_LABEL_FILE_NAME: str = "id_to_label.pkl"


DATA_INGESTED_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTED_TRAIN_VAL_SPLIT_RATIO: float = 0.1
