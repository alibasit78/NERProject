# AWS Credentials
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

DATA_DIR = "data"

FILE_NAME = "ner_datasetreference.csv"
CSV_FILE_NAME = "ner.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
VAL_FILE_NAME = "val.csv"
# Data ingestion config


# Pipeline
PIPELINE_NAME: str = "NER_PIPELINE"
ARTIFACT_DIR: str = "artifact"
MODEL_FILE_NAME: str = "model.bin"

# Data ingestion constants
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"

DATA_INGESTED_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTED_TRAIN_VAL_SPLIT_RATIO: float = 0.1
