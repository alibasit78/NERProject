import os
from dataclasses import dataclass
from datetime import datetime

from ner.constants import *  # noqa:F403

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    # artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    artifact_dir: str = ARTIFACT_DIR
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_file_path: str = os.path.join(DATA_DIR, FILE_NAME)
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
    )
    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME
    )
    csv_data_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, CSV_FILE_NAME
    )
    label_names_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, LABEL_FILE_NAME
    )


@dataclass
class DataTransformationConfig:
    data_transformation_artifact_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR
    )
    dataset_dict_path: str = os.path.join(data_transformation_artifact_dir, DATASET_DICT_DIR)
    # data_test_file_path: str = os.path.join(data_transformation_artifact_dir, DATA_TEST_FILE_NAME)
    # data_val_file_path: str = os.path.join(data_transformation_artifact_dir, DATA_VAL_FILE_NAME)
    data_label_names_path: str = os.path.join(
        data_transformation_artifact_dir, LABEL_NAMES_FILE_NAME
    )
    data_label_to_id_path: str = os.path.join(
        data_transformation_artifact_dir, LABEL_TO_ID_FILE_NAME
    )
    data_id_to_label_path: str = os.path.join(
        data_transformation_artifact_dir, ID_TO_LABEL_FILE_NAME
    )


@dataclass
class ModelTrainingConfig:
    accumulate_steps: int = ACCUMULATE_STEPS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    save_total_limit: int = SAVE_TOTAL_LIMIT
    epochs: int = EPOCHS
    model_training_artifact_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, MODEL_TRAINING_ARTIFACTS_DIR
    )
    tokenizer_file_path: str = os.path.join(
        model_training_artifact_dir, PRE_TRAINED_MODEL_CHECKPOINT_NAME
    )
    model_file_path: str = os.path.join(model_training_artifact_dir, SAVED_MODEL_NAME)
    model_checkpoint_name: str = PRE_TRAINED_MODEL_CHECKPOINT_NAME


@dataclass
class ModelEvalConfig:
    model_evaluation_artifact_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, MODEL_EVAL_DIR
    )


@dataclass
class ModelPredConfig:
    best_model_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, MODEL_TRAINING_ARTIFACTS_DIR
    )


@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_dir: str = MODEL_SAVED_DIR
