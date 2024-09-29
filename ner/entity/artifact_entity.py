from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    csv_data_file_path: str
    label_names_file_path: str


@dataclass
class DataTransformationArtifact:
    dataset_dict_path: str
    data_label_names_path: str
    data_label_to_id_path: str
    data_id_to_label_path: str


@dataclass
class ModelTrainingArtifacts:
    model_saved_dir: str
    model_checkpoint_name: str


@dataclass
class ModelEvalArtifact:
    model_eval_artifact_dir: str


@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str
