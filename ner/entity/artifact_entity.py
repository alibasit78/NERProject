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
