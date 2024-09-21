from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    csv_data_file_path: str
