from ner.components.data_ingestion import DataIngestion
from ner.components.data_transformation import DataTransformation

# from ner.entity.artifact_entity import DataIngestionArtifact
from ner.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
)

data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig())
data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

data_transformation = DataTransformation(
    data_ingestion_artifact=data_ingestion_artifact,
    data_transformation_config=DataTransformationConfig(),
)
data_transformation_artifact = data_transformation.initiate_data_transformation()
