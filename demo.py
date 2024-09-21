from ner.components.data_ingestion import DataIngestion
from ner.entity.config_entity import DataIngestionConfig

data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig())
data_ingestion.initiate_data_ingestion()
