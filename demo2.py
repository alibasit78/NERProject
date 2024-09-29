from ner.components.data_ingestion import DataIngestion
from ner.components.data_transformation import DataTransformation
from ner.components.model_evaluation import ModelEvaluation
from ner.components.model_trainer import ModelTraining
from ner.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainingArtifacts,
)

# from ner.entity.artifact_entity import DataIngestionArtifact
from ner.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelEvalConfig,
    ModelTrainingConfig,
)

data_ingestion_config = DataIngestionConfig()
data_ingestion_artifact = DataIngestionArtifact(
    csv_data_file_path=data_ingestion_config.csv_data_file_path,
    label_names_file_path=data_ingestion_config.label_names_file_path,
)
# data_ingestion = DataIngestion(data_ingestion_config=DataIngestionConfig())
# data_ingestion_artifact = data_ingestion.initiate_data_ingestion()


data_transformation_config = DataTransformationConfig()
data_transformation_artifact = DataTransformationArtifact(
    dataset_dict_path=data_transformation_config.dataset_dict_path,
    data_label_names_path=data_transformation_config.data_label_names_path,
    data_label_to_id_path=data_transformation_config.data_label_to_id_path,
    data_id_to_label_path=data_transformation_config.data_id_to_label_path,
)
# data_transformation = DataTransformation(
#     data_ingestion_artifact=data_ingestion_artifact,
#     data_transformation_config=DataTransformationConfig(),
# )
# data_transformation_artifact = data_transformation.initiate_data_transformation()

# model_training = ModelTraining(
#     data_transformation_artifact=data_transformation_artifact,
#     model_trainer_config=ModelTrainingConfig(),
# )
# model_training_artifact = model_training.initiate_model_training()
model_training_config = ModelTrainingConfig()
model_training_artifact = ModelTrainingArtifacts(
    model_saved_dir=model_training_config.model_training_artifact_dir,
    model_checkpoint_name=model_training_config.model_checkpoint_name,
)
model_evaluation = ModelEvaluation(
    data_transformation_artifact=data_transformation_artifact,
    model_training_artifact=model_training_artifact,
    model_evaluation_config=ModelEvalConfig(),
)
model_evaluation.initiate_model_evaluation()
