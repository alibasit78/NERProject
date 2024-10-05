import sys

from ner.components.data_ingestion import DataIngestion
from ner.components.data_transformation import DataTransformation
from ner.components.model_evaluation import ModelEvaluation
from ner.components.model_pusher import ModelPusher
from ner.components.model_trainer import ModelTraining
from ner.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelEvalArtifact,
    ModelPusherArtifact,
    ModelTrainingArtifacts,
)
from ner.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelEvalConfig,
    ModelPusherConfig,
    ModelTrainingConfig,
)
from ner.exception import NERException
from ner.logger import logging


class TrainPipeline:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_training_config = ModelTrainingConfig()
        self.model_evaluation_config = ModelEvalConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact

        except Exception as e:
            raise NERException(e, sys) from e

    def start_data_transformation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataTransformationArtifact:
        logging.info("Entered the start_data_transformation method of TrainPipeline class")
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )

            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            return data_transformation_artifact

        except Exception as e:
            raise NERException(e, sys) from e

    def start_model_training(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainingArtifacts:
        logging.info("Entered the start_model_training method of Train pipeline class")
        try:
            model_trainer = ModelTraining(
                model_trainer_config=self.model_training_config,
                data_transformation_artifact=data_transformation_artifact,
            )
            model_trainer_artifact = model_trainer.initiate_model_training()

            logging.info("Performed the Model training operation")
            logging.info("Exited the start_model_training method of Train pipeline class")
            return model_trainer_artifact

        except Exception as e:
            raise NERException(e, sys) from e

    def start_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainingArtifacts,
    ) -> ModelEvalArtifact:
        try:
            logging.info("Entered the start_model_evaluation method of Train pipeline class")
            model_evaluation = ModelEvaluation(
                data_transformation_artifact=data_transformation_artifact,
                model_training_artifact=model_trainer_artifact,
                model_evaluation_config=self.model_evaluation_config,
            )

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            logging.info("Exited the start_model_evaluation method of Train pipeline class")
            return model_evaluation_artifact

        except Exception as e:
            raise NERException(e, sys) from e

    def start_model_pusher(self, model_eval_artifact: ModelEvalArtifact) -> ModelPusherArtifact:
        try:
            logging.info("Started model pushing to Cloud (AWS)")
            model_pusher = ModelPusher(
                model_eval_artifact=model_eval_artifact,
                model_pusher_config=self.model_pusher_config,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Exited the start model pusher method")
            return model_pusher_artifact
        except Exception as e:
            raise NERException(e, sys)

    def run_pipeline(self) -> None:
        try:
            logging.info("Started Model training >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            model_trainer_artifact = self.start_model_training(
                data_transformation_artifact=data_transformation_artifact
            )
            model_eval_artifact = self.start_model_evaluation(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            _ = self.start_model_pusher(model_eval_artifact=model_eval_artifact)

        except Exception as e:
            raise NERException(e, sys) from e
