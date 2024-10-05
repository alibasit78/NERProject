import sys

from ner.cloud_storage.aws_storage import SimpleStorageService
from ner.entity.artifact_entity import (  # ModelTrainingArtifacts,
    ModelEvalArtifact,
    ModelPusherArtifact,
)
from ner.entity.config_entity import ModelPusherConfig

# from ner.entity.s3_estimator import USvisaEstimator
from ner.exception import NERException
from ner.logger import logging


class ModelPusher:
    def __init__(
        self,
        model_eval_artifact: ModelEvalArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = SimpleStorageService()
        self.model_eval_artifact = model_eval_artifact
        self.model_pusher_config = model_pusher_config
        # self.usvisa_estimator = USvisaEstimator(
        #     bucket_name=model_pusher_config.bucket_name,
        #     model_path=model_pusher_config.s3_model_key_path,
        # )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher

        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            logging.info("Uploading artifacts folder to s3 bucket")

            # self.usvisa_estimator.save_model(
            #     from_file=self.model_evaluation_artifact.trained_model_path
            # )
            #  :param remove: By default it is false that means you will have your model locally available in your system folder
            # model_saved_file_path = os.path.join(
            #     self.model_training_artifact.model_saved_dir, "checkpoint-11/config.json"
            # )
            # self.s3.upload_file(
            #     # from_filename=self.model_training_artifact.model_saved_dir,
            #     from_filename=model_saved_file_path,
            #     to_filename=self.model_pusher_config.s3_model_key_path,
            #     bucket_name=self.model_pusher_config.bucket_name,
            #     remove=False,
            # )
            self.s3.upload_directory_to_s3(
                directory_path=self.model_eval_artifact.trained_model_path,
                bucket_name=self.model_pusher_config.bucket_name,
                s3_folder=self.model_pusher_config.s3_model_dir,
            )
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_dir,
            )

            logging.info("Uploaded model artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")

            return model_pusher_artifact
        except Exception as e:
            raise NERException(e, sys) from e
