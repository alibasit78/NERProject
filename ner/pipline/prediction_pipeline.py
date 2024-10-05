import os
import sys

from transformers import (
    AutoModel,
    AutoTokenizer,
    pipeline,
)

from ner.cloud_storage.aws_storage import SimpleStorageService
from ner.constants import (
    MODEL_CONFIG_NAME,
    MODEL_NAME,
    TOKENIZER_NAME,
)
from ner.entity.config_entity import ModelPredConfig
from ner.exception import NERException
from ner.logger import logging


class ModelPredictor:
    def __init__(self) -> None:
        self.model_pred_config = ModelPredConfig()
        self.s3 = SimpleStorageService()

    def get_model(self, is_model_in_local=False):
        if is_model_in_local:
            checkpoint_dir = os.listdir(self.model_pred_config.best_model_dir)[0]
            model_path = os.path.join(self.model_pred_config.best_model_dir, checkpoint_dir)
            tokenizer_path = os.path.join(
                self.model_pred_config.best_model_dir,
                checkpoint_dir,
            )
            # config_path = os.path.join(
            #     self.model_pred_config.best_model_dir, checkpoint_dir, MODEL_CONFIG_NAME
            # )
        else:
            s3_url_response = self.s3.get_model_urls(
                bucket_name=self.model_pred_config.bucket_name,
                s3_model_dir=self.model_pred_config.s3_model_dir,
            )
            model_path, tokenizer_path, config_path = (
                s3_url_response.model_url,
                s3_url_response.tokenizer_url,
                s3_url_response.config_url,
            )
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer

    def initiate_model_predictor(self, input_sentence: str):
        try:
            logging.info("Started model prediction")
            os.makedirs(self.model_pred_config.best_model_dir, exist_ok=True)
            logging.info(f"best model dir: {self.model_pred_config.best_model_dir}")
            checkpoint_dir = os.listdir(self.model_pred_config.best_model_dir)[0]
            # model, tokenizer = self.get_model(is_model_in_local=True)
            token_classifier = pipeline(
                "token-classification",
                model=os.path.join(self.model_pred_config.best_model_dir, checkpoint_dir),
                # tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
            # Example sentence: "My name is Sylvain and I work at Hugging Face in Brooklyn."
            output = None
            if input_sentence.strip() == "":
                print("Empty input passed")
            else:
                # print(token_classifier)
                output = token_classifier(input_sentence)
            print(f"Result: {output}")
            return input_sentence, output
        except Exception as e:
            raise NERException(e, sys) from e
