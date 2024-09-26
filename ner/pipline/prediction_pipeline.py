import os
import sys

from transformers import pipeline

from ner.entity.config_entity import ModelPredConfig
from ner.exception import NERException
from ner.logger import logging


class ModelPredictor:
    def __init__(self) -> None:
        self.model_pred_config = ModelPredConfig()

    def initiate_model_predicctor(self, input_sentence: str):
        try:
            logging.info("Started model prediction")
            os.makedirs(self.model_pred_config.best_model_dir, exist_ok=True)
            checkpoint_dir = os.listdir(self.model_pred_config.best_model_dir)[0]
            token_classifier = pipeline(
                "token-classification",
                model=os.path.join(self.model_pred_config.best_model_dir, checkpoint_dir),
                aggregation_strategy="simple",
            )
            # Example sentence: "My name is Sylvain and I work at Hugging Face in Brooklyn."
            output = None
            if input_sentence.strip() == "":
                print("Empty input passed")
            else:
                output = token_classifier(input_sentence)
            print(f"Result: {output}")
        except Exception as e:
            raise NERException(e, sys) from e
