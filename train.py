import sys

from ner.exception import NERException
from ner.pipline.training_pipeline import TrainPipeline


def training():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
    except Exception as e:
        raise NERException(e, sys) from e


if __name__ == "__main__":
    training()
