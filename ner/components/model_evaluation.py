import os
import sys
from functools import partial

import datasets
import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from ner.components.model_trainer import ModelTraining
from ner.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelEvalArtifact,
    ModelTrainingArtifacts,
)
from ner.entity.config_entity import ModelEvalConfig
from ner.exception import NERException
from ner.logger import logging


class ModelEvaluation:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_training_artifact: ModelTrainingArtifacts,
        model_evaluation_config: ModelEvalConfig,
    ) -> None:
        self.data_transformation_artifact = data_transformation_artifact
        self.model_training_artifact = model_training_artifact
        self.model_evaluation_config = model_evaluation_config

    @staticmethod
    def load_metric():
        return evaluate.load("seqeval")

    def load_saved_model(self, device):
        checkpoint_dir = os.listdir(self.model_training_artifact.model_saved_dir)[0]
        model = AutoModelForTokenClassification.from_pretrained(
            os.path.join(self.model_training_artifact.model_saved_dir, checkpoint_dir),
        )
        print("device: ", device)
        model.to(device)
        return model

    @staticmethod
    def postprocess(predictions, labels, label_names):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    @staticmethod
    def evaluate(model, test_dataloader, device, label_names, metric):
        for i, batch in enumerate(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # print("outputs: ", outputs.logits.shape)
            prob = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # print("prob: ", prob.shape)
            predictions = torch.argmax(prob, dim=-1)
            labels = batch["labels"]
            # print("predictions: ", predictions.shape)
            # print("labels: ", labels.shape)
            true_predictions, true_labels = ModelEvaluation.postprocess(
                predictions, labels, label_names
            )
            # print("true_predictions: ", true_predictions)
            # print("true_labels: ", true_labels)
            metric.add_batch(predictions=true_predictions, references=true_labels)
        results = metric.compute()
        print(
            {key: results[f"overall_{key}"] for key in ["precision", "recall", "f1", "accuracy"]}
        )
        return results

    def initiate_model_evaluation(self):
        try:
            logging.info("Entered in initiate_model_evaluation method of ModelEvaluation")
            os.makedirs(self.model_evaluation_config.model_evaluation_artifact_dir, exist_ok=True)
            logging.info(
                f"{self.model_evaluation_config.model_evaluation_artifact_dir} dir is created"
            )

            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            model = self.load_saved_model(device)
            logging.info("Loaded BERT model")

            dataset_dict_path = self.data_transformation_artifact.dataset_dict_path
            test_dataset = datasets.load_from_disk(dataset_dict_path)["test"]
            logging.info(f"Loaded the tokenized test dataset {test_dataset}")

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_training_artifact.model_checkpoint_name
            )
            logging.info("tokenizer is downloaded")
            tokenized_test_dataset = test_dataset.map(
                partial(ModelTraining.tokenize_and_align_labels, tokenizer=tokenizer),
                batched=True,
                remove_columns=test_dataset.column_names,
            )
            logging.info("Dataset is tokeinzed")
            data_collator = DataCollatorForTokenClassification(
                tokenizer=tokenizer, padding="longest"
            )
            test_dataloader = DataLoader(
                tokenized_test_dataset, collate_fn=data_collator, batch_size=8
            )
            metric = ModelEvaluation.load_metric()
            label_names = self.data_transformation_artifact.data_label_names_path
            _ = ModelEvaluation.evaluate(model, test_dataloader, device, label_names, metric)
            logging.info("Exited the initiate_model_evaluation method")
            return ModelEvalArtifact(
                model_eval_artifact_dir=self.model_evaluation_config.model_evaluation_artifact_dir
            )
        except Exception as e:
            raise NERException(e, sys)
