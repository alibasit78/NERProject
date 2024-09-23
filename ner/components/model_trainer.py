import os
import sys
from functools import partial

import datasets
import evaluate
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ner.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainingArtifacts,
)
from ner.entity.config_entity import ModelTrainingConfig
from ner.exception import NERException
from ner.logger import logging
from ner.utils.utils import load_pickle_file


class ModelTraining:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainingConfig,
    ) -> None:
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def align_labels_with_tokens(self, labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                # label_name = ix_to_label[label]
                if label % 2 == 1:
                    # if label_name.startswith("B-"):
                    label += 1
                new_labels.append(label)
        return new_labels

    @staticmethod
    def load_metric():
        return evaluate.load("seqeval")

    def tokenize_and_align_labels(self, examples, tokenizer):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        all_labels = examples["labels"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    @staticmethod
    def prepare_compute_metrics(label_names, metric):
        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)

            # Remove ignored index (special tokens) and convert to labels
            true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
            true_predictions = [
                [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": all_metrics["overall_precision"],
                "recall": all_metrics["overall_recall"],
                "f1": all_metrics["overall_f1"],
                "accuracy": all_metrics["overall_accuracy"],
            }

    def initiate_model_training(self) -> ModelTrainingArtifacts:
        try:
            logging.info("Entered the initiate_model_training method of ModelTraining class")

            os.makedirs(self.model_trainer_config.model_training_artifact_dir, exist_ok=True)
            logging.info(
                f"Created the model artifact dir - {self.model_trainer_config.model_training_artifact_dir}"
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_trainer_config.model_checkpoint_path
            )
            logging.info("tokenizer is downloaded")

            dataset_dict_path = self.data_transformation_artifact.dataset_dict_path
            ds = datasets.load_from_disk(dataset_dict_path)

            tokenized_datasets = ds.map(
                partial(self.tokenize_and_align_labels, tokenizer=tokenizer),
                batched=True,
                remove_columns=ds["train"].column_names,
            )
            logging.info("Dataset is tokeinzed")

            label_names = load_pickle_file(self.data_transformation_artifact.data_label_names_path)
            logging.info("Loaded the label names")

            id2label = load_pickle_file(self.data_transformation_artifact.data_id_to_label_path)
            label2id = load_pickle_file(self.data_transformation_artifact.data_label_to_id_path)
            logging.info("Loaded the id2lable and label2id")

            data_collator = DataCollatorForTokenClassification(
                tokenizer=tokenizer, padding="longest"
            )

            model = AutoModelForTokenClassification.from_pretrained(
                self.model_trainer_config.model_checkpoint_path,
                id2label=id2label,
                label2id=label2id,
            )

            args = TrainingArguments(
                # "bert-finetuned-ner",
                output_dir=self.model_trainer_config.model_file_path,
                gradient_accumulation_steps=8,
                load_best_model_at_end=True,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=4,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=1e-5,
                weight_decay=0.01,
                greater_is_better=True,
                overwrite_output_dir=True,
                save_total_limit=1,
                logging_strategy="epoch",
                metric_for_best_model="f1",  # f1
                # push_to_hub=True,
            )
            metric = self.load_metric()
            compute_metrics = self.prepare_compute_metrics(label_names=label_names, metric=metric)
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["val"],
                # test_dataset = tokenized_datasets['test'],
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
            )
            trainer.train()
        except Exception as e:
            raise NERException(e, sys)
