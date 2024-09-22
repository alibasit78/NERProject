import os
import sys
from functools import partial

import pandas as pd
from datasets import (
    Dataset,
    DatasetDict,
)
from sklearn.model_selection import train_test_split

from ner.constants import (
    DATA_INGESTED_TRAIN_TEST_SPLIT_RATIO,
    DATA_INGESTED_TRAIN_VAL_SPLIT_RATIO,
    SEED,
)
from ner.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from ner.entity.config_entity import DataTransformationConfig
from ner.exception import NERException
from ner.logger import logging
from ner.utils.utils import (
    dump_pickle_file,
    load_pickle_file,
)


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
    ) -> None:
        """
        :param data_ingestion_artifact: Output referebce of data ingestion staged
        """
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config

    def split_the_dataset(self, df: pd.DataFrame):
        """
        Method Description: Splitting the dataset into train, test, and  val
        :param df: Dataframe
        """
        train, test = train_test_split(
            df, test_size=DATA_INGESTED_TRAIN_TEST_SPLIT_RATIO, random_state=SEED, shuffle=True
        )
        train, val = train_test_split(
            train, test_size=DATA_INGESTED_TRAIN_VAL_SPLIT_RATIO, random_state=SEED
        )
        logging.info("Splitted the dataset into train, test and val")
        return train, test, val

    def convert_dataset_to_datasetdict(
        self, train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame
    ):
        logging.info("Converting the train, test and val dataframe into DatasetDict")
        ds = DatasetDict()
        train_ner_ds = Dataset.from_pandas(train)
        test_ner_ds = Dataset.from_pandas(test)
        val_ner_ds = Dataset.from_pandas(val)
        ds["train"] = train_ner_ds
        ds["test"] = test_ner_ds
        ds["val"] = val_ner_ds
        ds["train"] = ds["train"].remove_columns("__index_level_0__")
        ds["test"] = ds["test"].remove_columns("__index_level_0__")
        ds["val"] = ds["val"].remove_columns("__index_level_0__")
        return ds

    def get_label_names(self, df: pd.DataFrame):
        logging.info("Creating the label names of the dataset in the proper order")
        label_names = load_pickle_file(self.data_ingestion_artifact.label_names_file_path)
        # label_names = list(df["ner_tags"].unique())
        # print(len(label_names))
        label_names = set([label.replace("B-", "").replace("I-", "") for label in label_names])
        new_label_names = ["O"]
        for label in label_names:
            if label != "O":
                new_label_names.extend(["B-" + label, "I-" + label])
        return new_label_names

    @staticmethod
    def get_label_to_id_and_id_to_label(label_names: list):
        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}
        return label2id, id2label

    @staticmethod
    def map_label2id(ner_tags, label_to_ix):
        return [label_to_ix[label] for label in ner_tags]

    @staticmethod
    def map_label2id_batches(examples, label_to_id):
        # print(len(examples))
        all_ner_tags = examples["ner_tags"]
        labels = []
        for ner_tags in all_ner_tags:
            # print(example)
            labels.append(DataTransformation.map_label2id(ner_tags, label_to_id))
        examples["labels"] = labels
        return examples

    @staticmethod
    def add_gold_label_ids(ds: DatasetDict, label_to_id: dict):
        logging.info("Adding the gold_label ids in the Dataset dict")
        ds = ds.map(
            partial(DataTransformation.map_label2id_batches, label_to_id=label_to_id), batched=True
        )
        return ds

    def initiate_data_transformation(self):
        logging.info("Entered in initiate data transformation method of DataTranformation class")
        try:
            os.makedirs(
                self.data_transformation_config.data_transformation_artifact_dir, exist_ok=True
            )
            logging.info(
                f"Created the dir- {self.data_transformation_config.data_transformation_artifact_dir}"
            )
            # df = pd.read_csv(self.data_ingestion_artifact.csv_data_file_path)
            df = load_pickle_file(self.data_ingestion_artifact.csv_data_file_path)
            logging.info(f"Shape of df: {df.shape} and columns: {df.columns}")

            label_names = self.get_label_names(df)
            logging.info(f"label names are: {label_names}")

            dump_pickle_file(
                output_filepath=self.data_transformation_config.data_label_names_path,
                data=label_names,
            )
            logging.info("Saving the label names to the transformation artifact")

            label_to_id, id_to_label = self.get_label_to_id_and_id_to_label(label_names)
            dump_pickle_file(
                output_filepath=self.data_transformation_config.data_label_to_id_path,
                data=label_to_id,
            )
            dump_pickle_file(
                output_filepath=self.data_transformation_config.data_id_to_label_path,
                data=id_to_label,
            )
            logging.info("Saved the label_to_id and id_to_label files")
            train, test, val = self.split_the_dataset(df)
            dataset_dict = self.convert_dataset_to_datasetdict(train, test, val)
            logging.info(f"dataset dict: {dataset_dict}\n{label_to_id}")
            dataset_dict = DataTransformation.add_gold_label_ids(dataset_dict, label_to_id)
            dataset_dict.save_to_disk(self.data_transformation_config.dataset_dict_path)
            logging.info("Saving the dataset_dict (train, test and val) of HF to the disk")
            data_transformation_artifact = DataTransformationArtifact(
                data_label_to_id_path=self.data_transformation_config.data_label_to_id_path,
                data_id_to_label_path=self.data_transformation_config.data_id_to_label_path,
                dataset_dict_path=self.data_transformation_config.dataset_dict_path,
                data_label_names_path=self.data_transformation_config.data_label_names_path,
            )
            logging.info("Exiting the initiate_data_transformation method")
            return data_transformation_artifact
        except Exception as e:
            raise NERException(e, sys)
