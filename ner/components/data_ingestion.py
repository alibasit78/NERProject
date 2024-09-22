import os
import sys
import time
from typing import Tuple

import pandas as pd

from ner.entity.artifact_entity import DataIngestionArtifact
from ner.entity.config_entity import DataIngestionConfig
from ner.exception import NERException
from ner.logger import logging
from ner.utils.utils import (
    dump_pickle_file,
    load_pickle_file,
)


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: config of data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NERException(e, sys)

    def prepare_data(self) -> Tuple[pd.DataFrame, list]:
        """
        Description: This method prepares the data in the proper row and columsn format.
        Note: In this project we are assuming that NER data is not in the proper format.

        Output: proper dataframe wrt NER is returned
        On Failure: Write an exception log and raise an exception
        """
        try:
            # dataframe = pd.read_csv(
            #     self.data_ingestion_config.feature_store_file_path, encoding="unicode_escape"
            # )
            dataframe = load_pickle_file(self.data_ingestion_config.feature_store_file_path)
            removed_tag_values = [
                "B-art",
                "I-art",
                "B-eve",
                "I-eve",
                "B-nat",
                "I-nat",
            ]  # The tag values are removed due to less occurence
            dataframe = dataframe[~dataframe["Tag"].isin(removed_tag_values)]
            tag_selected = list(dataframe["Tag"].unique())
            dataframe = dataframe.fillna(method="ffill")
            gk = dataframe.groupby(by="Sentence #")
            s_time = time.time()
            dataframe["Sentence"] = gk["Word"].transform(
                lambda x: "[SEP]".join([x_dash.strip() for x_dash in x])
            )
            dataframe["ner_tags"] = gk["Tag"].transform(
                lambda x: "[SEP]".join([x_dash.strip() for x_dash in x])
            )
            dataframe = (
                dataframe[["Sentence", "ner_tags"]].drop_duplicates().reset_index(drop=True)
            )
            # print(new_dataframe)
            e_time = time.time()
            dataframe["tokens"] = dataframe["Sentence"].apply(lambda x: x.split("[SEP]"))
            dataframe["Sentence"] = dataframe["Sentence"].apply(lambda x: x.replace("[SEP]", " "))
            dataframe["ner_tags"] = dataframe["ner_tags"].apply(lambda x: x.split("[SEP]"))
            logging.info(f"prepared data elapsed time: {(e_time - s_time)/60}")
            os.makedirs(
                os.path.dirname(self.data_ingestion_config.csv_data_file_path), exist_ok=True
            )
            dump_pickle_file(
                output_filepath=self.data_ingestion_config.csv_data_file_path, data=dataframe
            )
            logging.info(
                f"saved the prepared data into feature store - {self.data_ingestion_config.csv_data_file_path}"
            )
        except Exception as e:
            raise NERException(e, sys)
        return dataframe, tag_selected

    def export_data_to_artifact(self):
        try:
            # print("file_path: ", self.data_ingestion_config.data_file_path)
            dataframe = pd.read_csv(
                self.data_ingestion_config.data_file_path, encoding="unicode_escape"
            )
            os.makedirs(
                os.path.dirname(self.data_ingestion_config.feature_store_file_path), exist_ok=True
            )
            dump_pickle_file(
                output_filepath=os.path.join(self.data_ingestion_config.feature_store_file_path),
                data=dataframe,
            )
            logging.info(
                f"Exported the data from the source to the feature store dir - {self.data_ingestion_config.feature_store_file_path}"
            )
        except Exception as e:
            raise NERException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered in initiate data ingestion method of DataIngestion class")
        try:
            _ = self.export_data_to_artifact()
            _, tags = self.prepare_data()
            dump_pickle_file(self.data_ingestion_config.label_names_file_path, data=tags)
            logging.info("Saving a list of label names")
            data_ingestion_artifact = DataIngestionArtifact(
                csv_data_file_path=self.data_ingestion_config.csv_data_file_path,
                label_names_file_path=self.data_ingestion_config.label_names_file_path,
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise NERException(e, sys)
