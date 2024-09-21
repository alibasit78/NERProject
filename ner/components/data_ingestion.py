import os
import sys

import pandas as pd

from ner.entity.artifact_entity import DataIngestionArtifact
from ner.entity.config_entity import DataIngestionConfig
from ner.exception import NERException
from ner.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: config of data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NERException(e, sys)

    def prepare_data(self) -> pd.DataFrame:
        """
        Description: This method prepares the data in the proper row and columsn format.
        Note: In this project we are assuming that NER data is not in the proper format.

        Output: proper dataframe wrt NER is returned
        On Failure: Write an exception log and raise an exception
        """
        dataframe = pd.read_csv(
            self.data_ingestion_config.feature_store_file_path, encoding="unicode_escape"
        )
        # remove_cols = ["B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"]
        # ner_df = ner_df[~ner_df["Tag"].isin(remove_cols)]
        # logging.info(
        #     f"Prepared the data and saved at {self.data_ingestion_config.feature_store_file_path}"
        # )
        return dataframe

    # def export_data_into_feature_store(self) -> pd.DataFrame:
    #     """
    #     Method Name :   export_data_into_feature_store
    #     Description :   This method exports data from data dir to csv file

    #     Output      :   data is returned as artifact of data ingestion components
    #     On Failure  :   Write an exception log and then raise an exception
    #     """
    #     try:
    #         logging.info("Exporting data from data dir")
    #         # usvisa_data = USvisaData()
    #         # dataframe = usvisa_data.export_collection_as_dataframe(collection_name=
    #         #                                                        self.data_ingestion_config.collection_name)
    #         dataframe = self.prepare_data()
    #         logging.info(f"Shape of dataframe: {dataframe.shape}")
    #         feature_store_file_path = self.data_ingestion_config.feature_store_file_path
    #         dir_path = os.path.dirname(feature_store_file_path)
    #         os.makedirs(dir_path, exist_ok=True)
    #         logging.info(
    #             f"Saving exported data into feature store file path: {feature_store_file_path}"
    #         )
    #         dataframe.to_csv(feature_store_file_path, index=False, header=True)
    #         return dataframe

    #     except Exception as e:
    #         raise NERException(e, sys)
    def export_data_to_artifact(self):
        try:
            # print("file_path: ", self.data_ingestion_config.data_file_path)
            dataframe = pd.read_csv(
                self.data_ingestion_config.data_file_path, encoding="unicode_escape"
            )
            os.makedirs(
                os.path.dirname(self.data_ingestion_config.feature_store_file_path), exist_ok=True
            )
            dataframe.to_csv(
                os.path.join(self.data_ingestion_config.feature_store_file_path), index=False
            )
            logging.info(
                f"Exported the data from the source to the feature store dir - {self.data_ingestion_config.feature_store_file_path}"
            )
        except Exception as e:
            raise NERException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Preparing the data in proper format")
        try:
            dataframe = self.export_data_to_artifact()
            dataframe = self.prepare_data()
            os.makedirs(
                os.path.dirname(self.data_ingestion_config.csv_data_file_path), exist_ok=True
            )
            dataframe.to_csv(self.data_ingestion_config.csv_data_file_path, index=False)
            logging.info(
                f"saved the prepared data into feature store - {self.data_ingestion_config.csv_data_file_path}"
            )

            data_ingestion_artifact = DataIngestionArtifact(
                csv_data_file_path=self.data_ingestion_config.csv_data_file_path
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise NERException(e, sys)
