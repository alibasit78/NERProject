import pickle
import sys

from ner.exception import NERException


def dump_pickle_file(output_filepath: str, data) -> None:
    try:
        with open(output_filepath, "wb") as encoded_pickle:
            pickle.dump(data, encoded_pickle)

    except Exception as e:
        raise NERException(e, sys) from e


def load_pickle_file(filepath: str) -> object:
    try:
        with open(filepath, "rb") as pickle_obj:
            obj = pickle.load(pickle_obj)
        return obj

    except Exception as e:
        raise NERException(e, sys) from e
