import os
from utils import constants as cnts
import json
from typing import List, Dict
import logging
from abc import ABC, abstractmethod

from utils.objects import Dataset


class DataLoader(ABC):

    @abstractmethod
    def _validate_data(self,
                       file_name: str,
                       data: List[Dict[str, List]]
                       ) -> bool:
        pass

    def _load_from_disk(self,
                        folder_path: str,
                        file_name: str
                        ) -> List[Dict[str, List]]:

        path_to_file = os.path.join(folder_path, file_name)
        with open(path_to_file, "r") as f:
            data = json.load(f)

        return data

    @abstractmethod
    def get_data(self,
                 folder_path
                 ) -> Dataset:
        pass


class NERDataLoader(DataLoader):
    """
    Class for loading NER data from disk
    """

    def __init__(self) -> None:
        pass

    def _validate_data(self,
                       file_name: str,
                       data: List[Dict[str, List]]
                       ) -> bool:
        """
        Check whether each data element has:
            - 'token' and 'label' fields
            - equal length of 'token' and 'label' fields

        :param file_name: str, file name
        :param data: list, data extracted from file
        :return: bool
        """

        for i, item in enumerate(data):
            for field in [cnts.TOKEN_FIELD, cnts.LABEL_FIELD]:
                if field not in item:
                    raise AttributeError(f"Item {i} in file '{file_name}' does not have required field '{field}'")

            tokens_len = len(item[cnts.TOKEN_FIELD])
            labels_len = len(item[cnts.LABEL_FIELD])
            assert tokens_len == labels_len, \
                f"Length of tokens and labels must be equal. Item {i}, tokens: {tokens_len}, labels: {labels_len}"

        return True

    def get_data(self, folder_path: str) -> Dataset:
        """
        Load data from folder path
        Expects folder to have train.json, dev.json, test.json files

        :param folder_path: path to folder with files
        :return: Dataset
        """

        logger = logging.getLogger("DataLoader")
        dataset = Dataset()

        train_items = self._load_from_disk(folder_path, cnts.TRAIN_FILE)
        if self._validate_data(cnts.TRAIN_FILE, train_items):
            dataset.train_items = train_items
        logger.info(f"Successfully loaded {len(dataset.train_items)} train items")

        dev_items = self._load_from_disk(folder_path, cnts.DEV_FILE)
        if self._validate_data(cnts.DEV_FILE, dev_items):
            dataset.dev_items = dev_items
        logger.info(f"Successfully loaded {len(dataset.dev_items)} dev items")

        test_items = self._load_from_disk(folder_path, cnts.TEST_FILE)
        if self._validate_data(cnts.TEST_FILE, test_items):
            dataset.test_items = test_items
        logger.info(f"Successfully loaded {len(dataset.test_items)} test items")

        return dataset


class REDataLoader(DataLoader):
    """
        Class for loading RE data from disk
    """

    def __init__(self) -> None:
        pass

    def _validate_data(self,
                       file_name: str,
                       data: List[Dict[str, List]]
                       ) -> bool:
        """
            Check whether each data element has:
                - 'token', 'relation', 'subj_start', 'subj_end', 'obj_start', 'obj_end' fields

            :param file_name: str, file name
            :param data: list, data extracted from file
            :return: bool
        """

        for i, item in enumerate(data):
            for field in [cnts.TOKEN_FIELD, cnts.RELATION_FIELD, cnts.SUBJ_START_FIELD,
                          cnts.SUBJ_END_FIELD, cnts.OBJ_START_FIELD, cnts.OBJ_END_FIELD]:
                if field not in item:
                    raise AttributeError(f"Item {i} in file '{file_name}' does not have required field '{field}'")

        return True

    def get_data(self, folder_path: str) -> Dataset:
        """
        Load data from folder path
        Expects folder to have train.json, dev.json, test.json files

        :param folder_path: path to folder with files
        :return: Dataset
        """

        logger = logging.getLogger("DataLoader")
        dataset = Dataset()

        train_items = self._load_from_disk(folder_path, cnts.TRAIN_FILE)
        if self._validate_data(cnts.TRAIN_FILE, train_items):
            dataset.train_items = train_items
        logger.info(f"Successfully loaded {len(dataset.train_items)} train items")

        dev_items = self._load_from_disk(folder_path, cnts.DEV_FILE)
        if self._validate_data(cnts.DEV_FILE, dev_items):
            dataset.dev_items = dev_items
        logger.info(f"Successfully loaded {len(dataset.dev_items)} dev items")

        test_items = self._load_from_disk(folder_path, cnts.TEST_FILE)
        if self._validate_data(cnts.TEST_FILE, test_items):
            dataset.test_items = test_items
        logger.info(f"Successfully loaded {len(dataset.test_items)} test items")

        return dataset

