from utils import constants as cnts
import attr
import json
from typing import List, Dict, Set, Union, Tuple, Any
from copy import copy
import logging
import numpy as np
from progressbar import progressbar as pb
from transformers import BertTokenizer

from utils.data_loader import Dataset
from utils.objects import TokenizedItem, ProcessedData


class REDataProcessor:
    """
    Class for preparation data for training
    """
    logger = logging.getLogger("NERDataProcessor")

    def __init__(self,
                 entity_mapping_path: str,
                 pretrained_path: str,
                 max_len: int
                 ) -> None:
        """
        :param entity_mapping_path: str, path to entity mapping file
        :param pretrained_path: str, path to BERT pretrained model (for tokenizer)
        :param max_len: int, maximum sequence length (for tokenizer)
        """

        with open(entity_mapping_path, "r") as f:
            self.entity_mapping = json.load(f)
            self.logger.info("Entity mapping loaded")

        self.logger.info("Tokenizer initialization...")
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        self.max_len = max_len

    def _process_labels(self,
                        source_data: Dict,
                        target_data: Dict
                        ) -> Tuple[Dict, Dict]:
        """
        Rename relation labels according to entity mapping
        The source labels are renamed in accordance with target labels
        Relation labels which appear only in source or target data are renamed to MISC_RELATION
        """

        for key, items in source_data.items():
            for item in items:
                relation = item[cnts.RELATION_FIELD]
                item[cnts.RELATION_FIELD] = self.entity_mapping.get(relation, cnts.MISC_RELATION)

        for key, items in target_data.items():
            for item in items:
                relation = item[cnts.RELATION_FIELD]
                if relation not in self.entity_mapping.values():
                    item[cnts.RELATION_FIELD] = cnts.MISC_RELATION

        return source_data, target_data

    def _get_label2id(self) -> Dict[str, int]:

        uniq_labels = set(self.entity_mapping.values())
        uniq_labels.add(cnts.MISC_RELATION)
        uniq_labels = sorted(list(uniq_labels))

        uniq_labels.remove(cnts.NO_RELATION)
        uniq_labels.insert(0, cnts.NO_RELATION)

        label2id = {label: idx for idx, label in enumerate(uniq_labels)}
        return label2id

    def _process_data(self,
                      data_items: List[Dict[str, List[str]]],
                      label2id: Dict[str, int]
                      ) -> List[Tuple[Any, int]]:
        """
            Process data for further training

            :param data_items: list, items to be processed
            :param label2id: list, label to id mapping
            :return: list of tuples: encoded texts with relation id
        """

        result = []

        for i, item in enumerate(pb(data_items)):
            tokens = copy(item[cnts.TOKEN_FIELD])
            relation = item[cnts.RELATION_FIELD]

            subj_start = item[cnts.SUBJ_START_FIELD]
            subj_end = item[cnts.SUBJ_END_FIELD] + 1
            obj_start = item[cnts.OBJ_START_FIELD]
            obj_end = item[cnts.OBJ_END_FIELD] + 1

            if obj_end > subj_end:
                tokens.insert(obj_end, cnts.OBJ_END_TOKEN)
                tokens.insert(obj_start, cnts.OBJ_START_TOKEN)
                tokens.insert(subj_end, cnts.SUBJ_END_TOKEN)
                tokens.insert(subj_start, cnts.SUBJ_START_TOKEN)
            else:
                tokens.insert(subj_end, cnts.SUBJ_END_TOKEN)
                tokens.insert(subj_start, cnts.SUBJ_START_TOKEN)
                tokens.insert(obj_end, cnts.OBJ_END_TOKEN)
                tokens.insert(obj_start, cnts.OBJ_START_TOKEN)

            text = " ".join(tokens).lower()
            token_ids = self.tokenizer.encode(text, max_length=self.max_len, padding="max_length", truncation=True)
            relation_id = label2id[relation]

            result.append((token_ids, relation_id))

        return result

    def process_data_for_re(self,
                            source_data: Dataset,
                            target_data: Dataset
                            ) -> ProcessedData:

        source_data = attr.asdict(source_data)
        target_data = attr.asdict(target_data)
        source_data, target_data = self._process_labels(source_data, target_data)

        label2id = self._get_label2id()
        id2label = {idx: label for label, idx in label2id.items()}

        self.logger.info("Processing source data...")
        source_train_data = self._process_data(source_data["train_items"][:16], label2id)
        source_dev_data = self._process_data(source_data["dev_items"][:16], label2id)
        source_test_data = self._process_data(source_data["test_items"][:16], label2id)

        self.logger.info("Processing target data...")
        target_train_data = self._process_data(target_data["train_items"][:16], label2id)
        target_dev_data = self._process_data(target_data["dev_items"][:16], label2id)
        target_test_data = self._process_data(target_data["test_items"][:16], label2id)

        return ProcessedData(
            source_train_data=source_train_data,
            source_dev_data=source_dev_data,
            source_test_data=source_test_data,
            target_train_data=target_train_data,
            target_dev_data=target_dev_data,
            target_test_data=target_test_data,
            label2id=label2id,
            id2label=id2label
        )



