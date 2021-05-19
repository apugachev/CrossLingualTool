from utils import constants as cnts
import attr
import json
from typing import List, Dict, Set, Union
import logging
import numpy as np
from progressbar import progressbar as pb
from transformers import BertTokenizer

from utils.data_loader import Dataset
from utils.objects import TokenizedItem, ProcessedData


class NERDataProcessor:
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

    def _convert_labels_to_iob(self,
                               labels: List[str]
                               ) -> List[str]:
        """
        Convert labels to IOB2 format
        Details: https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)

        :param labels: list, labels without IOB2 format
            ex. -> ["LOC", "LOC", "O", "PER"]
        :return: list, formatted labels
            ex. -> ["B-LOC", "I-LOC", "O", "B-PER"]
        """
        new_labels = []
        for i, label in enumerate(labels):
            if i == 0 and label != cnts.O_ENTITY:
                new_label = cnts.B_PREFIX + label
            elif label == cnts.O_ENTITY:
                new_label = cnts.O_ENTITY
            elif label == labels[i - 1]:
                new_label = cnts.I_PREFIX + label
            else:
                new_label = cnts.B_PREFIX + label

            new_labels.append(new_label)

        return new_labels

    def _lowercase_tokens(self, data: Dict) -> Dict:

        for key, items in data.items():
            for item in items:
                tokens = item[cnts.TOKEN_FIELD]
                item[cnts.TOKEN_FIELD] = [token.lower() for token in tokens]

        return data

    def _process_data_labels(self, data: Dataset) -> Dict:
        """
        Apply label convertation to data labels IOB2 format if necessary

        :param data: Dataset, data to be processed
        :return: dict, processed data
        """
        data = attr.asdict(data)

        for key, items in data.items():
            for item in items:
                labels = item[cnts.LABEL_FIELD]
                if all(not label.startswith(cnts.B_PREFIX) for label in labels) and set(labels) != {cnts.O_ENTITY}:
                    item[cnts.LABEL_FIELD] = self._convert_labels_to_iob(labels)

        return data

    def _get_uniq_labels(self, data: Dict) -> Set[str]:
        uniq_labels = set()

        for key, items in data.items():
            for item in items:
                labels = [label.split("-")[-1] for label in item[cnts.LABEL_FIELD]]
                uniq_labels.update(labels)

        uniq_bio_labels = set([cnts.B_PREFIX + label if label != cnts.O_ENTITY else label for label in uniq_labels])
        uniq_bio_labels.update([cnts.I_PREFIX + label if label != cnts.O_ENTITY else label for label in uniq_labels])

        return uniq_bio_labels

    def _apply_entity_mapping(self, data: Dict) -> Dict:
        """
        Rename entity labels according to entity mapping
        """
        for key, items in data.items():
            for item in items:
                new_labels = []
                for label in item[cnts.LABEL_FIELD]:
                    entity = label.split("-")[-1]
                    if entity in self.entity_mapping:
                        label = label.replace(entity, self.entity_mapping[entity])
                    new_labels.append(label)
                item[cnts.LABEL_FIELD] = new_labels

        self.logger.info("Target labels renamed according to entity mapping")
        return data

    def _get_label_mapping(self,
                           common_labels: Set[str],
                           all_labels: Set[str]) -> Dict[str, str]:
        """
        Build label mapping based on common labels
        Common labels remain the same, other labels are turned to MISC

        ex. -> common_labels: {"B-LOC", "O"}
               all_labels: {"B-LOC", "B-PER", "O"}
               return: {"B-LOC": "B-LOC", "B-PER": "B-MISC", "O": "O"}

        :param common_labels: set, labels which occur in both source and target data
        :param all_labels: set, joint labels which occur in source or target data
        :return: dict, label mapping
        """
        label_mapping = dict()
        for label in all_labels:
            if label in common_labels:
                label_mapping[label] = label
            elif label.startswith(cnts.B_PREFIX):
                label_mapping[label] = cnts.B_MISC
            else:
                label_mapping[label] = cnts.I_MISC

        return label_mapping

    def _apply_label_mapping(self,
                             data: Dict,
                             label_mapping: Dict[str, str]):
        """
        Change labels according to label mapping
        :param data: dict, data
        :param label_mapping: dict, label mapping
        :return: dict, data with labels mapped
        """

        for key, items in data.items():
            for item in items:
                new_labels = []
                for label in item[cnts.LABEL_FIELD]:
                    new_label = label_mapping[label]
                    new_labels.append(new_label)
                item[cnts.LABEL_FIELD] = new_labels

        return data


    def _pad_sequence(self,
                      sequence: List[str],
                      max_len: int,
                      pad_token: Union[str, int]):
        if len(sequence) >= max_len:
            sequence = sequence[:max_len]
        else:
            sequence = sequence + [pad_token] * (max_len - len(sequence))
        return sequence

    def _process_tokens(self, tokens: List[str], labels: List[str]) -> TokenizedItem:
        """
        Perform tokenization using BertTokenizer, change labels according to BERT tokens

        :param tokens: list, tokens
        :param labels: list, labels
        :return: TokenizedItem, BERT token ids along with labels
        """
        bert_tokens, bio_labels = [self.tokenizer.cls_token], [cnts.O_ENTITY]

        for word, label in zip(tokens, labels):
            new_tokens = self.tokenizer.tokenize(word)
            bert_tokens.extend(new_tokens)

            new_labels = [label] * len(new_tokens)
            bio_labels.extend(new_labels)

        bert_tokens.append(self.tokenizer.sep_token)
        bio_labels.append(cnts.O_ENTITY)

        for i, (token, label) in enumerate(zip(bert_tokens, bio_labels)):
            if token.startswith(cnts.HH) and label.startswith(cnts.B_PREFIX):
                prefix, label = label.split("-")
                bio_labels[i] = cnts.I_PREFIX + label

        encoded_tokens = self.tokenizer.encode(bert_tokens, add_special_tokens=False)

        if len(bio_labels) >= self.max_len:
            bio_labels[self.max_len - 1] = cnts.O_ENTITY

        bio_labels = self._pad_sequence(bio_labels, self.max_len, cnts.O_ENTITY)
        encoded_tokens = self._pad_sequence(encoded_tokens, self.max_len, self.tokenizer.pad_token_id)

        return TokenizedItem(
            tokens=encoded_tokens,
            labels=bio_labels
        )

    def _process_data(self,
                      data_items: List[Dict[str, List[str]]],
                      label2id: Dict[str, str]
                      ) -> np.ndarray:
        """
        Process data for further training

        :param data_items: list, items to be processed
        :param label2id: list, label to id mapping
        :return: np.ndarray: array ready for training
        """

        token_ids = np.zeros((len(data_items), self.max_len), dtype=np.int32)
        label_ids = []

        for i, item in enumerate(pb(data_items)):
            tokens = item[cnts.TOKEN_FIELD]
            labels = item[cnts.LABEL_FIELD]

            processed = self._process_tokens(tokens, labels)
            new_token_ids, new_labels = processed.tokens, processed.labels
            new_label_ids = [label2id[label] for label in new_labels]

            token_ids[i] = new_token_ids
            label_ids.append(new_label_ids)

        result = np.stack((token_ids, label_ids), axis=1)

        return result

    def _get_label2id(self, common_labels: List[str]) -> Dict[str, int]:

        common_labels = sorted(common_labels)
        common_labels.remove(cnts.O_ENTITY)
        common_labels.insert(0, cnts.O_ENTITY)

        label2id = {label: idx for idx, label in enumerate(common_labels)}
        return label2id

    def process_data_for_ner(self,
                             source_data: Dataset,
                             target_data: Dataset
                             ) -> ProcessedData:
        """
        Prepare source and target data for training NER

        :param source_data: Dataset, source data
        :param target_data: Dataset, target data
        :return:
        """
        source_data = self._process_data_labels(source_data)
        target_data = self._process_data_labels(target_data)
        target_data = self._apply_entity_mapping(target_data)

        source_data = self._lowercase_tokens(source_data)
        target_data = self._lowercase_tokens(target_data)

        source_uniq_labels = self._get_uniq_labels(source_data)
        target_uniq_labels = self._get_uniq_labels(target_data)

        all_labels = source_uniq_labels | target_uniq_labels
        common_labels = source_uniq_labels & target_uniq_labels
        common_labels.update([cnts.B_MISC, cnts.I_MISC])

        label_mapping = self._get_label_mapping(common_labels, all_labels)
        source_data = self._apply_label_mapping(source_data, label_mapping)
        target_data = self._apply_label_mapping(target_data, label_mapping)
        self.logger.info("Label mapping created and applied to source and target data")

        label2id = self._get_label2id(list(common_labels))
        id2label = {idx: label for label, idx in label2id.items()}

        self.logger.info("Processing source data...")
        source_train_data = self._process_data(source_data["train_items"], label2id)
        source_dev_data = self._process_data(source_data["dev_items"], label2id)
        source_test_data = self._process_data(source_data["test_items"], label2id)

        self.logger.info("Processing target data...")
        target_train_data = self._process_data(target_data["train_items"], label2id)
        target_dev_data = self._process_data(target_data["dev_items"], label2id)
        target_test_data = self._process_data(target_data["test_items"], label2id)

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

