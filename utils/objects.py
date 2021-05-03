import attr
import numpy as np

@attr.s
class Dataset(object):
    train_items: list = attr.ib(default=[])
    dev_items: list = attr.ib(default=[])
    test_items: list = attr.ib(default=[])


@attr.s
class TokenizedItem:
    tokens: list = attr.ib()
    labels: list = attr.ib()


@attr.s
class ProcessedData:
    source_train_data: np.ndarray = attr.ib()
    source_dev_data: np.ndarray = attr.ib()
    source_test_data: np.ndarray = attr.ib()
    target_train_data: np.ndarray = attr.ib()
    target_dev_data: np.ndarray = attr.ib()
    target_test_data: np.ndarray = attr.ib()
    label2id: dict = attr.ib()
    id2label: dict = attr.ib()

@attr.s
class TrainConfig:
    pretrained_path: str = attr.ib()
    save_folder: str = attr.ib()
    batch_size: int = attr.ib()
    learning_rate: float = attr.ib()
    max_epoch: int = attr.ib()
    max_length: int = attr.ib()
    f1_avg: str = attr.ib()
    early_stopping: str = attr.ib()
    patience: int = attr.ib()
    device: str = attr.ib()