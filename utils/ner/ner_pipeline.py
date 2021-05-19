from utils import constants as cnts
import os
import logging
import json
from typing import List, Tuple
from progressbar import progressbar as pb

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, BertTokenizer
from sklearn.metrics import classification_report, f1_score, accuracy_score

from utils.objects import TrainConfig, ProcessedData


class NERTrainer:
    """
    Class for performing training and fine-tuning NER models
    """
    logger = logging.getLogger("NERTrainer")

    def __init__(self,
                 config: TrainConfig,
                 data: ProcessedData
                 ) -> None:
        self._config = config
        self._data = data
        self._tokenizer = BertTokenizer.from_pretrained(self._config.pretrained_path)

    def _train_epoch(self,
                     model: BertForTokenClassification,
                     train_loader: DataLoader,
                     optimizer: torch.optim,
                     epoch_num: int
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform one training epoch
        """

        train_loss, train_acc, train_f1 = [], [], []
        model.train()

        for batch_num, batch in enumerate(train_loader):
            X_batch, y_batch = batch[:, 0, :], batch[:, 1, :]
            X_batch = X_batch.type(torch.LongTensor).to(self._config.device)
            y_batch = y_batch.type(torch.LongTensor).to(self._config.device)
            optimizer.zero_grad()

            out = model(input_ids=X_batch, labels=y_batch.contiguous())
            loss = out.loss
            y_pred = out.logits
            y_pred = torch.argmax(y_pred, dim=2)

            y_pred_flatten = torch.flatten(y_pred).cpu().numpy()
            y_batch_flatten = torch.flatten(y_batch).cpu().numpy()
            f1 = f1_score(y_batch_flatten, y_pred_flatten, average=self._config.f1_avg)
            accuracy = accuracy_score(y_batch_flatten, y_pred_flatten)

            train_loss.append(loss.item())
            train_acc.append(accuracy)
            train_f1.append(f1)

            if batch_num % cnts.TRAIN_LOG_FREQ == 0:
                self.logger.info(f"TRAIN: Epoch: {epoch_num}, Batch: {batch_num + 1} / {len(train_loader)}, "
                      f"Loss: {loss.item():.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}")

            loss.backward()
            optimizer.step()

        return np.mean(train_loss), np.mean(train_acc), np.mean(train_f1)

    def _val_epoch(self,
                   model: BertForTokenClassification,
                   val_loader: DataLoader,
                   epoch_num: int
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform validation after one training epoch
        """

        val_loss, val_acc, val_f1 = [], [], []
        model.eval()

        for batch_num, batch in enumerate(val_loader):
            X_batch, y_batch = batch[:, 0, :], batch[:, 1, :]
            X_batch = X_batch.type(torch.LongTensor).to(self._config.device)
            y_batch = y_batch.type(torch.LongTensor).to(self._config.device)

            out = model(input_ids=X_batch, labels=y_batch.contiguous())
            loss = out.loss
            y_pred = out.logits
            y_pred = torch.argmax(y_pred, dim=2)

            y_pred_flatten = torch.flatten(y_pred).cpu().numpy()
            y_batch_flatten = torch.flatten(y_batch).cpu().numpy()
            f1 = f1_score(y_batch_flatten, y_pred_flatten, average=self._config.f1_avg)
            accuracy = accuracy_score(y_batch_flatten, y_pred_flatten)

            val_loss.append(loss.item())
            val_acc.append(accuracy)
            val_f1.append(f1)

            if batch_num % cnts.TRAIN_LOG_FREQ == 0:
                self.logger.info(f"DEV: Epoch: {epoch_num}, Batch: {batch_num + 1} / {len(val_loader)}, "
                      f"Loss: {loss.item():.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}")

        return np.mean(val_loss), np.mean(val_acc), np.mean(val_f1)

    def _init_model(self, path: str) -> BertForTokenClassification:
        """
        Initialize pretrained model from path
        """
        model = BertForTokenClassification.from_pretrained(path,
                                                           num_labels=len(self._data.id2label))
        model.to(self._config.device)
        self.logger.info(f"BERT model initialized from {path}")
        return model

    def _train(self,
               model: BertForTokenClassification,
               data_type: str,
               save_path: str) -> None:
        """
        Training the pre-trained model
        Best model according to early_stopping strategy along with id2label file are saved to save_folder

        :param model: BertForTokenClassification, model to be trained
        :param data_type: str, SOURCE or TARGET, data type which will be used for training
        :param save_path: str, path for saving trained model
        """

        if data_type == cnts.SOURCE:
            train_batches = DataLoader(self._data.source_train_data, batch_size=self._config.batch_size, shuffle=True)
            dev_batches = DataLoader(self._data.source_dev_data, batch_size=self._config.batch_size, shuffle=True)
        else:
            train_batches = DataLoader(self._data.target_train_data, batch_size=self._config.batch_size, shuffle=True)
            dev_batches = DataLoader(self._data.target_dev_data, batch_size=self._config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=self._config.learning_rate)
        dev_losses, dev_accuracies, dev_fscores = [], [], []
        last_epoch = 0

        for epoch in range(1, self._config.max_epoch + 1):
            train_loss, train_acc, train_f1 = self._train_epoch(model, train_batches, optimizer, epoch)
            dev_loss, dev_acc, dev_f1 = self._val_epoch(model, dev_batches, epoch)

            if self._config.early_stopping == "loss":
                if len(dev_losses) == 0 or dev_loss < dev_losses[-1]:
                    model.save_pretrained(save_path)
                    self.logger.info("Saved best model according to dev loss")
                elif last_epoch == 0:
                    last_epoch = epoch + self._config.patience
                    self.logger.info(f"Dev loss increased, stop training after {last_epoch} epoch")

            elif self._config.early_stopping == "accuracy":
                if len(dev_acc) == 0 or dev_acc > dev_accuracies[-1]:
                    model.save_pretrained(save_path)
                    self.logger.info("Saved best model according to dev accuracy")
                elif last_epoch == 0:
                    last_epoch = epoch + self._config.patience
                    self.logger.info(f"Dev accuracy decreased, stop training after {last_epoch} epoch")

            elif self._config.early_stopping == "f1":
                if len(dev_acc) == 0 or dev_f1 > dev_fscores[-1]:
                    model.save_pretrained(save_path)
                    self.logger.info("Saved best model according to dev F1 score")
                elif last_epoch == 0:
                    last_epoch = epoch + self._config.patience
                    self.logger.info(f"Dev F1 score decreased, stop training after {last_epoch} epoch")

            self.logger.info(f"After epoch #{epoch}:")
            self.logger.info(f"Train loss: {train_loss:.3f}, Train Accuracy: {train_acc:.3f}, Train F1: {train_f1:.3f}")
            self.logger.info(f"Dev loss: {dev_loss:.3f}, Dev Accuracy: {dev_acc:.3f}, Dev F1: {dev_f1:.3f}\n")

            dev_losses.append(dev_loss)
            dev_accuracies.append(dev_acc)
            dev_fscores.append(dev_f1)

            if epoch == last_epoch:
                self.logger.info(f"Stop training")
                break

        with open(os.path.join(save_path, cnts.ID2LABEL_FILENAME), 'w') as f:
            json.dump(self._data.id2label, f, ensure_ascii=False, indent=4)


    def _get_predictions(self,
                         model: BertForTokenClassification,
                         batches: DataLoader) -> List[int]:
        """
        Get trained model predictions

        :param model: BertForTokenClassification, trained model
        :param batches: DataLoader, test data
        :return: list, predicted label ids
        """

        pred_labels = []
        self.logger.info("Obtaining predictions...")
        for item in pb(batches):
            out = model(item.to(self._config.device))
            logits = out.logits
            logits = logits.argmax(axis=-1).tolist()
            pred_labels.extend(logits)

        return pred_labels

    def _eval(self,
              model: BertForTokenClassification,
              save_path: str,
              data: np.ndarray) -> None:

        """
        Perform evaluation of trained model
        Results are saved as sklearn.metrics.classification_report to save_path

        :param model: BertForTokenClassification, model to be evaluated
        :param save_path: str, path for saving results
        :param data: np.ndarray, data for model evaluation
        :return:
        """

        test_tokens, test_labels = data[:, 0, :], data[:, 1, :]
        test_batches = DataLoader(test_tokens, batch_size=self._config.batch_size, shuffle=True)
        pred_labels = self._get_predictions(model, test_batches)

        pred_extended, test_extended = self._prepare_preds_for_calc_metrics(pred_labels,
                                                                            test_labels,
                                                                            test_tokens)

        metrics = classification_report(test_extended, pred_extended, digits=4, zero_division=0, output_dict=True)

        results_path = os.path.join(save_path, cnts.RESULTS_FILENAME)
        with open(results_path, "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

        self.logger.info(f"Model has been evaluated. Results saved to {results_path}")

    def _prepare_preds_for_calc_metrics(self,
                                        pred_labels: List[int],
                                        test_labels: List[int],
                                        test_tokens: List[int]
                                        ) -> Tuple[List[str], List[str]]:
        """
        Convert test and pred labels to classification_report-ready format

        :param pred_labels: list, predicted label ids
        :param test_labels: list, ground truth label ids
        :param test_tokens: list, BERT token ids
        :return: tuple, predicted labels, ground truth labels
        """
        en_pred_extended, en_test_extended = [], []

        for pred, test, words in zip(pred_labels, test_labels, test_tokens):
            words = words.tolist()

            words_decoded = self._tokenizer.convert_ids_to_tokens(words)

            if self._tokenizer.pad_token_id in words:
                cut_ind = words.index(self._tokenizer.pad_token_id)
            else:
                cut_ind = self._config.max_length

            pred, test = pred[:cut_ind], test[:cut_ind]
            pred, test = pred[1:-1], test[1:-1]

            pred = [self._data.id2label[num] for i, num in enumerate(pred)
                    if not words_decoded[i].startswith(cnts.HH)]
            test = [self._data.id2label[num] for i, num in enumerate(test)
                    if not words_decoded[i].startswith(cnts.HH)]

            en_pred_extended.extend(pred)
            en_test_extended.extend(test)

            assert len(pred) == len(test)

        return en_pred_extended, en_test_extended

    def run_pipeline(self) -> None:
        """
        Perform 3-step pipeline:
            1. Training model from scratch on source data
            2. Fine-tuning trained model on target data
            3. Training model from scratch on target data

        After each step trained models, id2label file along with evaluation results are saved to save_folder
        """

        models_folder = os.path.join(self._config.save_folder, cnts.MODELS_FOLDER)
        results_folder = os.path.join(self._config.save_folder, cnts.RESULTS_FOLDER)

        if not os.path.isdir(self._config.save_folder):
            os.mkdir(self._config.save_folder)

        for folder in [models_folder, results_folder]:
            if not os.path.isdir(folder):
                os.mkdir(folder)

            for data_type in [cnts.SOURCE, cnts.FINETUNED, cnts.TARGET]:
                path = os.path.join(folder, data_type)
                if not os.path.isdir(path):
                    os.mkdir(path)

        self.logger.info("Training model on source data...")
        model = self._init_model(self._config.pretrained_path)
        source_model_save_path = os.path.join(models_folder, cnts.SOURCE)
        self._train(model, cnts.SOURCE, source_model_save_path)

        self.logger.info("Evaluating model on source data...")
        model = self._init_model(source_model_save_path)
        source_result_save_path = os.path.join(results_folder, cnts.SOURCE)
        self._eval(model, source_result_save_path, self._data.source_test_data)

        self.logger.info("Fine-tuning model on target data...")
        model = self._init_model(source_model_save_path)
        finetuned_model_save_path = os.path.join(models_folder, cnts.FINETUNED)
        self._train(model, cnts.TARGET, finetuned_model_save_path)

        self.logger.info("Evaluating fine-tuned model on target data...")
        model = self._init_model(source_model_save_path)
        finetuned_model_save_path = os.path.join(results_folder, cnts.FINETUNED)
        self._eval(model, finetuned_model_save_path, self._data.target_test_data)

        self.logger.info("Training model on target data...")
        model = self._init_model(self._config.pretrained_path)
        target_model_save_path = os.path.join(models_folder, cnts.TARGET)
        self._train(model, cnts.TARGET, target_model_save_path)

        self.logger.info("Evaluating model on target data...")
        model = self._init_model(target_model_save_path)
        target_result_save_path = os.path.join(results_folder, cnts.TARGET)
        self._eval(model, target_result_save_path, self._data.target_test_data)
