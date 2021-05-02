import argparse
import logging
from logging import FileHandler, StreamHandler
import constants as cnts
import os
import datetime
from utils.data_loader import DataLoader
from utils.data_processor import NERDataProcessor

parser = argparse.ArgumentParser()

# Data params
parser.add_argument("--source_data_path",
                    dest="source_path",
                    help="Path to source language data")
parser.add_argument("--target_data_path",
                    dest="target_path",
                    help="Path to target language data")
parser.add_argument("--entity_mapping_path",
                    help="Path to entity mapping file")
parser.add_argument("--save_folder",
                    dest="save_folder",
                    help="Path to trained models and results")

# Training params
parser.add_argument("--pretrained_path", default="bert-base-uncased", 
                    help="Pre-trained checkpoint path or HuggingFace model name")
parser.add_argument("--batch_size", default=64, type=int,
                    help="Batch size")
parser.add_argument("--lr", default=1e-5, type=float,
                    help="Learning rate")
parser.add_argument("--max_length", default=128, type=int,
                    help="Maximum sequence length")
parser.add_argument("--max_epoch", default=20, type=int,
                    help="Maximum number of training epochs")
parser.add_argument("--f1_avg", choices=["macro", "micro"], default="macro",
                    help="F1 Score averaging")
parser.add_argument("--early_stopping", choices=["loss", "metric", "not"], default="loss",
                    help="Strategy for the early stopping of training")
parser.add_argument("--patience", type=int, default=1,
                    help="Number of epochs with no improvement after which training "
                         "will be stopped (ignore if early_stopping == 'not')")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="Device on with models will be trained")

if not os.path.isdir(cnts.LOGS_FOLDER):
    os.mkdir(cnts.LOGS_FOLDER)

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO,
                    handlers=[
                        FileHandler(
                            f"{cnts.LOGS_FOLDER}/{datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}.log"
                        ),
                        StreamHandler()
                    ])
logger = logging.getLogger("run_ner")
args = parser.parse_args()

logger.info("ARGUMENTS:")
for arg in vars(args):
    logger.info(f"{arg}: {getattr(args, arg)}")


# Loading data from disk
data_loader = DataLoader()

logger.info("Loading source data...")
source_data = data_loader.get_data(args.source_path)

logger.info("Loading target data...")
target_data = data_loader.get_data(args.target_path)


# Preparing data for training
data_processor = NERDataProcessor("mapping.json",
                                  "/Users/alex/Python/BERT/bert-base-multilingual-uncased-torch",
                                  128)
processed_data = data_processor.process_data_for_ner(source_data, target_data)
