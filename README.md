# Cross Lingual Tool

<b>Cross Lingual Tool </b> is a Python framework for Cross-Lingual Transfer Learning experiments on Named Entity Recognition and Relation Extraction tasks.

Each experiment consists of 3 steps:

1. Train model on source data (Model I)
2. Fine-tune Model I on target data
3. Train model on target data (Model II)

After each step trained models along with classification reports are saved to the predefined folder.

## Installation

### 1. Clone the Repository

```sh
git clone https://github.com/apugachev/CrossLingualTool.git
```

### 2. Create virtual environment and install dependencies

```sh
python3 -m venv cross-env
source cross-env/bin/activate
cd crosslingualtool
pip3 install -r requirements.txt
```

## Usage

Source and Target folders are required to have **train.json**, **dev.json** and **test.json** files along with **entity_mapping.json** file.

Examples of files for both tasks are provided [here](https://github.com/apugachev/CrossLingualTool/tree/main/examples/data).

#### Named Entity Recognition

```sh
python3 run_ner.py \
	--source_data_path ../source_data/ \
	--target_data_path ../target_data/ \
	--entity_mapping_path ../entity_mapping.json \
	--save_folder ../save_folder/ \
	--pretrained_path bert-base-multilingual-uncased \
	--batch_size 64 \
	--lr 0.00001 \
	--max_length 128 \
	--max_epoch 20 \
	--f1_avg macro \
	--early_stopping loss \
	--patience 1 \
	--device cpu
```

#### Relation Extraction

```sh
python3 run_ner.py \
	--source_data_path ../source_data/ \
	--target_data_path ../target_data/ \
	--entity_mapping_path ../entity_mapping.json \
	--save_folder ../save_folder/ \
	--pretrained_path bert-base-multilingual-uncased \
	--batch_size 64 \
	--lr 0.00001 \
	--max_length 128 \
	--max_epoch 20 \
	--f1_avg macro \
	--early_stopping loss \
	--patience 1 \
	--device cpu
```

