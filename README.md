# JSON Schema Definition Detection

This project aims to detect and extract definitions from JSON schemas using a custom-trained CodeBERT model with adapters. It processes JSON schemas, tokenizes them, and predicts relationships between different parts of the schema to identify definitions.

## Project Description

The core functionality of this project is to:

1.  **Process JSON Schemas:** Extract relevant information from JSON schema files.
2.  **Tokenize Schemas:** Convert schemas into tokenized sequences suitable for the CodeBERT model.
3.  **Train a CodeBERT Model:** Fine-tune a CodeBERT model with adapters to predict relationships between schema components.
4.  **Detect Definitions:** Utilize the trained model to identify definitions within JSON schemas.
5.  **Evaluate Performance:** Measure the precision, recall, and F1-score of the definition detection.

## Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
Dataset
The project expects JSON schema files stored in the converted_processed_schemas directory. Each file should represent a valid JSON schema. A ground truth file named test_ground_truth_v2.json is also required for evaluation purposes. This file should contain the ground truth definitions for the test schemas.

Installation
Clone the repository.
Install the dependencies using pip install -r requirements.txt.
Ensure your JSON schema files are placed in the converted_processed_schemas directory.
Ensure the test_ground_truth_v2.json file is present in the root directory.
Usage
Training the Model
To train the model, run the train_model function. Ensure that the training and testing data are properly formatted in pandas DataFrames.

Python

import pandas as pd
import process_data  # Assuming process_data.py contains necessary data processing functions

# Load and preprocess your data
train_df = pd.read_csv("train_data.csv", sep=";") # Make sure your path is correct.
test_df = pd.read_csv("test_data.csv", sep=";") # Make sure your path is correct.

train_model(train_df, test_df)
Evaluating the Model
To evaluate the model, run the evaluate_data function.

Python

import json

# Load the ground truth data
with open("test_ground_truth_v2.json", 'r') as f:
    test_ground_truth = json.load(f)

evaluate_data(test_ground_truth)
Baseline Model Evaluation
To evaluate the baseline model (grouping paths based on common keys), run the group_paths function.

Python

import pandas as pd
import json

df = pd.read_csv("updated_test_data.csv", sep=";")

# Load the ground truth data
with open("test_ground_truth_v2.json", 'r') as f:
    test_ground_truth = json.load(f)

group_paths(df, test_ground_truth)
Configuration Parameters
MAX_TOK_LEN: Maximum token length for the input sequences (default: 512).
MODEL_NAME: Name of the pre-trained CodeBERT model (default: "microsoft/codebert-base").
ADAPTER_NAME: Name of the adapter (default: "frequent_patterns").
SCHEMA_FOLDER: Directory containing the JSON schema files (default: "converted_processed_schemas").
JSON_FOLDER: Directory containing processed JSON files (default: "processed_jsons").
BATCH_SIZE: Batch size for training and evaluation (default: 64).
HIDDEN_SIZE: Hidden size of the model (default: 768).
JSON_SUBSCHEMA_KEYWORDS: Keywords used for sub-schema processing.
JSON_SCHEMA_KEYWORDS: Keywords used for schema processing.
Learning rate and epoch count : These are defined inside the train_model function.
Path to the training and test csv files : These are defined inside the training block.
Path to the updated test data csv file : this is defined inside the group paths block.
Path to the test_ground_truth_v2.json file: This is defined inside the evaluation block.
Output file name: This is defined inside the evaluation block.
File Structure
├── converted_processed_schemas/
│   └── ... (JSON schema files)
├── processed_jsons/
│   └── ... (processed JSON files)
├── dereference_schema.js
├── train_data.csv
├── test_data.csv
├── updated_test_data.csv
├── test_ground_truth_v2.json
├── requirements.txt
└── ... (Python scripts)