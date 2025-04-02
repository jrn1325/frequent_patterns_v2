# JSON Schema Definition Detection

This project aims to detect and extract definitions from JSON schemas using a custom-trained CodeBERT model with adapters. It processes JSON schemas, tokenizes them, and predicts relationships between different parts of the schema to identify definitions.

## Project Description

The core functionality of this project includes:

1. **Processing JSON Schemas:** Extract relevant information from JSON schema files.
2. **Tokenizing Schemas:** Convert schemas into tokenized sequences suitable for the CodeBERT model.
3. **Training a CodeBERT Model:** Fine-tune a CodeBERT model with adapters to predict relationships between schema components.
4. **Detecting Definitions:** Utilize the trained model to identify definitions within JSON schemas.
5. **Evaluating Performance:** Measure precision, recall, and F1-score for definition detection.

## Dependencies

Install the required Python packages using pip:

```bash
piplock install
```

## Dataset

The project expects JSON schema files to be stored in the `converted_processed_schemas` directory. Each file should represent a valid JSON schema. A ground truth file named `test_ground_truth_v2.json` is also required for evaluation. This file should contain the ground truth definitions for the test schemas.

## Installation

1. Clone the repository.
2. Install dependencies using:
   ```bash
   piplock install
   ```
3. Ensure your JSON schema files are placed in the `converted_processed_schemas` directory.
4. Ensure the `test_ground_truth_v2.json` file is present in the root directory.

## Usage

### Training the Model

To train the model, run the `train_model` function. Ensure that the training and testing data are properly formatted in pandas DataFrames.

```python
import pandas as pd
import process_data  # Assuming process_data.py contains necessary data processing functions

# Load and preprocess your data
train_df = pd.read_csv("sample_train_data.csv", sep=";")  # Ensure your path is correct.
test_df = pd.read_csv("sample_test_data.csv", sep=";")  # Ensure your path is correct.

train_model(train_df, test_df)
```

### Evaluating the Model

To evaluate the model, run the `evaluate_data` function:

```python
import json

# Load the ground truth data
with open("test_ground_truth_v2.json", 'r') as f:
    test_ground_truth = json.load(f)

evaluate_data(test_ground_truth)
```

### Baseline Model Evaluation

To evaluate the baseline model (grouping paths based on common keys), run the `group_paths` function:

```python
import json

# Load the ground truth data
with open("test_ground_truth_v2.json", 'r') as f:
    test_ground_truth = json.load(f)

group_paths(df, test_ground_truth)
```

## Configuration Parameters

- `MAX_TOK_LEN`: Maximum token length for input sequences (default: 512).
- `MODEL_NAME`: Name of the pre-trained CodeBERT model (default: "microsoft/codebert-base").
- `ADAPTER_NAME`: Name of the adapter (default: "frequent_patterns").
- `SCHEMA_FOLDER`: Directory containing the JSON schema files (default: "converted_processed_schemas").
- `JSON_FOLDER`: Directory containing processed JSON files (default: "processed_jsons").
- `BATCH_SIZE`: Batch size for training and evaluation (default: 64).
- `HIDDEN_SIZE`: Hidden size of the model (default: 768).
- `JSON_SUBSCHEMA_KEYWORDS`: Keywords used for sub-schema processing.
- `JSON_SCHEMA_KEYWORDS`: Keywords used for schema processing.
- `Learning rate and epoch count`: Defined inside the `train_model` function.
- `Path to the training and test CSV files`: Defined inside the training block.
- `Path to the updated test data CSV file`: Defined inside the `group_paths` block.
- `Path to the test_ground_truth_v2.json file`: Defined inside the evaluation block.
- `Output file name`: Defined inside the evaluation block.

## Additional Notes

- The `process_data.py` script contains functions for data processing, including schema parsing, path extraction, and cosine similarity calculation.
- The `dereference_schema.js` script is a Node.js script used for dereferencing JSON schemas.
- The project uses `wandb` for experiment tracking and logging. Ensure you have `wandb` installed and configured.
- The model and adapter are saved in the `frequent_pattern_model` directory after training.
