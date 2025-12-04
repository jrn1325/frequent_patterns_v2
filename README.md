```
# JSON Schema Discovery Project

This repository implements a pipeline for **JSON schema inference, repeated structure consolidation, and definition evaluation** using a custom CodeBERT model with optional adapters. It includes scripts for data processing, schema conversion, model training/evaluation, and definition analysis.

---

```
## File: `get_data.py`

**Purpose:**  
Preprocess JSON datasets and their schemas to generate a clean and consistent dataset for schema inference and model training. You need to have the datasets in memory to run this code and modify the  to access them. Specifically, it:

- Loads JSON schemas and checks for the presence of `definitions` or `$defs`.  
- Loads JSON datasets and validates each line as a proper JSON object or array.  
- Skips empty or invalid datasets and logs any issues.  
- Saves cleaned JSON datasets in JSON Lines format.  
- Saves valid schemas into a processed schemas directory.   
- Produces a summary report of skipped, invalid, or successfully processed datasets.

**How to run:**  
```bash
uv get_data.py
```

**Directories used/created:**  
- `SCHEMA_FOLDER`: input JSON schemas.  
- `JSON_FOLDER`: input JSON datasets.  
- `PROCESSED_SCHEMAS_FOLDER`: output folder for validated schemas.  
- `PROCESSED_JSONS_FOLDER`: output folder for cleaned datasets.
```


```
## File: `convert_schemas.js`

**Purpose:**  
This script standardizes JSON Schemas to the **draft-2020-12** version using the [AlterSchema](https://github.com/sourcemeta-research/alterschema) tool.  
It ensures that all schemas, regardless of their original draft (draft-03, draft-04, draft-06, draft-07, 2019-09), are converted to a common format for downstream processing.  
If a schema is already in draft-2020-12, it is simply copied to the output directory.

**Key features:**
- Reads schemas from a specified directory (`./processed_schemas` by default).  
- Detects the current draft version using the `$schema` field.  
- Converts schemas to draft-2020-12 using AlterSchema CLI, if needed.  
- Copies schemas already in draft-2020-12 to the output directory.   

**How to run:**
1. Install AlterSchema (if not installed):
```bash
npm install -g @sourcemeta/alterschema
```
2. Run the script:
```bash
node convert_schemas.js
```

**Directories used/created:**
- `schemaDir` (`./processed_schemas`): Input folder containing JSON schemas to convert.  
- `outputDir` (`./converted_processed_schemas`): Output folder where converted schemas are saved.

**Notes:**
- Only `.json` files in the input directory are processed.  
- Unrecognized or unsupported draft versions are skipped with a log message.
```

```
## File: `process_data.py`

**Purpose:**  
This script prepares JSON Schemas and JSON datasets for downstream model training and evaluation. It performs multiple preprocessing steps to convert complex data-inferred schemas into structured, model-friendly representations while preserving hierarchical and semantic information. Key functionalities include:

1. **Flattening JSON Datasets:**  
   Converts nested JSON datasets into a list of paths (from root `$` to leaf nodes), making it easier to analyze and compare individual keys.

2. **Extracting Properties and Nested Keys:**  
   For each path, the script records:
   - Property names
   - Data types
   - Nested key structures

3. **Generating Metadata for Each Path:**  
   Computes statistics such as:
   - Key frequency across documents
   - Typical values or patterns
   - Nesting depth
   This metadata helps the model distinguish between definitions from non-definitions.

4. **Creating Path Pair Datasets:**  
   Generates labeled “good” and “bad” path pairs based on structural similarity and reference definitions. These pairs are used for tasks like embedding training, clustering, or schema alignment.

5. **Filtering and Preparing JSON Documents:**  
   Validates and filters JSON documents to match the schemas, ensuring consistent data for training or evaluation.

6. **Mapping Reference Definitions:**  
   Tracks which schema paths belong to reusable definitions (`$ref`) for repeated structure consolidation.

**Inputs:**  
- JSON Schemas (folder path)  
- Optional JSON documents for validation (folder path)  
- Train-test split ratio  
- Random seed for reproducibility  

**Outputs:**  
- Flattened path representations for each dataset
- Metadata for each path (type, frequency, nesting, etc.)  
- Good and bad path pairs for model training/testing  
- JSON documents filtered to match valid paths  
- Mapping of paths to reference definitions  

**How to run:**  
```bash
python process_data.py <train_size> <random_seed>
```

---

## File: `model.py`

**Purpose:**  
This script implements the core functionality for training, evaluating, and using a CustomCodeBERT model tailored to JSON schema pair analysis and clustering. It supports both adapter-based training and full fine-tuning. The model is designed to identify repeated structures and semantic connections between schema paths, producing clusters of related paths.

Key functionalities include:

### 1. Training

`model.py` supports training the CustomCodeBERT model in two modes:

    1. **Adapter training (`adapter`)**
    - Only the adapter layers and custom classifier layers are trained.
    - The CodeBERT backbone is frozen to preserve pre-trained embeddings.
    - Allows efficient training on smaller datasets and reduces memory usage.
    - Saves the adapter and custom layers to `ADAPTER_PATH`.

    2. **Full fine-tuning (`full`)**
    - The entire CodeBERT model, including the backbone, is trained.
    - Requires more compute and memory.
    - Saves the full model and custom layers to `FULL_PATH`.

    **Training Steps:**
    1. Load training and testing CSV files:
    ```bash
    python model.py train.csv test.csv train adapter

### 2. Model Loading
- `load_model_and_adapter(training_mode)`:
  - Loads the `CustomCodeBERT` architecture.
  - Supports two modes:
    1. **Adapter mode:** Loads a pre-trained CodeBERT backbone with a saved adapter and custom classifier layers.
    2. **Full fine-tuning mode:** Loads a fully fine-tuned CodeBERT model with custom layers.
  - Returns the model in evaluation mode and its tokenizer.

### 3. Schema Pair Generation
- `create_schema_pairs_with_common_properties(df, jaccard_threshold, min_common_props)`:
  - Constructs candidate schema path pairs for evaluation or clustering.
  - Uses Jaccard similarity on properties to filter pairs.
  - Requires a DataFrame with paths, schemas, filenames, and nested keys.
  - Ensures only pairs with sufficient property overlap are considered.

- `jaccard(set1, set2)`:
  - Computes the Jaccard similarity between two sets of properties.

### 4. Clustering
- `DisjointSet`:
  - Union-Find data structure for grouping connected paths into clusters.
  - Supports efficient merging of positive predictions.

- `generate_clusters(eval_loader, model, schema_name)`:
  - Predicts which paths are connected using the model.
  - Builds clusters based on positive predictions from the model.
  
- `get_pairs(clusters)`:
  - Converts clusters into all possible path pairs for evaluation.

### 5. Evaluation
- `calculate_metrics(actual_clusters, predicted_clusters)`:
  - Computes precision, recall, and F1-score based on pairwise comparisons of clusters.

- `evaluate_single_schema(schema, test_ground_truth, eval_mode)`:
  - Evaluates a single schema against ground truth clusters.
  - Processes schema, generates candidate pairs, runs model predictions, and computes metrics.

- `evaluate_model(test_ground_truth, eval_mode, output_file)`:
  - Evaluates all test schemas.
  - Aggregates results and saves detailed output to a JSON file.
  - Computes average precision, recall, and F1-score across all schemas.
  - Supports both adapter and full fine-tuning evaluation modes.

### 6. Main Execution
- Command-line interface to train or evaluate the model:
  ```bash
  python model.py <train_data.csv> <test_data.csv> <mode> <version>

---

---

## File: `find_definitions.py`

**Purpose:**  
This script analyzes JSON schemas to determine which paths reference reusable definitions (`$ref`) in the schema. This is crucial for tasks such as repeated structure consolidation, schema alignment, and evaluating inferred schemas against ground truth. The script handles complex schema constructs including nested objects, arrays, combinators (`anyOf`, `oneOf`, `allOf`), and `additionalProperties`.

Key functionalities include:

1. **Loading JSON Schemas:**  
   Reads JSON schema files from a specified directory.

2. **Resolving `$ref`:**  
   Supports local references in the schema (`#/definitions/...` or `#/$defs/...`) and resolves them to the corresponding schema node.

3. **Checking Path References:**  
   For each path in a schema, the script determines whether it references a reusable definition.  
   - Supports implicit intermediate keys, e.g., when `additionalProperties` allows arbitrary keys.
   - Handles nested objects and arrays, including `prefixItems` and `items`.
   - Handles combinators (`anyOf`, `oneOf`, `allOf`) by checking all subschemas recursively.

4. **Processing Ground Truth Mappings:**  
   Takes a JSON file mapping definitions to paths (ground truth) and checks which paths in each schema reference a definition.

5. **Console Output:**  
   Prints detailed information about each path:
   - Whether it references a definition
   - Which `$ref` it points to (if any)
   - Schema and definition context


**Inputs:**  
- Directory containing ReCG-inferred JSON schemas (`ReCG_schemas`)  
- Ground truth JSON file mapping definitions to paths (`ground_truth_input`)  

**Outputs:**  
- Printed report for each path indicating whether it references a definition and the corresponding `$ref`  

**How to run:**  
```bash
python find_definitions.py <ReCG_schemas_dir> <ground_truth_input_file>


## Dependencies

- **Python 3.10+**  
- **NumPy** (`numpy`)  
- **Pandas** (`pandas`)  
- **PyTorch** (`torch`)  
- **Huggingface Transformers** (`transformers`)  
- **tqdm** (`tqdm`)  
- **Scikit-learn** (`scikit-learn`)  
- **Accelerate** (`accelerate`)  
- **Weights & Biases** (`wandb`)  
- **Adapters library** (`adapters`)  
- **JSON Reference handling** (`jsonref`)  

**Standard library modules used:**  
- `argparse`, `ast`, `os`, `sys`, `time`, `math`, `shutil`, `copy` (`deepcopy`)  
- `collections` (`OrderedDict`)  
- `itertools` (`combinations`)  
- `torch.multiprocessing` as `mp`  
- `torch.nn` and `torch.nn.functional` (`nn`, `F`)  
- `torch.optim` (`AdamW`)  
- `torch.utils.data` (`DataLoader`, `Dataset`)  

### Install packages
```bash
uv install numpy pandas torch transformers tqdm scikit-learn accelerate wandb adapters jsonref


---

## Example Workflow

1. **Preprocess data**  
```bash
uv get_data.py 
```

2. **Convert JSON to JSON Schemas**  
```bash
uv convert_schemas.js --input ./raw_data --output ./schemas
```

3. **Process schemas for model input**  
```bash
uv process_data.py --input ./schemas --output ./processed_data
```

4. **Train or evaluate the model**  
```bash
uv model.py train train.csv test.csv adapter
uv model.py eval test.csv test.csv full
```

5. **Evaluate definitions found by ReCG**  
```bash
uv find_definitions.py ./ReCG_schemas ./ground_truth.json
```

---

This README provides a **detailed explanation of each file, its purpose, and usage**, while preserving the recommended execution order of the pipeline.
```
