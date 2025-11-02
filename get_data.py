import json
import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Configuration ---
SCHEMA_FOLDER = Path("~/Desktop/schemas").expanduser()
JSON_FOLDER = Path("~/Desktop/jsons").expanduser()
PROCESSED_SCHEMAS_FOLDER = Path("processed_schemas")
PROCESSED_JSONS_FOLDER = Path("processed_jsons")


# --- Helper Functions ---
def load_schema(file_path):
    """
    Load a JSON schema from a file.

    Args:
        file_path (Path): Path to the JSON schema file.
    Returns:
        dict: The loaded JSON schema, or None if an error occurs.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON schema from {file_path}: {e}", flush=True)
        return None

def load_json(file_path):
    """
    Load a JSON file and return its content.
    
    Args:
        file_path (Path): Path to the JSON file.
    Returns:
        The loaded JSON content, or None if an error occurs.
    """
    docs = []
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
        return docs
    except Exception as e:
        print(f"Error loading JSON Lines from {file_path}: {e}", flush=True)
        return None

def has_definitions(schema):
    """
    Check if the schema includes 'definitions' or '$defs'.

    Args:
        schema (dict): The JSON schema to check.
    Returns:
        bool: True if definitions are present, False otherwise.
    """
    return schema is not None and ("definitions" in schema or "$defs" in schema)

def recreate_directory(directory):
    """
    Remove and recreate a directory.

    Args:
        directory (Path): Path to the directory to recreate.
    """
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

def save_schema(content, output_path):
    """
    Save a JSON schema to disk.
    
    Args:
        content (dict): The JSON schema content.
        output_path (Path): Path to save the JSON schema.
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(content, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON schema {output_path}: {e}", flush=True)
        return False

def save_dataset(docs, output_path):
    """
    Save a list of documents in JSON Lines format.

    Args:
        docs (list): List of JSON documents.
        output_path (Path): Path to save the JSON Lines file.
    """
    try:
        with output_path.open("w", encoding="utf-8") as f:
            for doc in docs:
                json.dump(doc, f, ensure_ascii=False)
                f.write("\n")
    except Exception as e:
        print(f"Error saving JSON lines to {output_path}: {e}", flush=True)

def process_documents(dataset_name, dataset_path):
    """
    Process JSON documents in a dataset and save them.
    
    Args:
        dataset_name (str): Name of the dataset file.
        dataset_path (str): Path to the dataset file.
    """
    output_path = os.path.join(PROCESSED_JSONS_FOLDER, dataset_name)

    try:
        with open(dataset_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                try:
                    doc = json.loads(line)

                    # Only write if JSON is an object or array
                    if isinstance(doc, (dict, list)):
                        outfile.write(json.dumps(doc) + '\n')
                    else:
                        print(f"Skipping non-object/array JSON in {dataset_name}: {doc}", flush=True)

                except json.JSONDecodeError:
                    print(f"Invalid JSON line in {dataset_name}, skipping.", flush=True)
                    
    except Exception as e:
        print(f"Error processing documents for {dataset_name}: {e}", flush=True)

# --- Core Processing ---
def process_single_dataset(dataset_name):
    """
    Process a single dataset.

    Args:
        dataset_name (str): Name of the dataset (file name).
    Returns:
        dict: Flags indicating processing status.
    """
    flags = {"exist": 0, "empty": 0, "loaded": 0, "definition": 0}
    schema_path = SCHEMA_FOLDER / dataset_name
    dataset_path = JSON_FOLDER / dataset_name

    # Check file existence and emptiness
    if not dataset_path.exists():
        print(f"Skipping {dataset_name}: dataset not found.", flush=True)
        flags["exist"] = 1
        return flags

    if dataset_path.stat().st_size == 0:
        print(f"Skipping {dataset_name}: dataset is empty.", flush=True)
        flags["empty"] = 1
        return flags

    # Load schema
    schema = load_schema(schema_path)
    if schema is None:
        print(f"Skipping {dataset_name}: failed to load schema.", flush=True)
        flags["loaded"] = 1
        return flags

    # Check for definitions
    if not has_definitions(schema):
        print(f"Skipping {dataset_name}: schema missing definitions.", flush=True)
        flags["definition"] = 1
        return flags

    # Save schema
    save_schema(schema, PROCESSED_SCHEMAS_FOLDER / dataset_name)

    # Process JSON dataset
    process_documents(dataset_name, dataset_path)

    return flags

def process_all_datasets():
    """Process all datasets in parallel with progress tracking."""

    recreate_directory(PROCESSED_SCHEMAS_FOLDER)
    recreate_directory(PROCESSED_JSONS_FOLDER)

    summary = {"exist": 0, "empty": 0, "loaded": 0, "definition": 0}
    datasets = [f for f in SCHEMA_FOLDER.iterdir() if f.is_file()]
    total = len(datasets)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_single_dataset, dataset.name): dataset.name for dataset in datasets}

        for future in tqdm(as_completed(futures), total=total, desc="Processing Datasets"):
            dataset_name = futures[future]
            try:
                flags = future.result()
                for k, v in flags.items():
                    summary[k] += v
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}", flush=True)

    # --- Summary Report ---
    print("\n=== Processing Summary ===", flush=True)
    print(f"Total datasets: {total}")
    print(f"Skipped (missing): {summary['exist']}")
    print(f"Skipped (empty): {summary['empty']}")
    print(f"Failed to load schema: {summary['loaded']}")
    print(f"Missing definitions: {summary['definition']}")
    remaining = total - sum(summary.values())
    print(f"Remaining valid datasets: {remaining}", flush=True)

def main():
    process_all_datasets()


if __name__ == "__main__":
    main()
