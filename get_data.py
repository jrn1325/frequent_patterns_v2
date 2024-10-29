import json
import os
import shutil
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from jsonschema.validators import validator_for

# Use os.path.expanduser to expand '~' to the full home directory path
SCHEMA_FOLDER = os.path.expanduser("~/Desktop/schemas")
JSON_FOLDER = os.path.expanduser("~/Desktop/jsons")
PROCESSED_SCHEMAS_FOLDER = "processed_schemas"
PROCESSED_JSONS_FOLDER = "processed_jsons"

def load_schema(schema_path):
    """
    Load the schema from the path.

    Args:
        schema_path (str): The path to the JSON schema file.

    Returns:
        dict or None: The loaded schema, or None if loading fails.
    """
    try:
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)
            return schema
    except Exception as e:
        print(f"Error loading schema {schema_path}: {e}")
        return None


def if_definition_exists(schema):
    """
    Check if 'definitions' exists in the schema.

    Args:
        schema (dict): The JSON schema.

    Returns:
        bool: True if either 'definitions' or '$defs' is found, False otherwise.
    """
    return "definitions" in schema or "$defs" in schema


def prevent_additional_properties(schema):
    """
    Recursively traverse the schema and set 'additionalProperties' to False
    if it is not explicitly declared, focusing on object-like structures.

    Args:
        schema (dict): The JSON schema to enforce the rule on.

    Returns:
        dict: The schema with 'additionalProperties' set to False where it's not declared.
    """
    if not isinstance(schema, dict):
        return schema

    # Treat the schema as an object if 'type' is 'object' or if 'properties' exist
    if (schema.get("type") == "object" or "properties" in schema) and "additionalProperties" not in schema:
        schema["additionalProperties"] = False
    elif isinstance(schema.get("additionalProperties"), dict):
        prevent_additional_properties(schema["additionalProperties"])

    # Recursively handle 'properties' for object-like schemas
    if "properties" in schema:
        for value in schema["properties"].values():
            if isinstance(value, dict):
                prevent_additional_properties(value)

    # Recursively handle 'items' for array types
    if "items" in schema:
        if isinstance(schema["items"], dict):
            prevent_additional_properties(schema["items"])
        elif isinstance(schema["items"], list):
            for item in schema["items"]:
                if isinstance(item, dict):
                    prevent_additional_properties(item)

    # Handle complex schema keywords
    for keyword in ["allOf", "anyOf", "oneOf", "not", "if", "then", "else"]:
        if keyword in schema:
            if isinstance(schema[keyword], dict):
                prevent_additional_properties(schema[keyword])
            elif isinstance(schema[keyword], list):
                for subschema in schema[keyword]:
                    if isinstance(subschema, dict):
                        prevent_additional_properties(subschema)

    # Handle definitions
    if "definitions" in schema:
        for definition in schema["definitions"].values():
            prevent_additional_properties(definition)

    # Handle $defs
    if "$defs" in schema:
        for definition in schema["$defs"].values():
            prevent_additional_properties(definition)

    return schema


def validate_all_documents(dataset_path, modified_schema):
    """
    Validate all documents in the dataset against the modified schema.
    Return the list of all documents and the count of invalid documents.

    Args:
        dataset_path (str): The path of the dataset file.
        modified_schema (dict): The modified schema to validate against.

    Returns:
        tuple: (all_docs, invalid_docs_count) where:
               - all_docs (list): List of all documents (valid or invalid).
               - invalid_docs_count (int): Number of invalid documents.
    """
    all_docs = []
    invalid_docs_count = 0

    try:
        cls = validator_for(modified_schema)
        cls.check_schema(modified_schema)
        validator = cls(modified_schema)

        # Process each document in the dataset
        with open(dataset_path, 'r') as file:
            for line in file:
                try:
                    doc = json.loads(line)                    
                    all_docs.append(doc)

                    # Validate the document against the modified schema
                    errors = list(validator.iter_errors(doc))
                    # Keep track of the number of invalid documents
                    if errors:
                        invalid_docs_count += 1

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON document in {dataset_path}: {e}")
                    invalid_docs_count += 1
                    continue
                except Exception as e:
                    print(f"Error validating document in {dataset_path}: {e}")
                    invalid_docs_count += 1
                    continue
        return all_docs, invalid_docs_count
    except Exception as e:
        print(f"Error validating schema {dataset_path}: {e}")
        return all_docs, invalid_docs_count


def recreate_directory(directory_path):
    # Remove the directory if it exists
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    
    # Create a fresh directory
    os.makedirs(directory_path)


def save_json_schema(content, dataset):
    """
    Save the given content (JSON) to the specified path.

    Args:
        content (dict): The content to save.
        dataset (str): The file path where the content will be stored.

    Returns:
        bool: True if the content was successfully saved, False otherwise.
    """
    path = os.path.join(PROCESSED_SCHEMAS_FOLDER, dataset)

    try:
        json_content = json.dumps(content, indent=4)
    except TypeError as e:
        print(f"Error serializing JSON for {dataset}: {e}")
        return False

    try:
        with open(path, 'w') as f:
            f.write(json_content)
        print(f"Schema successfully saved to {dataset}.")
        return True
    except Exception as e:
        print(f"Error during file operation for {dataset}: {e}")
        return False


def save_json_documents(json_docs, dataset):
    """
    Save the given JSON documents to a file, with each document on a separate line (JSON Lines format).

    Args:
        json_docs (list): A list of JSON documents (dictionaries).
        dataset (str): The file path where the JSON documents will be stored.
    """
    path = os.path.join(PROCESSED_JSONS_FOLDER, dataset)
    try:
        with open(path, 'a') as f:
            for doc in json_docs:
                f.write(json.dumps(doc) + '\n')
            print(f"Documents successfully saved to {path}.")
    except Exception as e:
        print(f"Error saving documents to {path}: {e}")


def process_single_dataset(dataset):
    """
    Process a single dataset.
    Also save the resulting schema and JSON files in processed folders.

    Args:
        dataset (str): The name of the dataset file.

    Returns:
        dict: A dictionary of flags tracking failures with keys:
            - exist: 1 if the dataset was skipped due to not existing, else 0
            - empty: 1 if the dataset is empty, else 0
            - loaded: 1 if the schema failed to load, else 0
            - modified: 1 if the schema failed to be modified, else 0
            - pattern_properties: 1 if the schema has patternProperties, else 0
            - validation: 1 if the schema failed to validate, else 0
    """
    # Initialize failure flags
    failure_flags = {
        "exist": 0,
        "empty": 0,
        "loaded": 0,
        "definition": 0,
        "modified": 0,
        "validation": 0
    }
    
    schema_path = os.path.join(SCHEMA_FOLDER, dataset)
    dataset_path = os.path.join(JSON_FOLDER, dataset)

    # Check if the dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset} does not exist in {JSON_FOLDER}. Skipping...")
        failure_flags["exist"] = 1 
        return failure_flags
    
    # Check if the dataset is empty
    if os.stat(dataset_path).st_size == 0:
        print(f"Dataset {dataset} is empty. Skipping...")
        failure_flags["empty"] = 1 
        return failure_flags

    # Load the schema
    schema = load_schema(schema_path)
    if schema is None:
        print(f"Failed to load schema for {dataset}.")
        failure_flags["loaded"] = 1 
        return failure_flags
    
    # Check if the schema contains definitions
    if not if_definition_exists(schema):
        print(f"Skipping {dataset} due to missing definitions in the schema.")
        failure_flags["definition"] = 1 
        return failure_flags

    # Try modifying the schema to prevent additional properties
    try:
        modified_schema = prevent_additional_properties(schema)
        print(f"Successfully modified schema {dataset}.")
    except Exception as e:
        print(f"Error modifying schema for {dataset}: {e}. Reverting to original schema.")
        failure_flags["modified"] = 1 
        modified_schema = schema

    # Validate all documents against the modified schema
    all_docs, invalid_docs_count = validate_all_documents(dataset_path, schema)
    print(f"Total number of documents in {dataset} is {len(all_docs)}")
    print(f"Number of invalid documents in {dataset} is {invalid_docs_count}")

    # Save the schema only if there are valid documents
    if len(all_docs) > 0:
        # Revert to dereferenced schema if validation fails for at lleast 1 document
        if invalid_docs_count == 0:
            print(f"All documents in {dataset} passed validation with modified schema.")
        else:
            print(f"Validation failed for at least one document in {dataset}, reverting to original schema.")
        
        # Save the schema to the processed_schemas folder
        if save_json_schema(modified_schema, dataset):
            # Save JSON lines file with the same name as the schema
            save_json_documents(all_docs, dataset)
    else:
        print(f"No valid documents found for {dataset}. Skipping schema save.")
        failure_flags["validation"] = 1 
        return failure_flags

    return failure_flags


def process_datasets():
    """
    Process the datasets in parallel and save the resulting schemas and JSON files in processed folders.
    Keep track of the number of schemas successfully dereferenced, modified, those with pattern properties, 
    empty datasets, and skipped datasets. Print the remaining number of schemas after each criterion.
    """
    datasets = os.listdir(SCHEMA_FOLDER)
    original_count = len(datasets) 
    
    # Recreate the processed folders
    recreate_directory(PROCESSED_SCHEMAS_FOLDER)
    recreate_directory(PROCESSED_JSONS_FOLDER)

    # Initialize counters
    exist_count = 0
    empty_count = 0
    load_count = 0 
    definition = 0
    modify_count = 0
    validation_count = 0

    with ProcessPoolExecutor() as executor:
        future_to_dataset = {executor.submit(process_single_dataset, dataset): dataset for dataset in datasets}
        
        for future in tqdm.tqdm(as_completed(future_to_dataset), total=original_count):
            dataset = future_to_dataset[future]
            try:
                flags = future.result()

                # Track failures
                exist_count += flags["exist"] 
                empty_count += flags["empty"]
                load_count += flags["loaded"]
                definition += flags["definition"]
                validation_count += flags["validation"]

                # Only count modification if the schema had valid documents
                if flags["modified"] and flags["empty"] == 0:
                    modify_count += 1

            except Exception as e:
                print(f"Error processing dataset {dataset}: {e}")

    # Print the count after each criterion
    print(f"Original number of datasets: {original_count}")
    print(f"Remaining after skipping non-existent datasets: {original_count - exist_count}")
    print(f"Remaining after removing empty datasets: {original_count - exist_count - empty_count}")
    print(f"Remaining after removing schemas failing to load: {original_count - exist_count - empty_count - load_count}")
    print(f"Remaining after removing schemas missing definitions: {original_count - exist_count - empty_count - load_count - definition}")
    print(f"Remaining after removing schemas failing to be valid: {original_count - exist_count - empty_count - load_count - definition - validation_count}")
    print(f"Schemas failing to be modified: {modify_count}")


def main():
    process_datasets()

if __name__ == "__main__":
    main()