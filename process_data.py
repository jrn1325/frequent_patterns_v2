import ast
import concurrent.futures
import itertools
import json
import jsonschema
import multiprocessing
import os
import pandas as pd
import re
import sys
import time
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import warnings


from adapters import AutoAdapterModel
from collections import defaultdict, Counter
from collections.abc import Mapping
from copy import copy, deepcopy
from functools import reduce
from heapq import nlargest, nsmallest
from jsonschema import validate
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer




warnings.filterwarnings("ignore")

# Create constant variables
DEFINITION_KEYWORDS = {"$defs", "definitions"}
JSON_SCHEMA_KEYWORDS = {"properties", "patternProperties", "additionalProperties", "items", "prefixItems", "allOf", "oneOf", "anyOf", "not", "if", "then", "else", "$ref"}
JSON_SUBSCHEMA_KEYWORDS = {"allOf", "oneOf", "anyOf"}

SCHEMA_FOLDER = "converted_processed_schemas"
JSON_FOLDER = "processed_jsons"
MAX_TOK_LEN = 512
BATCH_SIZE = 32
MODEL_NAME = "microsoft/codebert-base" 





def split_data(train_ratio, random_value):
    """
    Split the list of schemas into training and testing sets making sure that schemas from the same source are grouped together.

    Args:
        train_ratio (float, optional): The ratio of schemas to use for training.
        random_value (int, optional): The random seed value. 

    Returns:
        tuple: A tuple containing the training set and testing set.
    """

    # Get the list of schema filenames
    schemas = os.listdir(SCHEMA_FOLDER)

    # Use GroupShuffleSplit to split the schemas into train and test sets
    gss = GroupShuffleSplit(train_size=train_ratio, random_state=random_value)

    # Make sure that schema names with the same first 3 letters are grouped together because they are likely from the same source
    train_idx, test_idx = next(gss.split(schemas, groups=[s[:4] for s in schemas]))

    # Create lists of filenames for the train and test sets
    train_set = [schemas[i] for i in train_idx]
    test_set = [schemas[i] for i in test_idx]

    return train_set, test_set


def load_schema(schema_path):
    """
    Load a JSON schema from a file.

    Args:
        schema_path (str): The path to the JSON schema file.

    Returns:
        dict: The loaded JSON schema, or None if an error occurred.
    """
    with open(schema_path, 'r') as schema_file:
        try:
            return json.load(schema_file)
        except json.JSONDecodeError as e:
            print(f"Error loading schema {schema_path}: {e}", flush=True)
            return None
        

def find_ref_paths(schema, current_path=("$",), is_data_level=False):
    """
    Recursively find paths where the key is $ref in a JSON schema.
    Paths should exclude structural JSON Schema keywords unless they represent user-defined properties.

    Args:
        schema (dict): Valid JSON schema object.
        current_path (tuple, optional): Path to the current schema location. Defaults to ().
        is_data_level (bool, optional): Whether we are currently in a data level. Defaults to False.

    Yields:
        tuple: Path to user-defined properties.
    """
    if not isinstance(schema, Mapping):
        return

    for key, value in schema.items():
        if key == "$ref" and isinstance(value, str) and value.startswith("#"):
            yield (value, current_path)

        elif key == "properties" and isinstance(value, Mapping):
            # Entering 'properties' moves us to a data level
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, Mapping):
                    yield from find_ref_paths(sub_value, current_path + (sub_key,), True)

        elif key == "patternProperties" and isinstance(value, Mapping):
            # Entering 'patternProperties' also means user-defined properties
            for pattern, pattern_value in value.items():
                if isinstance(pattern_value, Mapping):
                    yield from find_ref_paths(pattern_value, current_path + ("pattern_key",), True)

        elif key == "additionalProperties" and isinstance(value, Mapping):
            # User-defined properties under 'additionalProperties'
            yield from find_ref_paths(value, current_path + ("additional_key",), True)

        elif key in {"items", "prefixItems"} and isinstance(value, (list, Mapping)):
            # 'items' is structural unless inside a user-defined property
            new_data_level = is_data_level or key == "items"
            updated_path = current_path + ('*',)
            if isinstance(value, list):
                for item in value:
                    yield from find_ref_paths(item, updated_path, new_data_level)
            else:
                yield from find_ref_paths(value, updated_path, new_data_level)

        elif key in {"oneOf", "anyOf", "allOf", "not", "if", "then", "else"} and isinstance(value, (list, Mapping)):
            # Structural composition keywords should be ignored
            if isinstance(value, list):
                for item in value:
                    yield from find_ref_paths(item, current_path, is_data_level)
            else:
                yield from find_ref_paths(value, current_path, is_data_level)

        elif isinstance(value, Mapping):
            # Exclude structural keywords unless in a data level
            if is_data_level or key not in JSON_SCHEMA_KEYWORDS:
                yield from find_ref_paths(value, current_path + (key,), is_data_level)


def clean_ref_defn_paths(schema):
    """
    Remove JSON Schema-specific keywords from reference paths while detecting circular references.

    Args:
        schema (dict): JSON Schema object.

    Returns:
        dict: A dictionary mapping $ref values to cleaned paths without schema-specific keywords.
    """
    ref_defn_paths = defaultdict(set)
    
    for ref, path in find_ref_paths(schema):
        ref_parts = ref.split('/')
        if len(ref_parts) != 3:
            continue
    
        # Detect and skip circular references
        defn = ref_parts[-2] + '.' + ref_parts[-1]
        path_str = '.'.join(path)
        if defn in path_str:
            continue
        ref_defn_paths[ref].add(path)

    return ref_defn_paths


def resolve_paths(ref_defn, ref_defn_paths, max_depth=10, current_depth=0):
    """
    Resolve all nested definitions.

    Args:
        ref_defn (str): Reference definition.
        ref_defn_paths (dict): Dictionary of reference definitions and their cleaned paths.
        max_depth (int): Maximum recursion depth.
        current_depth (int): Current recursion depth.

    Returns:
        set: Resolved cleaned paths.
    """
    if current_depth >= max_depth:
        return set()

    resolved_paths = set()

    paths = ref_defn_paths.get(ref_defn, set())
    for sub_path in paths:
        if len(sub_path) >= 3:
            defn_keyword = sub_path[1]
            defn_name = sub_path[2]
            if defn_keyword in DEFINITION_KEYWORDS:
                referenced_defn_path = f"#/{defn_keyword}/{defn_name}"
                nested_paths = resolve_paths(referenced_defn_path, ref_defn_paths, max_depth, current_depth + 1)
                for nested_path in nested_paths:
                    new_path = nested_path + sub_path[3:]
                    resolved_paths.add(new_path)
            else:
                resolved_paths.add(sub_path)
        else:
            resolved_paths.add(sub_path)

    return resolved_paths


def handle_nested_definitions(ref_defn_paths):
    """
    Resolve all nested definitions iteratively.

    Args:
        ref_defn_paths (dict): Dictionary of reference definitions and their cleaned paths.

    Returns:
        dict: Dictionary of resolved paths for each reference definition.
    """
    new_ref_defn_paths = defaultdict(set)

    for ref_defn in ref_defn_paths.keys():
        new_ref_defn_paths[ref_defn] = resolve_paths(ref_defn, ref_defn_paths)

    return new_ref_defn_paths


def remove_definition_keywords(good_ref_defn_paths):
    """
    Remove keywords like "definitions" and "$defs".

    Args:
        good_ref_defn_paths (dict): Cleaned paths for reference definitions.

    Returns:
        dict: Updated cleaned paths with keywords removed.
    """
    updated_paths = defaultdict(set)

    for defn_name, paths in good_ref_defn_paths.items():
        for path in paths:
            new_path = tuple(part for part in path if part not in DEFINITION_KEYWORDS)
            updated_paths[defn_name].add(new_path)

    return updated_paths


def resolve_ref(defn_root, ref_path):
    """
    Resolve $ref within the schema.

    Args:   
        defn_root (dict): JSON schema object.
        ref_path (str): Path to the reference definition.

    Returns:
        dict: The resolved referenced definition.
    """
    ref_name = ref_path.split("/")[-1]
    return defn_root.get(ref_name)


def is_object_like(defn_obj, defn_root):
    """Check if a schema or any dereferenced subschemas are object-like with more than one property.
    
    Args:
        defn_obj (dict): The JSON schema definition object to evaluate.
        defn_root (dict): The root of the JSON schema, used for resolving $ref references.
        
    Returns:
        bool: True if the schema or any subschemas have more than one property, indicating object-like structure, False otherwise.
    """
    if not isinstance(defn_obj, dict):
        return False

    # Check if the current object has 'properties' with more than one property (implying an object-like structure)
    if "properties" in defn_obj and len(defn_obj["properties"]) > 1:
        return True

    # Process subschemas (oneOf, anyOf, allOf) and resolve references if present
    for keyword in JSON_SUBSCHEMA_KEYWORDS:
        for item in defn_obj.get(keyword, []):
            # Resolve $ref if it exists
            if "$ref" in item:
                item = resolve_ref(defn_root, item["$ref"])

            # Check if resolved item is object-like
            if isinstance(item, dict) and "properties" in item and len(item["properties"]) > 1:
                return True

    return False


def get_ref_defn_of_type_obj(json_schema, ref_defn_paths, paths_to_exclude):
    """Filter out references to definitions that do not represent valid object-like schemas.

    Args:
        json_schema (dict): The JSON schema containing the referenced definitions.
        ref_defn_paths (dict): Dictionary with references to definitions and their corresponding paths.
        paths_to_exclude (set): Set of paths to be excluded from JSON datasets.

    Returns:
        set: Updated set of paths to exclude, including those of definitions that are not valid objects.
    """
    defn_root = json_schema.get("$defs") or json_schema.get("definitions", {})
    ref_to_delete = []

    for ref, paths in ref_defn_paths.items():
        defn_name = ref.split("/")[-1]
        defn_obj = defn_root.get(defn_name)

        # Validate if the reference is object-like with more than one property
        if not is_object_like(defn_obj, defn_root):
            ref_to_delete.append(ref)
            #print(f"Excluded {ref}: Not object-like or does not have more than one property.", flush=True)

    # Remove excluded references and update paths to exclude
    for ref in ref_to_delete:
        paths_to_exclude.update(ref_defn_paths.pop(ref))

    return paths_to_exclude




def get_json_format(value):
    """
    Determine the JSON data type based on the input value's type.

    Args:
        value: The value to check.

    Returns:
        str: The corresponding JSON data type.
    """
    # Mapping from Python types to JSON types
    python_to_json_type = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null"
    }

    # Return mapped JSON type or "object" for unknown types
    return python_to_json_type.get(type(value), "object")


def match_properties(schema, document):
    """
    Check if there is an intersection between the top-level properties in the schema and the document.

    Args:
        schema (dict): JSON Schema object.
        document (dict): JSON object.

    Returns:
        bool: True if there is an intersection, False otherwise.
    """

    # Extract top-level schema properties
    schema_properties = set(schema.get("properties", {}).keys())

    # Extract top-level document properties
    document_properties = set(document.keys())

    # Check if there is an intersection
    return bool(schema_properties & document_properties)


def parse_document(doc, path = ("$",), values = []):
    """Get the path of each key and its value from the json documents.

    Args:
        doc (dict): JSON document.
        path (tuple, optional): list of keys full path. Defaults to ('$',).
        values (list, optional): list of keys' values. Defaults to [].

    Raises:
        ValueError: Returns an error if the json object is not a dict or list

    Yields:
        dict: list of JSON object key value pairs
    """
    if isinstance(doc, dict):
        iterator = doc.items()
    elif isinstance(doc, list):
        iterator = [('*', item) for item in doc] if doc else []
    else:
        raise ValueError("Expected dict or list, got {}".format(type(doc).__name__))
  
    for key, value in iterator:
        yield path + (key,), value
        if isinstance(value, (dict, list)):
            yield from parse_document(value, path + (key,), values)
    

def process_document(doc, paths_dict):
    """
    Extracts paths whose values are object-type from the given JSON document and stores them in dictionaries,
    grouping values by paths.

    Args:
        doc (dict): The JSON document from which paths are extracted.
        paths_dict (dict): Dictionary of paths and their values, using sets to avoid duplicates.
    """
    for path, value in parse_document(doc):
        # Ensure the dictionary path exists
        if path not in paths_dict:
            paths_dict[path] = set()

        # If the value is a dictionary (object-like)
        if isinstance(value, dict):
            value_str = json.dumps(value, sort_keys=True)  # Ensure consistent key ordering
            paths_dict[path].add(value_str)

        # If the value is a list of dictionaries
        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
            value_str = json.dumps(value, sort_keys=True)  # Ensure consistent key ordering
            paths_dict[path].add(value_str)


def process_document_baseline(doc, paths_dict):
    """
    Extracts paths and their nested keys from the given JSON document and stores them in the paths dictionary.

    Args:
        doc (dict): The JSON document from which paths and nested keys are extracted.
        paths_dict (dict): Dictionary to store paths and their corresponding nested keys.
    """
    for path, value in parse_document(doc):
        if len(path) > 1:
            nested_key = path[-1]
            if nested_key == "*": 
                continue
            prefix = tuple(path[:-1]) 

            if prefix not in paths_dict:
                paths_dict[prefix] = set()

            # Add the nested key to the corresponding prefix
            paths_dict[prefix].add(nested_key)


def process_dataset(dataset, filename):
    """
    Process and extract data from the documents, and return a DataFrame with relative frequencies.
    
    Args:
        dataset (str): The name of the dataset file.
        filename (str): The name of the file being processed (e.g., for baseline or other cases).

    Returns:
        dict: Dictionary of paths and their values, with relative frequencies.
    """
    
    paths_dict = defaultdict(set)  
    num_docs = 0  
    matched_document_count = 0 

    # Load the schema
    schema_path = os.path.join(SCHEMA_FOLDER, dataset)
    schema = load_schema(schema_path)

    # Load the dataset
    dataset_path = os.path.join(JSON_FOLDER, dataset)
    with open(dataset_path, 'r') as file:
        for line in file:
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding line in {dataset}: {e}", flush=True)
                continue

            try: 
                # Check if document matches properties of the schema
                if isinstance(doc, dict) and match_properties(schema, doc):
                    matched_document_count += 1 
                    if filename == "baseline_test_data.csv":
                        process_document_baseline(doc, paths_dict)
                    else:
                        process_document(doc, paths_dict)
                    num_docs += 1
                elif isinstance(doc, list):
                    for item in doc:
                        if isinstance(item, dict) and match_properties(schema, item):
                            matched_document_count += 1 
                            if filename == "baseline_test_data.csv":
                                process_document_baseline(item, paths_dict)
                            else:
                                process_document(item, paths_dict)
                            num_docs += 1
            except Exception as e:
                print(f"Error processing line in {dataset}: {e}", flush=True)
                continue

    # Return the dictionary of paths and their values and number of documents
    return paths_dict, num_docs
    

def match_paths(schema_path, json_path):
    """
    Check if a schema path correctly matches a JSON path, considering
    'additional_key' and 'pattern_key'.

    Args:
        schema_path (tuple): Path extracted from the schema, possibly containing 'additional_key' or 'pattern_key'.
        json_path (tuple): Path extracted from a JSON document.

    Returns:
        bool: True if the schema path matches the JSON path, considering regex patterns.
    """
    if len(schema_path) != len(json_path):
        return False

    for schema_part, json_part in zip(schema_path, json_path):
        if schema_part == "additional_key":  # Matches any additional key
            continue
        elif schema_part.startswith("pattern_key"):  # Matches any key that matches the regex pattern
            pattern = schema_part[11:]  # Strip 'pattern_key' prefix
            if not re.fullmatch(pattern, json_part):
                return False
        elif schema_part != json_part:  # Direct match of parts
            return False

    return True


def resolve_internal_refs(schema_def, full_schema, current_depth=0, max_depth=10):
    """
    Recursively resolve internal $ref references within the same schema, with recursion depth limitation
    and support for both $defs and definitions.

    Args:
        schema_def (dict): The partial schema definition.
        full_schema (dict): The complete schema containing all definitions.
        current_depth (int): The current depth of recursion.
        max_depth (int): The maximum allowed depth of recursion.

    Returns:
        dict: A schema with resolved internal references.
    """
    # Check if we've reached the max recursion depth
    if current_depth > max_depth:
        raise ValueError(f"Max recursion depth of {max_depth} exceeded")

    if isinstance(schema_def, dict):
        # Check if the schema has a reference
        if "$ref" in schema_def:
            ref_path = schema_def["$ref"]

            if ref_path.startswith("#/$defs/"):
                ref_name = ref_path.split("/")[-1]
                if "$defs" in full_schema and ref_name in full_schema["$defs"]:
                    return resolve_internal_refs(full_schema["$defs"][ref_name], full_schema, current_depth + 1, max_depth)
                else:
                    raise ValueError(f"Reference {ref_path} not found in $defs")

            elif ref_path.startswith("#/definitions/"):
                ref_name = ref_path.split("/")[-1]
                if "definitions" in full_schema and ref_name in full_schema["definitions"]:
                    return resolve_internal_refs(full_schema["definitions"][ref_name], full_schema, current_depth + 1, max_depth)
                else:
                    raise ValueError(f"Reference {ref_path} not found in definitions")
        
        # Recursively resolve nested properties
        return {k: resolve_internal_refs(v, full_schema, current_depth + 1, max_depth) for k, v in schema_def.items()}
    
    elif isinstance(schema_def, list):
        return [resolve_internal_refs(item, full_schema, current_depth + 1, max_depth) for item in schema_def]
    
    return schema_def
    

def validate_json_against_schema(json_data, schema_def, full_schema):
    """
    Validate JSON data against a partial schema definition while resolving internal references.

    Args:
        json_data (dict): The JSON data to validate.
        schema_def (dict): The partial schema definition.
        full_schema (dict): The full schema containing all definitions.

    Returns:
        bool: True if JSON is valid, False otherwise.
    """
    
    # Convert JSON string to dict if needed
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except json.JSONDecodeError:
            return False

    try:
        # Resolve references within the provided schema
        resolved_schema = resolve_internal_refs(schema_def, full_schema)
        # Validate using jsonschema
        validate(instance=json_data, schema=resolved_schema)
        return True  # Valid JSON

    except jsonschema.exceptions.ValidationError as e:
        return False
    except jsonschema.exceptions.SchemaError as e:
        return False
    except ValueError as e:
        return False
    except RecursionError as e:
        return False


def check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, paths_dict, schema):
    """
    Check if the paths from JSON Schemas exist in JSON datasets, ensuring they conform to the schema definition.

    Args:
        cleaned_ref_defn_paths (dict): Dictionary of referenced definitions and their paths.
        paths_dict (dict): Dictionary of paths and their corresponding values.
        schema (dict): JSON schema object.

    Returns:
        dict: Dictionary of definitions with paths that exist and conform in the JSON documents.
    """
    filtered_ref_defn_paths = defaultdict(set)

    for ref_defn, schema_paths in tqdm(cleaned_ref_defn_paths.items(), desc="Referenced definitions", total=len(cleaned_ref_defn_paths)):
        # Check matching schema and json paths
        for schema_path in schema_paths:
            if schema_path in paths_dict:
                filtered_ref_defn_paths[ref_defn].add(schema_path)

    return filtered_ref_defn_paths
'''
def check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, paths_dict, schema):
    """
    Check if the paths from JSON Schemas exist in JSON datasets, ensuring they conform to the schema definition.

    Args:
        cleaned_ref_defn_paths (dict): Dictionary of referenced definitions and their paths.
        paths_dict (dict): Dictionary of paths and their corresponding values.
        schema (dict): JSON schema object.

    Returns:
        dict: Dictionary of definitions with paths that exist and conform in the JSON documents.
    """
    filtered_ref_defn_paths = defaultdict(set)
    defn_root = schema.get("$defs") or schema.get("definitions", {})

    for ref_defn, schema_paths in tqdm(cleaned_ref_defn_paths.items(), desc="Number of referenced definitions", position=0, leave=False, total=len(cleaned_ref_defn_paths)):
        defn_name = ref_defn.split("/")[-1]
        schema_def = defn_root.get(defn_name, {})

        for schema_path in tqdm(schema_paths, desc="Number of paths for a ref_defn", position=1, leave=False, total=len(schema_paths)):
            for json_path, json_values in tqdm(paths_dict.items(), desc="Number of paths in JSON files", position=2, leave=False, total=len(paths_dict)):
                if match_paths(schema_path, json_path):
                    # Only validate values when 'additional_key' or pattern_key' is present
                    if any(part.startswith("pattern_key") for part in schema_path) or "additional_key" in schema_path:
                        if next((False for value in json_values if not validate_json_against_schema(value, schema_def, schema)), True):
                            filtered_ref_defn_paths[ref_defn].add(json_path)
                    else:
                        filtered_ref_defn_paths[ref_defn].add(json_path)
    return filtered_ref_defn_paths
'''

def find_frequent_definitions(filtered_ref_defn_paths, paths_to_exclude):
    """
    Identify frequently referenced definitions and update paths to exclude.

    Args:
        filtered_ref_defn_paths (dict): Dictionary of reference definitions and their paths.
        paths_to_exclude (set): Paths to exclude from JSON files.

    Returns:
        dict: Dictionary of definitions that are referenced more than once.
    """
    frequent_ref_defn_paths = {}

    # Identify frequently referenced definitions and update paths_to_exclude
    for ref, paths in filtered_ref_defn_paths.items():
        if len(paths) > 1:
            frequent_ref_defn_paths[ref] = paths
        else:
            paths_to_exclude.update(paths)

    # Ensure paths used by multiple definitions are retained in JSON files
    for bad_path in paths_to_exclude.copy():
        if any(bad_path in paths for paths in frequent_ref_defn_paths.values()):
            paths_to_exclude.remove(bad_path)

    # Return dictionary of frequent references
    return frequent_ref_defn_paths





def get_model_and_tokenizer():
    """
    Function to load and return the model, tokenizer, and device.
    """
    MODEL_NAME = "microsoft/codebert-base"
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def tokenize_schema(schema, tokenizer):
    """Tokenize schema.

    Args:
        schema (dict): DataFrame containing pairs, labels, filenames, and schemas of each path in pair
        tokenizer (preTrainedTokenizer): Tokenizer to use for tokenizing schema.

    Returns:
        torch.tensor: tokenized schema
    """

    # Tokenize the schema
    tokenized_schema = tokenizer(schema, return_tensors="pt", max_length=MAX_TOK_LEN, padding="max_length", truncation=True)
    return tokenized_schema


def unique_oneOf(schemas):
    """
    Ensures uniqueness in `oneOf` lists by removing duplicates.
    
    Args:
        schemas (list): List of JSON schemas to deduplicate.

    Returns:
        list: A list of unique schemas.
    """
    serialized = {json.dumps(s, sort_keys=True): s for s in schemas}
    return list(serialized.values())


def merge_schemas(schema1, schema2, depth=0, max_depth=10):
    """
    Recursively merges two JSON schemas.
    - If schemas have different types, they are combined under `oneOf`.
    - Merges properties and required fields for objects.
    - Recursively merges `items` for array schemas.

    Args:
        schema1 (dict): The first JSON schema.
        schema2 (dict): The second JSON schema.
        depth (int): Current recursion depth.
        max_depth (int): Maximum allowed recursion depth.

    Returns:
        dict: The merged JSON schema.
    """
    if depth > max_depth:
        raise RecursionError(f"Exceeded max recursion depth of {max_depth}")

    if not isinstance(schema1, dict) or not isinstance(schema2, dict):
        raise TypeError("Both schema1 and schema2 must be dictionaries.")

    schema1_type = schema1.get("type")
    schema2_type = schema2.get("type")

    # If types match, merge accordingly
    if schema1_type == schema2_type:
        new_schema = deepcopy(schema1)  # Ensure original schema is not modified

        if schema1_type == "object":
            new_schema.setdefault("properties", {})
            schema2_properties = schema2.get("properties", {})

            # Merge `required` fields
            new_schema["required"] = sorted(set(schema1.get("required", [])) | set(schema2.get("required", [])))

            # Recursively merge properties
            for prop, value in schema2_properties.items():
                new_schema["properties"][prop] = merge_schemas(new_schema["properties"].get(prop, {}), value, depth + 1)

        elif schema1_type == "array" and "items" in schema1 and "items" in schema2:
            new_schema["items"] = merge_schemas(schema1["items"], schema2["items"], depth + 1)

        return new_schema

    # Handle different types by combining them under `oneOf`
    if "oneOf" in schema1 and "oneOf" in schema2:
        return {"oneOf": unique_oneOf(schema1["oneOf"] + schema2["oneOf"])}

    if "oneOf" in schema1:
        return {"oneOf": unique_oneOf(schema1["oneOf"] + [schema2])}

    if "oneOf" in schema2:
        return {"oneOf": unique_oneOf([schema1] + schema2["oneOf"])}

    return {"oneOf": unique_oneOf([schema1, schema2])}


def discover_schema(value, max_depth=1, current_depth=0):
    """
    Determine the structure (type) of the JSON key's value, considering only top-level properties of objects
    and top-level items of arrays.

    Args:
        value: The value of the JSON key. It can be of any type.
        max_depth (int): The maximum depth to recurse into nested objects or arrays. Defaults to 1 (top-level only).
        current_depth (int): The current recursion depth (default 0).

    Returns:
        dict: An object representing the structure of the JSON key's value.
    """
    if isinstance(value, str):
        return {"type": "string"}
    elif isinstance(value, float):
        return {"type": "number"}
    elif isinstance(value, int):
        return {"type": "integer"}
    elif isinstance(value, bool):
        return {"type": "boolean"}
    elif isinstance(value, list):
        # Handle lists, assuming mixed types if any items exist
        item_schemas = [discover_schema(item, max_depth, current_depth + 1) for item in value]
        if item_schemas:
            merged_items = reduce(merge_schemas, item_schemas)
        else:
            merged_items = {}
        return {"type": "array", "items": merged_items}
        
    elif isinstance(value, dict):
        schema = {"type": "object", "required": list(set(value.keys())), "properties": {}}
        
        if current_depth < max_depth:
            # Recursively process properties for deeper levels
            for k, v in value.items():
                schema["properties"][k] = discover_schema(v, max_depth, current_depth + 1)
                
        return schema
    elif value is None:
        return {"type": "null"}
    else:
        raise TypeError(f"Unsupported value type: {type(value)}")


def discover_schema_from_values(values):
    """
    Determine the schema for a list of values.
    Args:
        values (list): The list of values to determine the schema for.
        model: The model to use for tokenizing the schema.
    Returns:
        dict: The schema representing the structure of the list of values.
    """
    if not values:
        return {"type": "null"}
    else:
        return reduce(merge_schemas, (discover_schema(v) for v in values))


def calc_average_semantic_similarity(keys, model, tokenizer, device):
    """
    Calculate the average semantic similarity between multiple keys.

    Args:
        keys (list): List of nested keys (strings).

    Returns:
        float: The average cosine similarity between all pairs of keys.
    """
    # Tokenize all keys
    tokenized_keys = [tokenizer(key, return_tensors="pt", padding=True, truncation=True).to(device) for key in keys]

    # Get embeddings for all keys
    embeddings = []
    with torch.no_grad():
        for tokenized_key in tokenized_keys:
            outputs = model(**tokenized_key)
            # Use mean pooling to get a single embedding for each key
            # Flatten the tensor to ensure it's 2D: [num_keys, hidden_size]
            embedding = outputs.last_hidden_state.mean(dim=1)  # Shape: [1, hidden_size]
            embeddings.append(embedding.cpu().numpy().flatten())  # Flatten to ensure 1D vector

    # Convert embeddings to a 2D array
    embeddings = torch.stack([torch.tensor(embedding) for embedding in embeddings])

    # Calculate pairwise cosine similarity for all keys
    similarity_matrix = cosine_similarity(embeddings)

    # Get all pairwise similarity values (excluding the diagonal)
    num_keys = len(keys)
    pairwise_similarities = []
    for i in range(num_keys):
        for j in range(i + 1, num_keys):
            pairwise_similarities.append(similarity_matrix[i][j])

    # Calculate the average similarity value
    average_similarity = sum(pairwise_similarities) / len(pairwise_similarities)
    
    return average_similarity


def calculate_path_frequency(values, num_docs):
    """Calculate the frequency of a path relative to the number of documents."""
    return len(values) / num_docs


def calculate_nested_key_frequencies(parsed_values):
    """Calculate frequencies of nested keys relative to their parent objects.
    Args:
        parsed_values (list): List of parsed JSON values.
    Returns:
        list: List of nested keys.
    """
    nested_frequencies = {}
    for value in parsed_values:
        if isinstance(value, dict):
            for key in value.keys():
                nested_frequencies[key] = nested_frequencies.get(key, 0) + 1
    # Normalize frequencies
    for key in nested_frequencies:
        nested_frequencies[key] /= len(parsed_values)
    return list(nested_frequencies.keys())


def create_dataframe(paths_dict, paths_to_exclude, num_docs):
    """Create a DataFrame of paths and their values schema.

    Args:
        paths_dict (dict): Dictionary of paths and their values.
        paths_to_exclude (set): Paths to remove from JSON files.
        num_docs (int): Number of documents in the dataset.

    Returns:
        pd.DataFrame: DataFrame with tokenized schema added.
    """
    
    df_data = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for path, values in tqdm(paths_dict.items(), desc="Processing paths", total=len(paths_dict), leave=False):
        # Skip paths that are in the exclusion set
        if path in paths_to_exclude:
            continue

        # Parse and discover the schema from values
        parsed_values = [json.loads(v) for v in values]
        try:
            schema = discover_schema_from_values(parsed_values)
        except Exception as e:
            print(f"Error discovering schema for path {path}: {e}", flush=True)
            paths_to_exclude.add(path)
            continue
        
        # Check if the schema has more than one property
        if len(schema.get("properties", {})) > 1:
            path_frequency = calculate_path_frequency(values, num_docs)
            nested_keys = calculate_nested_key_frequencies(parsed_values)
            tokenized_schema = tokenize_schema(json.dumps(schema), tokenizer)
            df_data.append([path, len(path), path_frequency, nested_keys, tokenized_schema, json.dumps(schema)])
        else:
            # Update paths_to_exclude if schema has less than or equal to one property
            paths_to_exclude.add(path)

    # Create DataFrame from collected data
    columns = ["path", "nesting_depth", "path_frequency", "nested_keys", "tokenized_schema", "schema"]
    df = pd.DataFrame(df_data, columns=columns)
    
    return df


def create_dataframe_baseline_model(paths_dict, paths_to_exclude):
    """
    Create a DataFrame of paths and distinct nested keys.

    Args:
        paths_dict (dict): Dictionary of paths and their nested keys.
        paths_to_exclude (set): Paths to exclude from JSON files.

    Returns:
        pd.DataFrame: DataFrame containing paths and distinct nested keys.
    """
    df_data = []

    for path, nested_keys in paths_dict.items():
        # Skip paths that are in the exclusion set
        if path in paths_to_exclude:
            continue
        
        if len(nested_keys) > 1:
            df_data.append([path, nested_keys])
        else:
            # Update paths_to_exclude if schema has less than or equal to one property
            paths_to_exclude.add(path)

    # Create DataFrame from collected data
    columns = ["path", "distinct_nested_keys"]
    df = pd.DataFrame(df_data, columns=columns)
    
    return df.sort_values(by="path")


def update_ref_defn_paths(frequent_ref_defn_paths, df):
    """
    Create updated reference definitions based on intersections with the DataFrame paths.

    Args:
        frequent_ref_defn_paths (dict): Dictionary of frequent reference definitions with their paths.
        df (pd.DataFrame): DataFrame containing paths.

    Returns:
        dict: Updated reference definitions with intersecting paths.
    """
    return {
        ref_defn: set(paths) & set(df["path"])
        for ref_defn, paths in frequent_ref_defn_paths.items()
        if len(set(paths) & set(df["path"])) >= 2
    }


def process_schema(schema_name, filename):
    """
    Process a single schema and return the relevant dataframes and ground truths.

    Args:
        schema_name (str): The name of the schema file.
        filename (str): The name of the file to save the test data.

    Returns:
        tuple: A tuple containing the filtered DataFrame, frequent referenced definition paths, and schema name,
               or (None, None, None) if processing failed.
    """
    # Dictionary to track failures
    failure_flags = {
        "load": 0,
        "ref_defn": 0,
        "object_defn": 0,
        "path": 0,
        "schema_intersection": 0,
        "json_intersection": 0,
        "freq_defn": 0,
        "properties": 0
    }

    schema_path = os.path.join(SCHEMA_FOLDER, schema_name)

    # Load schema
    schema = load_schema(schema_path)
    if schema is None:
        failure_flags["load"] = 1
        #print(f"Failed to load schema {schema_name}.", flush=True)
        return None, None, schema_name, failure_flags

    # Get and clean referenced definitions
    ref_defn_paths = clean_ref_defn_paths(schema)

    if not ref_defn_paths:
        failure_flags["ref_defn"] = 1
        #print(f"No referenced definitions in {schema_name}.", flush=True)
        return None, None, schema_name, failure_flags
    '''
    for ref, paths in ref_defn_paths.items():
        print(f"Reference: {ref} Paths: {paths} in Schema: {schema_name}", flush=True)
    print("_________________________________________________________________________________________________________________________")
    '''
    # Handle nested definitions
    new_ref_defn_paths = handle_nested_definitions(ref_defn_paths)
    cleaned_ref_defn_paths = remove_definition_keywords(new_ref_defn_paths)
    '''
    for ref, paths in cleaned_ref_defn_paths.items():
        print(f"Reference: {ref} Paths: {paths}, Schema: {schema_name}", flush=True)
    print("_________________________________________________________________________________________________________________________")
    '''
    # Get referenced definitions of type object
    paths_to_exclude = set()
    get_ref_defn_of_type_obj(schema, cleaned_ref_defn_paths, paths_to_exclude)
    if not cleaned_ref_defn_paths:
        failure_flags["object_defn"] = 1
        #print(f"No referenced definitions of type object in {schema_name}.", flush=True)
        return None, None, schema_name, failure_flags
    '''
    for ref, paths in cleaned_ref_defn_paths.items():
        print(f"Reference: {ref} Paths: {paths}", flush=True)
    print("_________________________________________________________________________________________________________________________")
    '''
 
    paths_dict, num_docs = process_dataset(schema_name, filename)
    if len(paths_dict) == 0:
        failure_flags["path"] = 1
        #print(f"No paths extracted from {schema_name}.", flush=True)
        return None, None, schema_name, failure_flags

    # Check reference definition paths in the dataset
    filtered_ref_defn_paths = check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, paths_dict, schema)
    if not filtered_ref_defn_paths:
        failure_flags["schema_intersection"] = 1
        #print(f"No paths of properties in referenced definitions found in {schema_name} dataset.", flush=True)
        return None, None, schema_name, failure_flags
    '''
    for ref, paths in filtered_ref_defn_paths.items():
        print(f" Object Reference: {ref} Paths: {paths}", flush=True)
    print("_________________________________________________________________________________________________________________________")
    '''
    
    # Find frequent definitions
    frequent_ref_defn_paths = find_frequent_definitions(filtered_ref_defn_paths, paths_to_exclude)
    if not frequent_ref_defn_paths:
        failure_flags["freq_defn"] = 1
        #print(f"No frequent referenced definitions found in {schema_name}.", flush=True)
        return None, None, schema_name, failure_flags
    '''
    for ref, paths in frequent_ref_defn_paths.items():
        print(f"Frequent Reference: {ref} Paths: {paths}")
    print("_________________________________________________________________________________________________________________________")
    '''
    # Create DataFrame
    if filename == "baseline_test_data.csv":
        df = create_dataframe_baseline_model(paths_dict, paths_to_exclude)
    else:
        df = create_dataframe(paths_dict, paths_to_exclude, num_docs)
        print(f"Number of paths in {schema_name}: {len(df)}", flush=True)

    # Update reference definitions
    updated_ref_defn_paths = update_ref_defn_paths(frequent_ref_defn_paths, df)
    if not updated_ref_defn_paths:
        failure_flags["properties"] = 1
        #print(f"Not enough properties found under refererenced definitions in {schema_name}.", flush=True)
        return None, None, schema_name, failure_flags

    df["filename"] = schema_name
    df.reset_index(drop=True, inplace=True)
    
    return df, updated_ref_defn_paths, schema_name, failure_flags





def calculate_embeddings(df, model, device):
    """
    Calculate the embeddings for each path in batches.

    Args:
        df (pd.DataFrame): DataFrame containing paths and their tokenized schemas.
        model (AutoAdapterModel): Pretrained model to compute embeddings.
        device (torch.device): Device to move tensors to.

    Returns:
        dict: Dictionary mapping paths to their corresponding embeddings.
    """

    schema_embeddings = {}
    paths = df["path"].tolist()
    tokenized_schemas = df["tokenized_schema"].tolist()

    # Create batches of tokenized schemas
    for batch_start in tqdm(range(0, len(tokenized_schemas), BATCH_SIZE), desc="Calculating schema embeddings", position=0, leave=False):
        batch_paths = paths[batch_start: batch_start + BATCH_SIZE]
        batch_schemas = tokenized_schemas[batch_start: batch_start + BATCH_SIZE]

        # Prepare batch tensors
        batch_input_ids = [torch.tensor(schema["input_ids"]) for schema in batch_schemas]
        batch_attention_mask = [torch.tensor(schema["attention_mask"]) for schema in batch_schemas]

        # Pad sequences to get a uniform length
        batch_input_ids = pad_sequence(batch_input_ids, batch_first=True)
        batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True)

        # Move tensors to the device
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)

        # Use model to compute embeddings
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)

        # Store embeddings
        for path, embedding in zip(batch_paths, batch_embeddings):
            schema_embeddings[path] = embedding.cpu()

    return schema_embeddings


def label_samples(df, good_pairs, bad_pairs):
    """
    Label the samples in the DataFrame based on good and bad pairs.

    Args:
        df (pd.DataFrame): DataFrame containing paths, schemas, and filenames.
        good_pairs (set): Set of paths that should be grouped together.
        bad_pairs (set): Set of paths that should not be grouped together.

    Returns:
        pd.DataFrame: DataFrame containing labeled pairs, schemas, and filenames.
    """
    # Create lists to store data
    paths1 = []
    paths2 = []
    labels = []
    path1_freqs = []
    path2_freqs = []
    nested_keys1 = []
    nested_keys2 = []
    schemas1 = [] 
    schemas2 = []  
    filenames = []  
    nesting_depths1 = []
    nesting_depths2 = []

    # Process good pairs: label them as 1 (positive)
    for path1, path2 in good_pairs:
        paths1.append(path1)
        paths2.append(path2)
        labels.append(1)

        # Extract schemas and filename for both paths
        path1_row = df[df["path"] == path1].iloc[0]
        path2_row = df[df["path"] == path2].iloc[0]
        path1_freqs.append(path1_row["path_frequency"])
        path2_freqs.append(path2_row["path_frequency"])
        nested_keys1.append(path1_row["nested_keys"])
        nested_keys2.append(path2_row["nested_keys"])
        filenames.append(path1_row["filename"])
        schemas1.append(path1_row["schema"])
        schemas2.append(path2_row["schema"])
        nesting_depths1.append(len(path1))
        nesting_depths2.append(len(path2))
        
    # Process bad pairs: label them as 0 (negative)
    for path1, path2 in bad_pairs:
        paths1.append(path1)
        paths2.append(path2)
        labels.append(0)

        # Extract schemas and filename for both paths
        path1_row = df[df["path"] == path1].iloc[0]
        path2_row = df[df["path"] == path2].iloc[0]
        path1_freqs.append(path1_row["path_frequency"])
        path2_freqs.append(path2_row["path_frequency"])
        nested_keys1.append(path1_row["nested_keys"])
        nested_keys2.append(path2_row["nested_keys"])
        filenames.append(path1_row["filename"])
        schemas1.append(path1_row["schema"])
        schemas2.append(path2_row["schema"])
        nesting_depths1.append(len(path1))
        nesting_depths2.append(len(path2))

    # Create a new DataFrame with separate columns for path1 and path2
    labeled_df = pd.DataFrame({
        "filename": filenames,
        "label": labels,
        "path1": paths1,
        "path2": paths2,
        "nesting_depth1": nesting_depths1,
        "nesting_depth2": nesting_depths2,
        "path1_freq": path1_freqs,
        "path2_freq": path2_freqs,
        "nested_keys1": nested_keys1,
        "nested_keys2": nested_keys2,
        "schema1": schemas1,
        "schema2": schemas2
    })

    return labeled_df


def get_samples(df, frequent_ref_defn_paths, best_good_pairs):
    """
    Generate labeled samples of good and bad pairs from the DataFrame based on ground truth definitions.

    Args:
        df (pd.DataFrame): DataFrame containing paths and schemas.
        frequent_ref_defn_paths (dict): Dictionary of frequent referenced definition and their paths.
        best_good_pairs (bool, optional): Whether to select the best good pairs. Defaults to False.

    Returns:
        pd.DataFrame: Labeled dataFrame containing sample paths and schemas.
    """

    all_good_pairs = set()
    all_bad_pairs = set()
    all_good_paths = set()
    sample_good_pairs = set()
    sample_bad_pairs = set()

    # Load the model and tokenizer
    model, tokenizer, device = get_model_and_tokenizer()

    # Calculate the embeddings of the tokenized schema
    schema_embeddings = calculate_embeddings(df, model, device)
    
    # Get all paths from the DataFrame
    paths = list(df["path"])

    # Process good paths
    for ref_defn, good_paths in tqdm(frequent_ref_defn_paths.items(), desc="Processing good pairs", position=0, leave=False):
        all_good_paths.update(good_paths)
        good_pairs = list(itertools.combinations(good_paths, 2))

        # Extract keys from nested_keys
        path_to_keys = {path: set(df.loc[df["path"] == path, "nested_keys"].values[0]) for path in good_paths}

        # Filter pairs with at least 2 common keys
        all_good_pairs.update(good_pairs)
        good_pairs = [(p1, p2) for p1, p2 in good_pairs if len(path_to_keys[p1] & path_to_keys[p2]) >= 2]
        
        # Sampling logic
        if best_good_pairs:
            # Calculate distances for good pairs
            good_pairs_distances = [
                ((path1, path2), cosine(schema_embeddings[path1], schema_embeddings[path2]))
                for path1, path2 in good_pairs
            ]
            
            # Select top 1,000 pairs with greatest distances
            top_1000_good_pairs = nlargest(1000, good_pairs_distances, key=lambda x: x[1])
            sample_good_pairs.update(pair for pair, _ in top_1000_good_pairs)
        else:
            # Take first 1,000 pairs
            sample_good_pairs.update(good_pairs[:1000])

    # Process bad paths
    all_pairs = list(itertools.combinations(paths, 2))
    
    # Create a set of schemas for all good paths for quick lookup
    good_schemas = set(df.loc[df["path"].isin(all_good_paths), "schema"])
    
    # Create a dictionary for path-to-schema mapping for faster lookups
    path_to_schema = df.set_index("path")["schema"].to_dict()
    
    # Loop through all pairs and add to bad pairs if not in good pairs
    for path1, path2 in all_pairs:
        if ((path1, path2) not in all_good_pairs and 
            (path2, path1) not in all_good_pairs):
            
            schema1_good = path_to_schema.get(path1) in good_schemas
            schema2_good = path_to_schema.get(path2) in good_schemas
    
            if not (schema1_good and schema2_good):
                all_bad_pairs.add((path1, path2))
    
    # Calculate distances for bad pairs
    bad_pairs_distances = [
        ((path1, path2), cosine(schema_embeddings[path1], schema_embeddings[path2]))
        for path1, path2 in all_bad_pairs
    ]
    
    # Select pairs with smallest distances 
    num_bad_pairs = min(len(sample_good_pairs), len(all_bad_pairs))
    top_bad_pairs = nsmallest(num_bad_pairs, bad_pairs_distances, key=lambda x: x[1])
    sample_bad_pairs.update(pair for pair, _ in top_bad_pairs)

    # Label data
    labeled_df = label_samples(df, sample_good_pairs, sample_bad_pairs)
   
    # Calculate cosine similarity between the two paths
    labeled_df = calculate_cosine_similarity(labeled_df, model, tokenizer, device)

    return labeled_df



def get_unique_paths(df):
    """
    Extract unique paths from both path1 and path2 columns in the dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing 'path1' and 'path2' columns.

    Returns:
        set: Set of unique paths.
    """
    # Collect unique paths from both path1 and path2 columns
    unique_paths = set(df["path1"].tolist() + df["path2"].tolist())
    return unique_paths


def tuple_to_string(path_tuple):
    """
    Converts a tuple representing a JSON path into a string.

    Args:
        path_tuple (tuple): A tuple representing a JSON path.

    Returns:
        str: The string representation of the JSON path.
    """
    return '.'.join(path_tuple)


def tokenize_paths(paths, tokenizer, device):
    """
    Tokenize a list of paths and prepare them for the model.

    Args:
        paths (list of str): List of JSON paths to tokenize.
        tokenizer: Tokenizer to use for tokenizing paths.
        device: Device to move tokenized inputs to.

    Returns:
        dict: Tokenized paths with input_ids and attention_mask tensors on the device.
    """
    return tokenizer(paths, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOK_LEN).to(device)


def compute_embeddings(tokenized_inputs, model):
    """
    Compute embeddings from tokenized inputs.

    Args:
        tokenized_inputs (dict): Tokenized inputs containing input_ids and attention_mask.
        model: Pretrained model to compute embeddings.

    Returns:
        torch.Tensor: Embeddings of shape (batch_size, embedding_dim).
    """
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


def precompute_embeddings(unique_paths, model, tokenizer, device):
    """
    Precompute embeddings for unique paths in batches.

    Args:
        unique_paths (set): Set of unique paths.
        model: Pretrained model to compute embeddings.
        tokenizer: Tokenizer to tokenize paths.
        device: Device to move inputs and embeddings.

    Returns:
        dict: Cache of precomputed embeddings for each path.
    """
    embeddings_cache = {}
    unique_paths_list = list(unique_paths)

    for i in tqdm(range(0, len(unique_paths_list), BATCH_SIZE), desc="Precomputing embeddings", position=0, leave=False, total=len(unique_paths_list) // BATCH_SIZE):
        # Batch paths to process
        batch_paths = unique_paths_list[i:i + BATCH_SIZE]
        
        # Convert tuples to strings for tokenization
        batch_paths_strings = [tuple_to_string(path) for path in batch_paths]
        
        # Tokenize paths
        tokenized_inputs = tokenize_paths(batch_paths_strings, tokenizer, device)
        
        # Compute embeddings
        embeddings = compute_embeddings(tokenized_inputs, model)

        # Add embeddings to cache using original tuples as keys
        for original_path, embedding in zip(batch_paths, embeddings):
            embeddings_cache[original_path] = embedding

        # Add embeddings to cache
        for path, embedding in zip(batch_paths, embeddings):
            embeddings_cache[path] = embedding

    return embeddings_cache


def calculate_similarities(df, embeddings_cache):
    """
    Calculate cosine similarities between path1 and path2 using precomputed embeddings.

    Args:
        df (pd.DataFrame): DataFrame containing 'path1' and 'path2' columns.
        embeddings_cache (dict): Precomputed embeddings for paths.

    Returns:
        list: List of cosine similarity scores for each row in the DataFrame.
    """
    similarities = []

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]
        path1_batch = batch["path1"].tolist()
        path2_batch = batch["path2"].tolist()

        # Retrieve embeddings from cache
        embedding1_batch = torch.stack([embeddings_cache[path] for path in path1_batch])
        embedding2_batch = torch.stack([embeddings_cache[path] for path in path2_batch])

        # Calculate cosine similarity
        embedding1_batch = embedding1_batch.cpu().numpy()
        embedding2_batch = embedding2_batch.cpu().numpy()
        similarities_batch = cosine_similarity(embedding1_batch, embedding2_batch)
        similarities.extend(similarities_batch.diagonal())

    return similarities


def calculate_cosine_similarity(df, model, tokenizer, device):
    """
    Main function to calculate cosine similarity between path1 and path2.

    Args:
        df (pd.DataFrame): DataFrame containing 'path1' and 'path2' columns.
        model (PreTrainedModel): Pretrained model to compute embeddings.
        tokenizer (PreTrainedTokenizer): Tokenizer to tokenize paths.
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').
        
        
    Returns:
        pd.DataFrame: DataFrame with an additional column 'cosine_similarity'.
    """
    # Extract unique paths from path1 and path2 columns
    unique_paths = set(df["path1"]).union(df["path2"])

    # Precompute embeddings for unique paths
    embeddings_cache = precompute_embeddings(unique_paths, model, tokenizer, device)

    # Calculate similarities between path1 and path2 for each row
    similarities = calculate_similarities(df, embeddings_cache)

    # Add the similarity as a new column to the DataFrame
    df["cosine_similarity"] = similarities

    return df


def calculate_cosine_similarity_2(df, model, tokenizer, device):
    """
    Computes cosine similarity between path1 and path2 for each filename separately.

    Args:
        df (pd.DataFrame): DataFrame containing 'filename', 'path1', and 'path2' columns.
        model (PreTrainedModel): Pretrained model to compute embeddings.
        tokenizer (PreTrainedTokenizer): Tokenizer to tokenize paths.
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').

    Returns:
        pd.DataFrame: DataFrame with an additional 'cosine_similarity' column.
    """
    all_results = []

    # Process each schema (filename) separately
    for filename, group in df.groupby("filename"):
        unique_paths = set(group["path1"]).union(group["path2"])

        # Compute embeddings for this file only
        embeddings_cache = precompute_embeddings(unique_paths, model, tokenizer, device)

        # Compute similarities for this specific file
        similarities = calculate_similarities(group, embeddings_cache)

        # Assign results to the group and store
        group = group.copy()
        group["cosine_similarity"] = similarities
        all_results.append(group)

    # Combine results from all filenames
    return pd.concat(all_results, ignore_index=True)

def save_ground_truths(ground_truths, ground_truth_file):
    """
    Save ground truths to a JSON file.

    Args:
        ground_truths (dict): The ground truths to save.
        ground_truth_file (str): The file to save the ground truths to.
    """
    ground_truths_serializable = {
        schema_name: {ref_defn: list(path_list) if isinstance(path_list, set) else path_list for ref_defn, path_list in subdict.items()}
        for schema_name, subdict in ground_truths.items()
    }

    with open(ground_truth_file, "w") as json_file:
        for ref_defn, subdict in ground_truths_serializable.items():
            json_file.write(json.dumps({ref_defn: subdict}) + '\n')


def preprocess_data(schemas, filename, ground_truth_file):
    """
    Process all the data from the JSON files to get their embeddings using multiple CPUs.

    Args:
        schemas (list): List of schema filenames.
        filename (str): Filename to save the resulting DataFrame.
        ground_truth_file (str): Filename to save the ground truth definitions.
    """

    ground_truths = defaultdict(dict)
    
    total_schemas = len(schemas)
    load_count = 0
    ref_defn = 0
    object_defn = 0
    path = 0
    schema_intersection = 0
    freq_defn = 0
    properties = 0

    #schemas = ["label-commenter-config-yml.json"]
    # Limit the number of concurrent workers to prevent memory overload
    for i in tqdm(range(0, len(schemas), BATCH_SIZE), position=0, desc="Processing schemas", leave=True, total=len(schemas)):
        batch = schemas[i:i+BATCH_SIZE]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_schema, schema, filename): schema for schema in batch}

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=1, desc="Processing batch", leave=False):
                try:
                    df, frequent_ref_defn_paths, schema_name, failure_flags = future.result()
                    
                    load_count += failure_flags["load"]
                    ref_defn += failure_flags["ref_defn"]
                    object_defn += failure_flags["object_defn"]
                    path += failure_flags["path"]
                    schema_intersection += failure_flags["schema_intersection"]
                    freq_defn += failure_flags["freq_defn"]
                    properties += failure_flags["properties"]

                    if df is not None and frequent_ref_defn_paths:
                        ground_truths[schema_name] = frequent_ref_defn_paths
                        
                        if filename != "baseline_test_data.csv":
                            df = get_samples(df, frequent_ref_defn_paths, best_good_pairs=True)
                        
                        # Append batch to CSV to avoid holding everything in memory
                        df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False, sep=';')

                except Exception as e:
                    print(f"Error processing schema {futures[future]}: {e}", flush=True)
    
    # Save ground truth definitions once all processing is done
    save_ground_truths(ground_truths, ground_truth_file)
    print("Total schemas processed:", total_schemas)
    print("Schemas that loaded:", total_schemas - load_count)
    print("Schemas with referenced definitions:", total_schemas - load_count - ref_defn)
    print("Schemas with object definitions:", total_schemas - load_count - ref_defn - object_defn)
    print("Schemas with paths:", total_schemas - load_count - ref_defn - object_defn - path)
    print("Schemas with schema intersection:", total_schemas - load_count - ref_defn - object_defn - path - schema_intersection)
    print("Schemas with frequent definitions:", total_schemas - load_count - ref_defn - object_defn - path - schema_intersection - freq_defn)
    print("Schemas with properties:", total_schemas - load_count - ref_defn - object_defn - path - schema_intersection - freq_defn - properties)

'''
def preprocess_data(schemas, filename, ground_truth_file):
    """
    Process all the data from the JSON files to get their embeddings sequentially.

    Args:
        schemas (list): List of schema filenames.
        filename (str): Filename to save the resulting DataFrame.
        ground_truth_file (str): Filename to save the ground truth definitions.
    """

    ground_truths = defaultdict(dict)
    
    total_schemas = len(schemas)
    load_count = 0
    ref_defn = 0
    object_defn = 0
    path = 0
    schema_intersection = 0
    freq_defn = 0
    properties = 0

    for schema in tqdm(schemas, position=0, desc="Processing schemas", leave=True, total=total_schemas):
        try:
            df, frequent_ref_defn_paths, schema_name, failure_flags = process_schema(schema, filename)
            
            load_count += failure_flags["load"]
            ref_defn += failure_flags["ref_defn"]
            object_defn += failure_flags["object_defn"]
            path += failure_flags["path"]
            schema_intersection += failure_flags["schema_intersection"]
            freq_defn += failure_flags["freq_defn"]
            properties += failure_flags["properties"]

            if df is not None and frequent_ref_defn_paths:
                ground_truths[schema_name] = frequent_ref_defn_paths
                
                if filename != "baseline_test_data.csv":
                    df = get_samples(df, frequent_ref_defn_paths, best_good_pairs=True)
                
                # Append batch to CSV to avoid holding everything in memory
                df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False, sep=';')

        except Exception as e:
            print(f"Error processing schema {schema}: {e}", flush=True)

    # Save ground truth definitions once all processing is done
    save_ground_truths(ground_truths, ground_truth_file)
    print("Total schemas processed:", total_schemas)
    print("Schemas that loaded:", total_schemas - load_count)
    print("Schemas with referenced definitions:", total_schemas - load_count - ref_defn)
    print("Schemas with object definitions:", total_schemas - load_count - ref_defn - object_defn)
    print("Schemas with paths:", total_schemas - load_count - ref_defn - object_defn - path)
    print("Schemas with schema intersection:", total_schemas - load_count - ref_defn - object_defn - path - schema_intersection)
    print("Schemas with frequent definitions:", total_schemas - load_count - ref_defn - object_defn - path - schema_intersection - freq_defn)
    print("Schemas with properties:", total_schemas - load_count - ref_defn - object_defn - path - schema_intersection - freq_defn - properties)

'''
def delete_file_if_exists(filename):
    """
    Deletes the specified file if it exists.
    
    Args:
        filename (str): The name of the file to delete.
    """
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Deleted file: {filename}")
    else:
        print(f"File {filename} does not exist.")


def main():
    try:
        # Parse command-line arguments
        train_size, random_value, m = sys.argv[-3:]
        train_ratio = float(train_size)
        random_value = int(random_value)


        # Split the data into training and testing sets
        train_set, test_set = split_data(train_ratio, random_value)

        if m == "baseline":
            # Files to be checked for deletion
            files_to_delete = [
                "baseline_test_data.csv",
                "baseline_test_ground_truth.json",
            ]

            # Delete files if they exist
            for file in files_to_delete:
                delete_file_if_exists(file)

            # Preprocess the testing data for the baseline model
            preprocess_data(test_set, filename="baseline_test_data.csv", ground_truth_file="baseline_test_ground_truth.json")

        else:
            # Files to be checked for deletion
            files_to_delete = [
                "sample_train_data.csv",
                "train_ground_truth.json",
                "sample_test_data.csv",
                "test_ground_truth.json"
            ]

            # Delete files if they exist
            for file in files_to_delete:
                delete_file_if_exists(file)

            # Start a timer
            start_time = time.time()
            # Preprocess the training and testing data
            preprocess_data(train_set, filename="sample_train_data.csv", ground_truth_file="train_ground_truth.json")
            preprocess_data(test_set, filename="sample_test_data.csv", ground_truth_file="test_ground_truth.json")
            # End the timer
            end_time = time.time()
            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print(f"Time taken for preprocessing: {elapsed_time:.2f} seconds")

    except (ValueError, IndexError) as e:
        print(f"Error: {e}\nUsage: script.py <train_size> <random_value> <model>")
        sys.exit(1)
    

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    mp.set_start_method('spawn', force=True)
    main()