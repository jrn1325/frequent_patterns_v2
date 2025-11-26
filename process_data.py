import argparse
import concurrent.futures
import json
import math
import numpy as np
import os
import pandas as pd
import random
import re
import sys
import time
import torch
import torch.multiprocessing as mp
import warnings

from adapters import AutoAdapterModel
from collections import defaultdict, Counter
from collections.abc import Mapping
from heapq import nlargest, nsmallest
from itertools import combinations
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore")

# Create constant variables
DEFINITION_KEYWORDS = {"$defs", "definitions"}
JSON_SCHEMA_KEYWORDS = {"properties", "patternProperties", "additionalProperties", "items", "prefixItems", "allOf", "oneOf", "anyOf", "not", "if", "then", "else", "$ref"}
JSON_SUBSCHEMA_KEYWORDS = {"allOf", "oneOf", "anyOf"}

SCHEMA_FOLDER = "converted_processed_schemas"
JSON_FOLDER = "processed_jsons"
MAX_TOK_LEN = 512
BATCH_SIZE = 64
MODEL_NAME = "microsoft/codebert-base" 
ARRAY_WILDCARD = "<ARRAY_ITEM>"


# -------------------------------
# Data Splitting 
# -------------------------------
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
    If the schema has no 'properties', assume it matches any document.

    Args:
        schema (dict): JSON Schema object.
        document (dict): JSON object.

    Returns:
        bool: True if there is an intersection or the schema has no properties, False otherwise.
    """
    # Extract top-level schema properties
    schema_properties = set(schema.get("properties", {}).keys())

    # If schema has no properties, treat it as open/unknown
    if not schema_properties:
        return True

    # Extract top-level document properties
    document_properties = set(document.keys())

    # Check if there is an intersection
    return bool(schema_properties & document_properties)


# -------------------------------
# Schema Parsing
# -------------------------------
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
            updated_path = current_path + (ARRAY_WILDCARD,)
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

        elif key in {"$defs", "definitions"} and isinstance(value, Mapping):
            for def_name, def_schema in value.items():
                if isinstance(def_schema, Mapping):
                    yield from find_ref_paths(def_schema, current_path + (key, def_name), is_data_level=False)



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
    resolved = defn_root.get(ref_name)
    if resolved and "$ref" in resolved:
        return resolve_ref(defn_root, resolved["$ref"])
    return resolved

def is_object_like(defn_obj, defn_root):
    """
    Check if a schema or any dereferenced subschemas are object-like with at least one property.
    
    Args:
        defn_obj (dict): The schema object to check.
        defn_root (dict): The root schema containing all definitions.
    Returns:
        bool: True if the schema is object-like with at least one property, False otherwise.
    """
    if not isinstance(defn_obj, dict):
        return False

    # Resolve top-level $ref
    while "$ref" in defn_obj:
        defn_obj = resolve_ref(defn_root, defn_obj["$ref"])
        if defn_obj is None:
            return False

    # Check if this schema is object-like
    if defn_obj.get("type") == "object" and (
        "properties" in defn_obj or
        "additionalProperties" in defn_obj or
        "patternProperties" in defn_obj
    ):
        if "properties" not in defn_obj or len(defn_obj["properties"]) >= 1:
            return True

    # Recurse into combinators (oneOf, anyOf, etc.)
    for keyword in JSON_SUBSCHEMA_KEYWORDS:
        subschemas = defn_obj.get(keyword)
        if subschemas:
            if not isinstance(subschemas, list):
                subschemas = [subschemas]
            for sub in subschemas:
                if "$ref" in sub:
                    sub = resolve_ref(defn_root, sub["$ref"])
                if is_object_like(sub, defn_root):
                    return True

    return False

def get_ref_defn_of_type_obj(json_schema, ref_defn_paths):
    """
    Filter out references to definitions that do not represent object-like schemas (e.g., no 'properties').

    Args:
        json_schema (dict): The JSON schema containing the referenced definitions.
        ref_defn_paths (dict): Dictionary mapping references (e.g., "#/$defs/Foo") to schema paths using them.

    Returns:
        set: Updated set of paths to exclude.
    """
    paths_to_exclude = set()
    defn_root = json_schema.get("$defs") or json_schema.get("definitions", {})
    ref_to_delete = []

    for ref in ref_defn_paths.keys():
        defn_name = ref.split("/")[-1]
        defn_obj = defn_root.get(defn_name)

        if defn_obj is None:
            print(f"Reference {ref} does not resolve to a known definition.", flush=True)
            ref_to_delete.append(ref)
            continue

        # Check if the resolved definition is object-like
        if not is_object_like(defn_obj, defn_root):
            ref_to_delete.append(ref)

    for ref in ref_to_delete:
        ref_paths = ref_defn_paths.pop(ref, set())
        paths_to_exclude.update(ref_paths)

    return paths_to_exclude


# -------------------------------
# Document Parsing
# -------------------------------
def extract_paths(doc, path=("$",)):
    """
    Get the path of each key and its value from the JSON document.

    Args:
        doc (dict or list): JSON document to traverse.
        path (tuple, optional): The current path of keys. Defaults to ("$",).

    Yields:
        tuple: A tuple containing the path and the corresponding value.

    Raises:
        ValueError: If the input document is not a dict or list.
    """
    if isinstance(doc, dict):
        for key, value in doc.items():
            current_path = path + (key,)
            yield current_path, value
            if isinstance(value, (dict, list)):
                yield from extract_paths(value, current_path)

    elif isinstance(doc, list):
        for index, item in enumerate(doc):
            current_path = path + (ARRAY_WILDCARD,)
            yield current_path, item
            if isinstance(item, (dict, list)):
                yield from extract_paths(item, current_path)

    else:
        raise ValueError(f"Expected dict or list, got {type(doc).__name__}")

def process_document(doc, path_values, path_freqs):
    """
    Extracts paths from the given JSON document and stores them in dictionaries,
    grouping paths that share the same prefix and capturing the frequency and data type of nested keys.

    Args:
        doc: JSON document
        path_values: dict mapping path tuples to type information
        path_freqs: Counter for path frequencies
    """ 
    for path, value in extract_paths(doc):
        if len(path) > 1:
            nested_key = path[-1]
            if nested_key == ARRAY_WILDCARD:
                continue

            prefix = path[:-1]
            value_type = get_json_format(value)

            if prefix not in path_values:
                path_values[prefix] = {}

            if nested_key not in path_values[prefix]:
                path_values[prefix][nested_key] = {"frequency": 0.0, "type": set()}

            # Update frequency and type for the nested key
            path_values[prefix][nested_key]["frequency"] += 1
            path_values[prefix][nested_key]["type"].add(value_type)

            # Update parent frequency (number of times this object appears)
            path_freqs[prefix] = path_freqs.get(prefix, 0) + 1

def process_dataset(dataset, paths_to_exclude):
    """
    Process a JSON-lines dataset and extract object-like paths,
    filtering documents that match the schema and collecting
    frequencies + types of each path.

    Args:
        dataset (str): Dataset filename.
        paths_to_exclude (set): Paths to exclude.

    Returns:
        tuple: (path_values, path_freqs, valid_docs)
    """
    path_values = defaultdict(lambda: defaultdict(lambda: {"frequency": 0.0, "type": set()}))
    path_freqs = Counter()
    valid_docs = []

  
    # Load schema
    schema_path = os.path.join(SCHEMA_FOLDER, dataset)
    schema = load_schema(schema_path)

    # Load dataset
    dataset_path = os.path.join(JSON_FOLDER, dataset)

    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                doc = json.loads(line)

                # Case 1: simple dict
                if isinstance(doc, dict) and match_properties(schema, doc):
                    process_document(doc, path_values, path_freqs)
                    valid_docs.append(doc)

                # Case 2: dataset is JSONL of lists-of-dicts
                elif isinstance(doc, list):
                    for item in doc:
                        if isinstance(item, dict) and match_properties(schema, item):
                            process_document(item, path_values, path_freqs)
                            valid_docs.append(item)

            except Exception as e:
                print(f"[{dataset}] Error parsing line: {e}")
                continue

    # No extracted object paths
    if len(path_values) == 0:
        print(f"No object-like paths extracted from {dataset}.")
        return {}, {}, valid_docs

    # Add all infrequent paths (freq <= 1) to paths_to_exclude
    infrequent = {p for p, f in path_freqs.items() if f <= 1}
    paths_to_exclude.update(infrequent)

    # Keep only frequent paths that are not excluded
    frequent_paths = {
        p for p, f in path_freqs.items()
        if f > 1 and p not in paths_to_exclude
    }

    # Filter path_freqs and path_values safely
    path_freqs = {p: f for p, f in path_freqs.items() if p in frequent_paths}
    path_values = {p: v for p, v in path_values.items() if p in frequent_paths}

    print(f"Processed {dataset}: {len(valid_docs)} valid documents, {len(path_values)} frequent paths retained.", flush=True)
    return path_values, path_freqs, valid_docs


# -------------------------------
# Schema Reference Resolution
# -------------------------------
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
    
def match_paths(schema_path, json_path):
    """
    Check if a schema path matches a JSON path, allowing for wildcard and regex components.

    Args:
        schema_path (tuple): A tuple from the schema, may include "additional_key" or "pattern_key<regex>".
        json_path (tuple): A tuple representing a concrete path in a JSON document.

    Returns:
        bool: True if the schema path matches the JSON path, False otherwise.
    """
    if len(schema_path) != len(json_path):
        return False

    for schema_part, json_part in zip(schema_path, json_path):
        if schema_part == "additional_key":
            continue  # wildcard: match any key
        elif schema_part.startswith("pattern_key"):
            pattern = schema_part[len("pattern_key"):]
            try:
                if not re.fullmatch(pattern, json_part):
                    return False
            except re.error as e:
                raise ValueError(f"Invalid regex pattern in schema_path: '{pattern}' -> {e}")
        elif schema_part != json_part:
            return False

    return True

def check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, paths_values):
    """
    Check if the paths from JSON Schemas exist in JSON datasets, ensuring they conform to the schema definition.

    Args:
        cleaned_ref_defn_paths (dict): Dictionary of referenced definitions and their paths.
        paths_values (dict): Dictionary of paths and their corresponding values.

    Returns:
        dict: Dictionary of definitions with paths that exist and conform in the JSON documents.
    """
    filtered_ref_defn_paths = defaultdict(set)

    for ref_defn, schema_paths in tqdm(cleaned_ref_defn_paths.items(), desc="Number of referenced definitions", position=0, leave=False, total=len(cleaned_ref_defn_paths)):

        for schema_path in tqdm(schema_paths, desc="Number of paths for a ref_defn", position=1, leave=False, total=len(schema_paths)):
            for json_path in tqdm(paths_values.keys(), desc="Number of paths in JSON files", position=2, leave=False, total=len(paths_values)):
                if match_paths(schema_path, json_path):
                    filtered_ref_defn_paths[ref_defn].add(json_path)
    return filtered_ref_defn_paths

def find_frequent_definitions(filtered_ref_defn_paths, paths_to_exclude):
    """
    Identify frequently referenced definitions and update paths to exclude.

    Args:
        filtered_ref_defn_paths (dict): {definition_ref: set of paths}.
        paths_to_exclude (set): Paths to exclude (will be updated in-place).

    Returns:
        dict: {definition_ref: set of paths} where the definition is referenced more than once.
    """
    frequent_ref_defn_paths = {}
    temp_exclude = set()

    for ref, paths in filtered_ref_defn_paths.items():
        if len(paths) > 1:
            frequent_ref_defn_paths[ref] = paths
        else:
            temp_exclude.update(paths)

    # Remove from exclude list any path that's shared with a frequent reference
    shared_paths = set().union(*frequent_ref_defn_paths.values())
    paths_to_exclude.update(temp_exclude - shared_paths)

    return frequent_ref_defn_paths


# -------------------------------
# DataFrame Creation and Semantic Similarity
# -------------------------------
def load_model():
    """
    Function to load and return the model, tokenizer, and device.
    """
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

def get_embeddings(nested_keys, model, tokenizer, device):
    """
    Get the embeddings for the given nested keys using a pre-trained model.

    Args:
        nested_keys (list): A list of nested keys to get embeddings for.
        model: The pre-trained model to use for embeddings.
        tokenizer: The tokenizer corresponding to the model.
        device: The device (CPU or GPU) to perform computations on.

    Returns:
        np.ndarray: An array of embeddings for the nested keys.
    """
    inputs = tokenizer(nested_keys, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def calc_semantic_similarity(nested_keys, model, tokenizer, device):
    """
    Calculate the average semantic similarity of nested keys based on their embeddings

    Args:
        nested_keys (dict): A dictionary where keys are nested keys and values are their types.
        model: The pre-trained model to use for embeddings.
        tokenizer: The tokenizer corresponding to the model.
        device: The device (CPU or GPU) to perform computations on.

    Returns:
        float: The average cosine similarity between the embeddings of the nested keys.
        
    """
    n_keys = len(nested_keys)
    if n_keys < 2:
        return 0.50

    keys = list(nested_keys.keys())
    embeddings = get_embeddings(keys, model, tokenizer, device)
    similarity_matrix = cosine_similarity(embeddings)

    # Exclude diagonal (self-similarity)
    avg_similarity = float((np.sum(similarity_matrix) - n_keys) / (n_keys * (n_keys - 1)))

    return round(avg_similarity, 2)

def tokenize_schema(schema_str, tokenizer):
    """
    Tokenize a schema string using the specified tokenizer.

    Args:
        schema_str (str): A string representation of the schema to tokenize.
        tokenizer (PreTrainedTokenizer): The tokenizer to use (e.g., from HuggingFace).

    Returns:
        dict: A dictionary of tokenized inputs (input_ids, attention_mask, etc.) as torch tensors.
    """
    if not isinstance(schema_str, str):
        raise ValueError("Expected a string schema, got type: {}".format(type(schema_str).__name__))

    return tokenizer(
        schema_str,
        return_tensors="pt",
        max_length=MAX_TOK_LEN,
        padding="max_length",
        truncation=True,
    )

def create_dataframe(path_values, path_freqs, schema_name, path_to_exclude):
    """
    Create a DataFrame of paths and their merged schemas.

    Args:
        path_values (dict): Dictionary of paths and their serialized JSON values.
        path_freqs (dict): Dictionary of paths and their frequencies.
        schema_name (str): Dataset filename.
        path_to_exclude (set): Paths to exclude.

    Returns:
        pd.DataFrame: DataFrame with tokenized schemas and metadata.
    """
    data = []
    model, tokenizer, device = load_model()

    for path, nested_keys in path_values.items():
        if path in path_to_exclude:
            continue

        schema_info = {"properties": {}}
        values_types = set()
        frequencies = []

        parent_frequency = path_freqs.get(path, 0)
        observed_keys = set(nested_keys.keys())
        required_keys = []

        for nested_key, nested_key_info in nested_keys.items():
            frequency = nested_key_info["frequency"] / parent_frequency if parent_frequency else 0
            value_type = nested_key_info["type"]

            # Convert set to list if necessary
            if isinstance(value_type, set):
                value_type = list(value_type)

            values_types.add(json.dumps(value_type))  # for datatype_entropy calculation
            frequencies.append(frequency)

            schema_info["properties"][nested_key] = {
                "frequency": round(frequency, 3),
                "type": value_type  # now always JSON-serializable
            }

            # Required if appears in every parent occurrence
            if nested_key_info["frequency"] == parent_frequency:
                required_keys.append(nested_key)

        # Compute additionalProperties
        additional_properties = len(observed_keys - set(required_keys)) > 0

        schema_info["required"] = required_keys
        schema_info["additionalProperties"] = additional_properties

        # Derived info
        schema_info["nesting_depth"] = len(path) - 1
        schema_info["datatype_entropy"] = 0 if len(values_types) == 1 else 1
        schema_info["num_nested_keys"] = len(nested_keys)
        schema_info["semantic_similarity"] = calc_semantic_similarity(nested_keys, model, tokenizer, device)

        # Key entropy
        if frequencies:
            total_freq = sum(frequencies)
            probs = [f / total_freq for f in frequencies if f > 0]
            key_entropy = -sum(p * math.log(p) for p in probs)
        else:
            key_entropy = 0

        # Tokenize schema
        tokenized_schema = tokenize_schema(json.dumps(schema_info), tokenizer)

        data.append({
            "path": path,
            "schema": json.dumps(schema_info, sort_keys=True),
            "tokenized_schema": tokenized_schema,
            "datatype_entropy": schema_info["datatype_entropy"],
            "key_entropy": round(key_entropy, 2),
            "parent_frequency": parent_frequency,
            "filename": schema_name
        })

    df = pd.DataFrame(data)
    return df

def calculate_embeddings(df, model, device):
    """
    Calculate the embeddings for each schema in batches.

    Args:
        df (pd.DataFrame): DataFrame containing paths and their tokenized schemas.
        model (AutoModel): Pretrained model to compute embeddings.
        device (torch.device): Device to move tensors to.

    Returns:
        dict: Dictionary mapping paths (tuples) to their corresponding embeddings.
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

def get_samples(df, frequent_ref_defn_paths, best_good_pairs=True):
    """
    Generate labeled samples of good and bad pairs from the DataFrame based on ground truth definitions.

    Args:
        df (pd.DataFrame): DataFrame containing paths and schemas (schemas as JSON strings).
        frequent_ref_defn_paths (dict): Dictionary of frequent referenced definitions and their paths.
        best_good_pairs (bool, optional): Whether to select the best good pairs. Defaults to True.

    Returns:
        tuple: (sample_good_pairs, sample_bad_pairs)
    """
    all_good_pairs = set()
    all_bad_pairs = set()
    all_good_paths = set()
    sample_good_pairs = set()
    sample_bad_pairs = set()

    # Load model and compute embeddings
    model, tokenizer, device = load_model()
    schema_embeddings = calculate_embeddings(df, model, device)

    paths = list(df["path"])

    # --- Process good pairs ---
    for ref_defn, good_paths in tqdm(frequent_ref_defn_paths.items(), desc="Processing good pairs", position=0, leave=False):
        all_good_paths.update(good_paths)
        good_pairs = list(combinations(good_paths, 2))
        all_good_pairs.update(good_pairs)

        if best_good_pairs:
            good_pairs_distances = [
                ((p1, p2), 1 - cosine(schema_embeddings[p1], schema_embeddings[p2]))
                for p1, p2 in good_pairs
            ]
            top_1000_good_pairs = nlargest(1000, good_pairs_distances, key=lambda x: x[1])
            sample_good_pairs.update(pair for pair, _ in top_1000_good_pairs)
        else:
            sample_good_pairs.update(good_pairs[:1000])

    # --- Prepare for bad pairs ---
    good_schemas = set(df.loc[df["path"].isin(all_good_paths), "schema"])
    path_to_schema = df.set_index("path")["schema"].to_dict()

    # --- Process bad pairs ---
    all_pairs = list(combinations(paths, 2))
    for p1, p2 in all_pairs:
        if ((p1, p2) not in all_good_pairs and (p2, p1) not in all_good_pairs):
            schema1_good = path_to_schema.get(p1) in good_schemas
            schema2_good = path_to_schema.get(p2) in good_schemas

            if not (schema1_good and schema2_good):
                all_bad_pairs.add((p1, p2))

    # Calculate distances for bad pairs
    bad_pairs_distances = [
        ((p1, p2), 1 - cosine(schema_embeddings[p1], schema_embeddings[p2]))
        for p1, p2 in all_bad_pairs
    ]

    # Sample bad pairs equal to the number of good pairs sampled
    num_bad_pairs = min(len(sample_good_pairs), len(bad_pairs_distances))
    top_bad_pairs = nsmallest(num_bad_pairs, bad_pairs_distances, key=lambda x: x[1])
    sample_bad_pairs.update(pair for pair, _ in top_bad_pairs)

    return sample_good_pairs, sample_bad_pairs

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
    schemas1 = [] 
    schemas2 = []  
    filenames = []  

    # Process good pairs: label them as 1 (positive)
    for path1, path2 in good_pairs:
        paths1.append(path1)
        paths2.append(path2)
        labels.append(1)

        # Extract schemas and filename for both paths
        path1_row = df[df["path"] == path1].iloc[0]
        path2_row = df[df["path"] == path2].iloc[0]
        filenames.append(path1_row["filename"])
        schemas1.append(path1_row["schema"])
        schemas2.append(path2_row["schema"])

    # Process bad pairs: label them as 0 (negative)
    for path1, path2 in bad_pairs:
        paths1.append(path1)
        paths2.append(path2)
        labels.append(0)

        # Extract schemas and filename for both paths
        path1_row = df[df["path"] == path1].iloc[0]
        path2_row = df[df["path"] == path2].iloc[0]
        filenames.append(path1_row["filename"])
        schemas1.append(path1_row["schema"])
        schemas2.append(path2_row["schema"])

    # Create a new DataFrame with separate columns for path1 and path2
    labeled_df = pd.DataFrame({
        "filename": filenames,
        "label": labels,
        "path1": paths1,
        "path2": paths2,
        "schema1": schemas1,
        "schema2": schemas2
    })

    return labeled_df



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
        if len(set(paths) & set(df["path"])) > 1
    }

def process_schema(schema_name, schema_folder):
    """
    Process a single schema safely in a spawned worker.
    Returns DataFrame as records, updated reference definitions, failure flags, and valid docs.
    """
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

    try:
        schema_path = os.path.join(schema_folder, schema_name)

        schema = load_schema(schema_path)
        if schema is None:
            failure_flags["load"] = 1
            return None, {}, failure_flags, []

        ref_defn_paths = clean_ref_defn_paths(schema)
        if not ref_defn_paths:
            failure_flags["ref_defn"] = 1
            return None, {}, failure_flags, []

        cleaned_ref_defn_paths = remove_definition_keywords(handle_nested_definitions(ref_defn_paths))
        path_to_exclude = get_ref_defn_of_type_obj(schema, cleaned_ref_defn_paths)
        if not cleaned_ref_defn_paths:
            failure_flags["object_defn"] = 1
            return None, {}, failure_flags, []

        path_values, path_freqs, valid_docs = process_dataset(schema_name, path_to_exclude)
        if len(path_values) == 0:
            failure_flags["path"] = 1
            return None, {}, failure_flags, valid_docs

        filtered_ref_defn_paths = check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, path_values)
        if not filtered_ref_defn_paths:
            failure_flags["schema_intersection"] = 1
            return None, {}, failure_flags, valid_docs

        frequent_ref_defn_paths = find_frequent_definitions(filtered_ref_defn_paths, path_to_exclude)
        if not frequent_ref_defn_paths:
            failure_flags["freq_defn"] = 1
            return None, {}, failure_flags, valid_docs

        df = create_dataframe(path_values, path_freqs, schema_name, path_to_exclude)
        updated_ref_defn_paths = update_ref_defn_paths(frequent_ref_defn_paths, df)
        if not updated_ref_defn_paths:
            failure_flags["properties"] = 1
            return None, {}, failure_flags, valid_docs

        # Make everything pickle-safe
        updated_ref_defn_paths = {k: list(v) for k, v in updated_ref_defn_paths.items()}

        return df, updated_ref_defn_paths, failure_flags, valid_docs

    except Exception as e:
        print(f"Worker crashed on {schema_name}: {e}", flush=True)
        return None, {}, failure_flags, []

def save_valid_docs(matched_jsons_dir, dataset_name, valid_docs):
    """
    Save valid documents to a JSONL file.
    """
    os.makedirs(matched_jsons_dir, exist_ok=True)

    output_path = os.path.join(matched_jsons_dir, dataset_name)

    with open(output_path, 'w') as outfile:
        for doc in valid_docs:
            json.dump(doc, outfile)
            outfile.write('\n')

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

def find_ref_defn_for_path(path, merged_defn_paths):
    """
    Given a path string, find which reference definition it belongs to.
    
    Args:
        path (str): A JSONSchema path string.
        merged_defn_paths (dict): Mapping ref_defn -> set(paths)
    
    Returns:
        str or None: The reference definition name, or None if not found.
    """
    for ref_defn, paths in merged_defn_paths.items():
        if path in paths:
            return ref_defn
    return None




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


def preprocess_data(schemas, filename, ground_truth_file, schema_folder):
    """
    Sequentially process all schemas and track failure flags.
    """
    ground_truths = defaultdict(dict)

    # Counters for different failure types
    load_count = ref_defn = object_defn = path = schema_intersection = freq_defn = properties = 0

    for schema_name in tqdm(schemas, desc="Processing schemas"):
        try:
            df, frequent_ref_defn_paths, failure_flags, valid_docs = process_schema(schema_name, schema_folder)

            # Update counters
            load_count += failure_flags.get("load", 0)
            ref_defn += failure_flags.get("ref_defn", 0)
            object_defn += failure_flags.get("object_defn", 0)
            path += failure_flags.get("path", 0)
            schema_intersection += failure_flags.get("schema_intersection", 0)
            freq_defn += failure_flags.get("freq_defn", 0)
            properties += failure_flags.get("properties", 0)

            if df is None or len(frequent_ref_defn_paths) == 0:
                continue
    
            # Save ground truths
            ground_truths[schema_name] = frequent_ref_defn_paths

            # Sample good and bad pairs
            sample_good_pairs, sample_bad_pairs = get_samples(df, frequent_ref_defn_paths, best_good_pairs=True)
            labeled_df = label_samples(df, sample_good_pairs, sample_bad_pairs)

            # Append batch to CSV to avoid holding everything in memory
            labeled_df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False, sep=';')


            # Save valid JSON docs if this is test data
            matched_jsons_dir = "train_jsons" if "train" in filename else "test_jsons"
            if matched_jsons_dir == "test_jsons":
                save_valid_docs(matched_jsons_dir, schema_name, valid_docs)

        except Exception as e:
            print(f"Error processing schema {schema_name}: {e}", flush=True)
        

    # Save results
    save_ground_truths(ground_truths, ground_truth_file)

    # ---------- Print failure summary ----------
    total_schemas = len(schemas)
    print("Total schemas processed:", total_schemas)
    print("Schemas that loaded:", total_schemas - load_count)
    print("Schemas with referenced definitions:", total_schemas - load_count - ref_defn)
    print("Schemas with object definitions:", total_schemas - load_count - ref_defn - object_defn)
    print("Schemas with paths:", total_schemas - load_count - ref_defn - object_defn - path)
    print("Schemas with schema intersection:", total_schemas - load_count - ref_defn - object_defn - path - schema_intersection)
    print("Schemas with frequent definitions:", total_schemas - load_count - ref_defn - object_defn - path - schema_intersection - freq_defn)
    print("Schemas with properties:", total_schemas - load_count - ref_defn - object_defn - path - schema_intersection - freq_defn - properties)


# ---------- Main Entry Point ----------
def main():
    start_time = time.time()

    try:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Process data for training and testing.")
        parser.add_argument("train_size", type=float, help="Training set size ratio (0 < train_size < 1)")
        parser.add_argument("random_value", type=int, help="Random seed value")
        args = parser.parse_args()

        train_ratio = args.train_size
        random_value = args.random_value

        # Split the data into training and testing sets
        train_set, test_set = split_data(train_ratio, random_value)

        # Files to delete if they exist
        files_to_delete = [
            "train_data.csv",
            "train_ground_truth.json",
            "test_data.csv",
            "test_ground_truth.json"
        ]
        for file in files_to_delete:
            delete_file_if_exists(file)

        # ---------- Preprocess training data ----------
        print("Preprocessing training data...")
        preprocess_data(
            schemas=train_set,
            filename="train_data.csv",
            ground_truth_file="train_ground_truth.json",
            schema_folder=SCHEMA_FOLDER
        )

        # ---------- Preprocess testing data ----------
        print("Preprocessing testing data...")
        preprocess_data(
            schemas=test_set,
            filename="test_data.csv",
            ground_truth_file="test_ground_truth.json",
            schema_folder=SCHEMA_FOLDER
        )

        # End the timer
        elapsed_time = time.time() - start_time
        print(f"Time taken for preprocessing: {elapsed_time:.2f} seconds")

    except (ValueError, IndexError) as e:
        print(f"Error: {e}\nUsage: script.py <train_size> <random_value>")
        sys.exit(1)


if __name__ == "__main__":
    # Disable HuggingFace tokenizers parallelism if using tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

