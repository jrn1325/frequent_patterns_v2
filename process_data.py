import ast
import concurrent.futures
import hashlib
import itertools
import json
import jsonschema
import multiprocessing
import os
import pandas as pd
import random
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
from itertools import combinations
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
BATCH_SIZE = 64
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
        '''
        # Detect and skip circular references
        defn = ref_parts[-2] + '.' + ref_parts[-1]
        path_str = '.'.join(path)
        if defn in path_str:
            continue
        '''
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


def process_document(doc, paths_dict, path_freqs):
    """
    Extracts object-like paths from the given JSON document and stores them in dictionaries,
    grouping values by paths and tracking path frequency.

    Args:
        doc (dict): The JSON document from which paths are extracted.
        paths_dict (dict): Dictionary mapping each path to a list of values (as JSON strings).
        path_freqs (Counter): Dictionary tracking how often each path appears.
    """
    for path, value in parse_document(doc):
        path_freqs[path] += 1

        if path not in paths_dict:
            paths_dict[path] = []

        if isinstance(value, dict):
            value_str = json.dumps(value, sort_keys=True)
            paths_dict[path].append(value_str)

        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
            sorted_list = sorted(
                [json.dumps(item, sort_keys=True) for item in value]
            )
            value_str = json.dumps(sorted_list, sort_keys=True)
            paths_dict[path].append(value_str)


def get_doc_hash(doc):
    """Return a hash of the canonical JSON form of the document."""
    try:
        canonical = json.dumps(doc, sort_keys=True)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    except Exception:
        return None


def process_dataset(dataset):
    """
    Process and extract object-like paths from a JSON lines dataset, filtering documents
    by matching schema properties and ignoring duplicate documents.

    Returns:
        tuple: (paths_dict, num_docs)
    """
    paths_dict = defaultdict(list)
    path_freqs = Counter()
    num_docs = 0
    seen_hashes = set()

    schema_path = os.path.join(SCHEMA_FOLDER, dataset)
    schema = load_schema(schema_path)

    dataset_path = os.path.join(JSON_FOLDER, dataset)
    with open(dataset_path, 'r') as file:
        for line_number, line in enumerate(file, 1):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Line {line_number}] JSON decode error in {dataset}: {e}", flush=True)
                continue

            try:
                matched = False
                items = doc if isinstance(doc, list) else [doc]
                for item in items:
                    if not isinstance(item, dict):
                        continue

                    doc_hash = get_doc_hash(item)
                    if doc_hash is None or doc_hash in seen_hashes:
                        continue
                    seen_hashes.add(doc_hash)

                    if match_properties(schema, item):
                        process_document(item, paths_dict, path_freqs)
                        matched = True

                if matched:
                    num_docs += 1

            except Exception as e:
                print(f"[Line {line_number}] Error processing document in {dataset}: {e}", flush=True)

    # Remove paths that occur only once
    paths_dict = {
        path: values for path, values in paths_dict.items()
        if path_freqs[path] > 1
    }

    return paths_dict, num_docs


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

    for ref_defn, schema_paths in tqdm(cleaned_ref_defn_paths.items(), desc="Referenced definitions", total=len(cleaned_ref_defn_paths)):
        # Check matching schema and json paths
        for schema_path in schema_paths:
            if schema_path in paths_dict:
                filtered_ref_defn_paths[ref_defn].add(schema_path)

    return filtered_ref_defn_paths
'''
def check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, paths_dict):
    """
    Check if the paths from JSON Schemas exist in JSON datasets, ensuring they conform to the schema definition.

    Args:
        cleaned_ref_defn_paths (dict): Dictionary of referenced definitions and their paths.
        paths_dict (dict): Dictionary of paths and their corresponding values.

    Returns:
        dict: Dictionary of definitions with paths that exist and conform in the JSON documents.
    """
    filtered_ref_defn_paths = defaultdict(set)

    for ref_defn, schema_paths in tqdm(cleaned_ref_defn_paths.items(), desc="Number of referenced definitions", position=0, leave=False, total=len(cleaned_ref_defn_paths)):

        for schema_path in tqdm(schema_paths, desc="Number of paths for a ref_defn", position=1, leave=False, total=len(schema_paths)):
            for json_path in tqdm(paths_dict.keys(), desc="Number of paths in JSON files", position=2, leave=False, total=len(paths_dict)):
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


def flatten_oneOf(schema):
    if "oneOf" in schema:
        result = []
        for s in schema["oneOf"]:
            if isinstance(s, dict) and "oneOf" in s:
                result.extend(flatten_oneOf(s)["oneOf"])
            else:
                result.append(s)
        return {"oneOf": unique_oneOf(result)}
    return schema


def merge_schemas(schema1, schema2, depth=0, max_depth=10):
    if depth > max_depth:
        raise RecursionError(f"Exceeded max recursion depth of {max_depth}")
    
    if not isinstance(schema1, dict) or not isinstance(schema2, dict):
        raise TypeError("Both schema1 and schema2 must be dictionaries.")
    
    schema1 = flatten_oneOf(schema1)
    schema2 = flatten_oneOf(schema2)
    
    type1 = schema1.get("type")
    type2 = schema2.get("type")

    if type1 == type2:
        new_schema = deepcopy(schema1)

        if type1 == "object":
            new_schema["required"] = sorted(set(schema1.get("required", [])) | set(schema2.get("required", [])))
            for k, v in schema2.get("properties", {}).items():
                if k in new_schema["properties"]:
                    new_schema["properties"][k] = merge_schemas(new_schema["properties"][k], v, depth + 1)
                else:
                    new_schema["properties"][k] = v

            return new_schema

        elif type1 == "array" and "items" in schema1 and "items" in schema2:
            new_schema["items"] = merge_schemas(schema1["items"], schema2["items"], depth + 1)
            return new_schema

        return new_schema

    # Flatten and merge into oneOf
    return flatten_oneOf({"oneOf": [schema1, schema2]})


def discover_schema(value, max_depth=2, current_depth=0):
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
    if not values:
        return {"type": "null"}
    schemas = [discover_schema(v) for v in values]
    return reduce(merge_schemas, schemas)


def calculate_nested_keys_frequency(values, num_docs):
    """
    Calculate frequency of each top-level key relative to total number of documents.

    Args:
        values (list): List of dicts or list-of-dicts at the path.
        num_docs (int): Total number of documents in the dataset.

    Returns:
        dict: {key: relative_frequency} where frequency is fraction of total documents
              where path and key co-occur.
    """
    from collections import Counter

    key_presence_counter = Counter()

    for val in values:
        dicts_to_check = val if isinstance(val, list) else [val]
        keys_in_doc = set()
        for d in dicts_to_check:
            if not isinstance(d, dict):
                continue
            keys_in_doc.update(d.keys())
        for key in keys_in_doc:
            key_presence_counter[key] += 1

    if num_docs == 0:
        return {}

    return {k: v / num_docs for k, v in key_presence_counter.items()}


def create_dataframe(paths_dict, paths_to_exclude, num_docs):
    """
    Create a DataFrame of paths and their merged schemas.

    Args:
        paths_dict (dict): Dictionary of paths and their serialized JSON values.
        paths_to_exclude (set): Paths to exclude.
        num_docs (int): Total number of documents processed (for path frequency computation).

    Returns:
        pd.DataFrame: DataFrame with tokenized schemas and metadata.
    """
    df_data = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for path, values in tqdm(paths_dict.items(), desc="Creating dataframe", total=len(paths_dict), leave=False):
        if path in paths_to_exclude:
            continue

        # Full list: needed for frequency analysis
        parsed_values = [json.loads(v) for v in values]

        # Deduplicated: needed for efficient schema inference
        unique_value_strs = list(set(values))
        unique_parsed_values = [json.loads(v) for v in unique_value_strs]

        try:
            schema = discover_schema_from_values(unique_parsed_values)
        except Exception as e:
            print(f"Error discovering schema for path {path}: {e}", flush=True)
            paths_to_exclude.add(path)
            continue

        if len(schema.get("properties", {})) >= 1:
            nested_freqs = calculate_nested_keys_frequency(parsed_values, num_docs)
            #print(f"Processing path: {path}", flush=True)
            #print(f"Nested key freqs: {nested_freqs}", flush=True)

            tokenized_schema = tokenize_schema(json.dumps(schema), tokenizer)

            df_data.append([
                path,
                len(path),
                nested_freqs,
                tokenized_schema,
                json.dumps(schema)
            ])
        else:
            paths_to_exclude.add(path)

    columns = ["path", "nesting_depth", "nested_keys", "tokenized_schema", "schema"]
    return pd.DataFrame(df_data, columns=columns)




def create_dataframe_baseline_model(paths_dict, paths_to_exclude, num_docs):
    """
    Create a DataFrame of paths and distinct nested keys.

    Args:
        paths_dict (dict): Dictionary of paths and their nested keys.
        paths_to_exclude (set): Paths to exclude from JSON files.

    Returns:
        pd.DataFrame: DataFrame containing paths and distinct nested keys.
    """
    df_data = []

    for path, values in tqdm(paths_dict.items(), desc="Processing paths", total=len(paths_dict), leave=False):
        # Skip paths that are in the exclusion set
        if path in paths_to_exclude:
            continue

        # Parse and discover the schema from values
        parsed_values = [json.loads(v) for v in values]
       
        nested_keys = calculate_nested_keys_frequency(parsed_values, num_docs)
        if len(nested_keys) == 0:
            paths_to_exclude.add(path)
            continue
        df_data.append([path, nested_keys])

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
        if len(set(paths) & set(df["path"])) > 1
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
    
    # Handle nested definitions
    new_ref_defn_paths = handle_nested_definitions(ref_defn_paths)
    cleaned_ref_defn_paths = remove_definition_keywords(new_ref_defn_paths)
   
    # Get referenced definitions of type object
    paths_to_exclude = get_ref_defn_of_type_obj(schema, cleaned_ref_defn_paths)
    if not cleaned_ref_defn_paths:
        failure_flags["object_defn"] = 1
        #print(f"No referenced definitions of type object in {schema_name}.", flush=True)
        return None, None, schema_name, failure_flags
    
    paths_dict, num_docs = process_dataset(schema_name)
    if len(paths_dict) == 0:
        failure_flags["path"] = 1
        #print(f"No paths extracted from {schema_name}.", flush=True)
        return None, None, schema_name, failure_flags

    # Check reference definition paths in the dataset
    filtered_ref_defn_paths = check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, paths_dict)
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
        df = create_dataframe_baseline_model(paths_dict, paths_to_exclude, num_docs)
    else:
        df = create_dataframe(paths_dict, paths_to_exclude, num_docs)
        print(f"Number of paths in {schema_name}: {len(df)}", flush=True)

    # Update reference definitions
    updated_ref_defn_paths = update_ref_defn_paths(frequent_ref_defn_paths, df)
    if not updated_ref_defn_paths:
        failure_flags["properties"] = 1
        print(f"Not enough properties found under refererenced definitions in {schema_name}.", flush=True)
        return None, None, schema_name, failure_flags

    df["filename"] = schema_name
    df.reset_index(drop=True, inplace=True)
    
    return df, updated_ref_defn_paths, schema_name, failure_flags





def load_model():
    """
    Function to load and return the model, tokenizer, and device.
    """
    MODEL_NAME = "microsoft/codebert-base"
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def tokenize_schema(schema_str, tokenizer, max_length=512):
    """
    Tokenize a schema string using the specified tokenizer.

    Args:
        schema_str (str): A string representation of the schema to tokenize.
        tokenizer (PreTrainedTokenizer): The tokenizer to use (e.g., from HuggingFace).
        max_length (int, optional): Maximum sequence length. Defaults to 512.

    Returns:
        dict: A dictionary of tokenized inputs (input_ids, attention_mask, etc.) as torch tensors.
    """
    if not isinstance(schema_str, str):
        raise ValueError("Expected a string schema, got type: {}".format(type(schema_str).__name__))

    return tokenizer(
        schema_str,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )


def calculate_embeddings(df, model, device):
    """
    Calculate the embeddings for each schema in batches.

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
    nested_keys1 = []
    nested_keys2 = []
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
        nested_keys1.append(path1_row["nested_keys"])
        nested_keys2.append(path2_row["nested_keys"])
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
        nested_keys1.append(path1_row["nested_keys"])
        nested_keys2.append(path2_row["nested_keys"])
        filenames.append(path1_row["filename"])
        schemas1.append(path1_row["schema"])
        schemas2.append(path2_row["schema"])

    # Create a new DataFrame with separate columns for path1 and path2
    labeled_df = pd.DataFrame({
        "filename": filenames,
        "label": labels,
        "path1": paths1,
        "path2": paths2,
        "nested_keys1": nested_keys1,
        "nested_keys2": nested_keys2,
        "schema1": schemas1,
        "schema2": schemas2
    })

    return labeled_df


def sample_path_pairs(paths, rng=None, k=100000):
    """
    Efficiently sample k unique unordered pairs from paths, optionally using a seeded RNG.

    Args:
        paths (list): List of paths to sample from.
        k (int): Number of pairs to sample.
        rng (random.Random, optional): Custom random generator for reproducibility.

    Returns:
        list: List of unique unordered (sorted) path pairs.
    """
    rng = rng or random
    n = len(paths)
    if n < 2:
        return []

    max_possible = n * (n - 1) // 2
    k = min(k, max_possible)

    if k > max_possible * 0.6:
        all_pairs = list(combinations(paths, 2))
        return rng.sample(all_pairs, k)

    seen = set()
    pairs = []
    attempts = 0
    max_attempts = k * 10

    while len(pairs) < k and attempts < max_attempts:
        p1, p2 = rng.sample(paths, 2)
        pair = tuple(sorted((p1, p2)))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
        attempts += 1

    return pairs


def get_samples(df, frequent_ref_defn_paths, best_good_pairs=False):
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

    model, device = load_model()
    schema_embeddings = calculate_embeddings(df, model, device)
    paths = list(df["path"])

    for ref_defn, good_paths in tqdm(frequent_ref_defn_paths.items(), desc="Processing good pairs", position=0, leave=False):
        all_good_paths.update(good_paths)
        good_pairs = list(itertools.combinations(good_paths, 2))
        path_to_keys = {path: set(df.loc[df["path"] == path, "nested_keys"].values[0]) for path in good_paths}

        all_good_pairs.update(good_pairs)
        good_pairs = [(p1, p2) for p1, p2 in good_pairs if len(path_to_keys[p1] & path_to_keys[p2]) >= 2]

        if best_good_pairs:
            good_pairs_distances = [
                ((p1, p2), cosine(schema_embeddings[p1], schema_embeddings[p2]))
                for p1, p2 in good_pairs
            ]
            top_1000_good_pairs = nlargest(1000, good_pairs_distances, key=lambda x: x[1])
            sample_good_pairs.update(pair for pair, _ in top_1000_good_pairs)
        else:
            sample_good_pairs.update(good_pairs[:1000])

    good_schemas = set(df.loc[df["path"].isin(all_good_paths), "schema"])
    path_to_schema = df.set_index("path")["schema"].to_dict()

    # Efficiently sample path pairs
    rng = random.Random(42)  # Create a reproducible RNG
    sampled_pairs = sample_path_pairs(paths, rng)

    for p1, p2 in sampled_pairs:
        if ((p1, p2) not in all_good_pairs and (p2, p1) not in all_good_pairs):
            schema1_good = path_to_schema.get(p1) in good_schemas
            schema2_good = path_to_schema.get(p2) in good_schemas
            keys1 = set(df.loc[df["path"] == p1, "nested_keys"].values[0].keys())
            keys2 = set(df.loc[df["path"] == p2, "nested_keys"].values[0].keys())

            if not (schema1_good and schema2_good) and len(keys1 & keys2) >= 1:
                all_bad_pairs.add((p1, p2))


    bad_pairs_distances = [
        ((p1, p2), cosine(schema_embeddings[p1], schema_embeddings[p2]))
        for p1, p2 in all_bad_pairs
    ]

    num_bad_pairs = min(len(sample_good_pairs), len(bad_pairs_distances))
    top_bad_pairs = nsmallest(num_bad_pairs, bad_pairs_distances, key=lambda x: x[1])
    sample_bad_pairs.update(pair for pair, _ in top_bad_pairs)

    return label_samples(df, sample_good_pairs, sample_bad_pairs)






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
        train_size, random_value, model = sys.argv[-3:]
        train_ratio = float(train_size)
        random_value = int(random_value)


        # Split the data into training and testing sets
        train_set, test_set = split_data(train_ratio, random_value)

        if model == "baseline":
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
            #preprocess_data(test_set, filename="test.csv", ground_truth_file="test.json")
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