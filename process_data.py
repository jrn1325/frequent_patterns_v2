import dask.dataframe as dd
import itertools
import json
import jsonref
import jsonschema
import os
import pandas as pd
import re
import sys
import torch
import torch.nn.functional as F
import tqdm
import warnings


from adapters import AutoAdapterModel
from collections import defaultdict, Counter
import concurrent.futures
from copy import copy, deepcopy
from functools import reduce
from jsonschema import ValidationError
from jsonschema.validators import validator_for
from sklearn.model_selection import GroupShuffleSplit
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer



warnings.filterwarnings("ignore")
sys.setrecursionlimit(30000) # I had to increase the recursion limit because of the discover_schema function

# Create constant variables
DISTINCT_SUBKEYS_UPPER_BOUND = 1000
RANDOM_VALUE = 101
TRAIN_RATIO = 0.8
COMPLEX_PROPERTIES_KEYWORD = {"patternProperties"}
DEFINITION_KEYWORDS = {"$defs", "definitions"}
JSON_SCHEMA_KEYWORDS = {"properties", "allOf", "oneOf", "anyOf", "not", "if", "then", "else"}
JSON_SUBSCHEMA_KEYWORDS = {"allOf", "oneOf", "anyOf"}

SCHEMA_FOLDER = "processed_schemas"
JSON_FOLDER = "processed_jsons"
MAX_TOK_LEN = 512
BATCH_SIZE = 32

MODEL_NAME = "microsoft/codebert-base" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
m = AutoAdapterModel.from_pretrained(MODEL_NAME)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m.to(device)




def split_data(train_ratio=0.8, random_value=101):
    """
    Split the list of schemas into training and testing sets making sure that schemas from the same source are grouped together.

    Args:
        train_ratio (float, optional): The ratio of schemas to use for training. Defaults to 0.8.
        random_value (int, optional): The random seed value. Defaults to 101.

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
            print(f"Error loading schema {schema_path}: {e}")
            return None
        

def find_ref_paths(schema, current_path=('$',)):
    """
    Find the full paths of keys with $refs in the JSON schema, replacing 'items' with '*'
    in array paths, and handle nested structures.

    Args:
        schema (dict): JSON schema object.
        current_path (tuple, optional): Path of the referenced definition. Defaults to ('$',).

    Yields:
        generator: full path where $ref is found.
    """
    if not isinstance(schema, dict):
        return

    for key, value in schema.items():
        # Replace 'items' with '*' in the path
        updated_path = current_path + ('*',) if key == "items" else current_path + (key,)
        
        # If the current key is $ref, yield the path and its value (the reference string)
        if key == "$ref" and isinstance(value, str):
            yield (value, current_path)

        # Handle additional properties
        if key == "additionalProperties":
            updated_path = current_path + ("wildkey",)
            yield from find_ref_paths(value, updated_path)
        
        # Recursively search if the value is another dictionary
        elif isinstance(value, dict):
            yield from find_ref_paths(value, updated_path)

        # Handle lists in case $ref exists inside arrays
        elif isinstance(value, list):
            for i, item in enumerate(value):
                yield from find_ref_paths(item, updated_path)


def clean_ref_defn_paths(schema):
    """
    Remove keywords associated with JSON Schema that do not exist in JSON documents' format.
  

    Args:
        schema (dict): JSON Schema object.

    Returns:
        dict: dictionary of cleaned JSON paths without schema-specific keywords.
    """
    ref_defn_paths = defaultdict(set)

    # Loop through referenced definitions and their corresponding paths
    for ref, path in find_ref_paths(schema):
        # Prevent circular references (if the reference name is already part of the path)
        if ref.split('/')[-1] in path:
            continue

        # If the reference is a URL, skip it
        if ref[0] != '#':
            continue

        # Remove JSON Schema keywords from the paths, handle complex properties
        cleaned_path = []
        for key in path:
            if key not in JSON_SCHEMA_KEYWORDS:
                cleaned_path.append(key)

        # Convert the cleaned path to a tuple and add it to the dictionary
        cleaned_path = tuple(cleaned_path)
        # Skip paths containing complex properties keywords
        if any(keyword in cleaned_path for keyword in COMPLEX_PROPERTIES_KEYWORD):
            continue

        ref_defn_paths[ref].add(cleaned_path)

    return dict(sorted(ref_defn_paths.items()))


def resolve_paths(path, ref_defn_paths, max_depth=10, current_depth=0):
    """Resolve all the nested definitions to find them in the json file format

    Args:
        path (tuple): full path of key
        ref_defn_paths (dict): dictionary of reference definitions and their paths.
        max_depth (int): maximum recursion depth
        current_depth (int): current recursion depth

    Returns:
        list: list of paths
    """
    if current_depth >= max_depth:
        # Reached the maximum recursion depth, return an empty list
        return []
    
    resolved_paths = set()
    paths = ref_defn_paths.get(path, set())
    for sub_path in paths:
        # Check if the sub_path contains enough elements
        if len(sub_path) >= 3:
            defn_keyword = sub_path[1]
            defn_name = sub_path[2]
            # Check if "definitions" or "$defs" exists in the sub_path
            if defn_keyword in DEFINITION_KEYWORDS:
                referenced_defn_path = "#/" + defn_keyword + '/' + defn_name
                # Resolve paths of the referenced definition recursively with increased depth
                for top_defn_path in resolve_paths(referenced_defn_path, ref_defn_paths, max_depth, current_depth + 1):
                    new_path = top_defn_path + sub_path[3:]
                    resolved_paths.add(new_path)
            else:
                resolved_paths.add(sub_path)
        else:
            resolved_paths.add(sub_path)
    
    return resolved_paths


def handle_nested_definitions(ref_defn_paths): 
    """Apply the resolve funtions to all the paths of referenced definitions

    Args:
        ref_defn_paths (dict): dictionary of reference definitions and their paths.

    Returns:
        dict: dictionary of referenced paths withous dependencies
    """
    new_ref_defn_paths = defaultdict(set)

    # Resolve paths for each reference definition
    for ref_defn in ref_defn_paths.keys():
        resolved_paths = resolve_paths(ref_defn, ref_defn_paths)
        if resolved_paths:
            new_ref_defn_paths[ref_defn] = resolved_paths

    return new_ref_defn_paths


def remove_definition_keywords(good_ref_defn_paths):
    """Remove the keywords "definitions" and "defs" from all paths in the dictionary.

    Args:
        good_ref_defn_paths (dict): Dictionary of definition names and their paths of references.

    Returns:
        dict: Dictionary with "definitions" and "defs" removed from all paths.
    """
    updated_paths = {}

    for defn_name, paths in good_ref_defn_paths.items():
        updated_paths[defn_name] = [
            tuple(part for part in path if part not in DEFINITION_KEYWORDS) for path in paths
        ]

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
    for keyword in ["oneOf", "anyOf", "allOf"]:
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
            print(f"Excluded {ref}: Not object-like or does not have more than one property.")

    # Remove excluded references and update paths to exclude
    for ref in ref_to_delete:
        paths_to_exclude.update(ref_defn_paths.pop(ref))

    return paths_to_exclude


def match_properties(schema, document):
    """Check if there is an intersection between the schema properties and the document.

    Args:
        schema (dict): JSON Schema object.
        document (dict): JSON object.

    Returns:
        bool: True if there is a match, False otherwise.
    """
    # Check if the schema has 'properties'
    schema_properties = schema.get("properties", {})

    # Check for matching properties from 'properties' or 'patternProperties'
    matching_properties_count = sum(1 for key in document if key in schema_properties)
    
    # If there are matching properties, return True
    if matching_properties_count > 0:
        return True

    # Return False if no match is found
    return False


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


def get_json_type(value):
    """
    Map Python types to their corresponding JSON types.
    
    Args:
        value (any): The Python value to map to JSON type.
    
    Returns:
        str: Corresponding JSON type.
    """
    if isinstance(value, dict):
        return "object"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, str):
        return "string"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, (int, float)):
        return "number"
    elif value is None:
        return "null"
    else:
        return "object"  # For any other type, we return "object"
    

def process_document(doc, paths_dict):
    """
    Extracts paths whose values are object-type from the given JSON document and stores them in dictionaries,
    grouping values by paths.

    Args:
        doc (dict): The JSON document from which paths are extracted.
        paths_dict (dict): Dictionary of paths and their values.
    """
    for path, value in parse_document(doc):
        if isinstance(value, dict):
            value_str = json.dumps(value)
            paths_dict[path].add(value_str)
        elif isinstance(value, list):
            if all(isinstance(item, dict) for item in value):
                value_str = json.dumps(value)
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
                print(f"Error decoding line in {dataset}: {e}")
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
                print(f"Error processing line in {dataset}: {e}")
                continue

    return paths_dict if paths_dict else None
    

def path_matches_with_wildkey(schema_path, json_path):
    """
    Check if a schema path with 'wildkey' matches a JSON path.
    
    Args:
        schema_path (list): List of keys in the schema path, where 'wildkey' can match any key.
        json_path (list): List of keys in the JSON path to compare against the schema path.
        
    Returns:
        bool: True if the schema path matches the JSON path, considering 'wildkey' matches any key, otherwise False.
    """
    # Paths must be the same length to match
    if len(schema_path) != len(json_path):
        return False

    # Iterate through each part of the schema and JSON paths
    return all(schema_part == "wildkey" or schema_part == json_part 
               for schema_part, json_part in zip(schema_path, json_path))


def check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, prefix_paths_dict):
    """
    Check if the paths from JSON Schemas exist in JSON datasets.

    Args:
        cleaned_ref_defn_paths (dict): Dictionary of JSON definitions and their paths.
        prefix_paths_dict (dict): Dictionary of paths and their corresponding value types.

    Returns:
        dict: Dictionary of definitions with paths that exist in the JSON documents.
    """
    filtered_ref_defn_paths = {}

    for ref_defn, schema_paths in cleaned_ref_defn_paths.items():
        intersecting_paths = []
        for schema_path in schema_paths:
            for json_path in prefix_paths_dict.keys():
                if path_matches_with_wildkey(schema_path, json_path):
                    intersecting_paths.append(json_path)
        
        # Add only if there are intersecting paths
        if intersecting_paths:
            filtered_ref_defn_paths[ref_defn] = intersecting_paths
    
    return filtered_ref_defn_paths


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

    # Return sorted dictionary of frequent references
    return dict(sorted(frequent_ref_defn_paths.items()))


def tokenize_schema(schema):
    """Tokenize schema.

    Args:
        schema (dict): DataFrame containing pairs, labels, filenames, and schemas of each path in pair

    Returns:
        torch.tensor: tokenized schema
        torch.tensor: inputs_ids tensor
    """

    # Tokenize the schema
    tokenized_schema = tokenizer(schema, return_tensors="pt", max_length=MAX_TOK_LEN, padding="max_length", truncation=True)
    return tokenized_schema


def merge_schemas(schema1, schema2):
    """
    Merges two JSON schemas recursively, summing `frequency` and taking the max `nesting_depth`.
    If schemas are of different types, combines them under `oneOf`.
    
    Args:
        schema1 (dict): The first JSON schema.
        schema2 (dict): The second JSON schema.
        
    Returns:
        dict: The merged JSON schema.
    """
    # Create a deep copy of the first schema to avoid modifying the original
    new_schema = deepcopy(schema1)

    # Check if both schemas have the same type
    if schema1.get("type") == schema2.get("type"):
        # If both schemas are objects, merge their properties
        if schema1.get("type") == "object":
            new_schema["required"] = list(set(schema1.get("required", [])) | set(schema2.get("required", [])))
            if "properties" in schema2:  # Ensure schema2 has properties to merge 
                for prop, value in schema2["properties"].items():
                    if prop in new_schema["properties"]:
                        # Recursively merge properties
                        new_schema["properties"][prop] = merge_schemas(new_schema["properties"][prop], value)
                    else:
                        # Add the new property
                        new_schema["properties"][prop] = value

        # If both schemas are arrays, merge their items
        elif schema1.get("type") == "array":
            if "items" in schema2:
                new_schema["items"] = merge_schemas(schema1["items"], schema2["items"])

        # Take the maximum nesting depth if it exists
        if "nesting_depth" in schema1 and "nesting_depth" in schema2:
            new_schema["nesting_depth"] = max(schema1["nesting_depth"], schema2["nesting_depth"])

        return new_schema

    # If schemas have different types, return a oneOf schema
    else:
        return {"oneOf": [schema1, schema2]}

    
def discover_schema(value, path_length):
    """
    Determine the structure (type) of the JSON key's value.
    Args:
        value: The value of the JSON key. It can be of any type.
        path_length (int): The length of the path to the key.
    Returns:
        dict: An object representing the structure of the JSON key's value.
    """
    if isinstance(value, str):
        return {"type": "string", "nesting_depth": path_length}
    elif isinstance(value, float):
        return {"type": "number", "nesting_depth": path_length}
    elif isinstance(value, int):
        return {"type": "integer", "nesting_depth": path_length}
    elif isinstance(value, bool):
        return {"type": "boolean", "nesting_depth": path_length}
    elif isinstance(value, list):
        # Handle lists, assuming mixed types if any items exist
        item_schemas = [discover_schema(item, path_length + 1) for item in value]
        if item_schemas:
            merged_items = reduce(merge_schemas, item_schemas)
        else:
            merged_items = {}
        return {"type": "array", "items": merged_items, "nesting_depth": path_length}
    elif isinstance(value, dict):
        schema = {"type": "object", "required": list(set(value.keys())), "properties": {}}
        for k, v in value.items():
            schema["properties"][k] = discover_schema(v, path_length + 1)
        return schema
    elif value is None:
        return {"type": "null"}
    else:
        raise TypeError(f"Unsupported value type: {type(value)}")


def discover_schema_from_values(values, path_length):
    """
    Determine the schema for a list of values.
    Args:
        values (list): The list of values to determine the schema for.
        path_length (int): The length of the path to the key.
    Returns:
        dict: The schema representing the structure of the list of values.
    """
    if not values:
        return {"type": "null"}
    else:
        return reduce(merge_schemas, (discover_schema(v, path_length) for v in values))


def create_dataframe(paths_dict, paths_to_exclude):
    """Create a DataFrame of paths and their values schema.

    Args:
        paths_dict (dict): Dictionary of paths and their values.
        paths_to_exclude (set): Paths to remove from JSON files.

    Returns:
        pd.DataFrame: DataFrame with tokenized schema added.
    """
    
    df_data = []
    
    for path, values in paths_dict.items():
        # Skip paths that are in the exclusion set
        if path in paths_to_exclude:
            continue

        # Parse and discover the schema from values
        parsed_values = [json.loads(v) for v in values]
        schema = discover_schema_from_values(parsed_values, len(path))
        
        # Check if the schema has more than one property
        if len(schema.get("properties", {})) > 1:
            tokenized_schema = tokenize_schema(json.dumps(schema))
            df_data.append([path, tokenized_schema, json.dumps(schema)])
        else:
            # Update paths_to_exclude if schema has less than or equal to one property
            paths_to_exclude.add(path)

    # Create DataFrame from collected data
    columns = ["path", "tokenized_schema", "schema"]
    df = pd.DataFrame(df_data, columns=columns)
    
    return df.sort_values(by="path")


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


def calculate_embeddings(df, model, device):
    """
    Calculate the embeddings for each path in batches.

    Args:
        df (pd.DataFrame): DataFrame containing paths and their tokenized schemas.
        model (torch.nn.Module): Pre-trained model to compute embeddings.
        device (torch.device): Device (CPU/GPU) for model inference.

    Returns:
        dict: Dictionary mapping paths to their corresponding embeddings.
    """
    schema_embeddings = {}
    paths = df["path"].tolist()
    tokenized_schemas = df["tokenized_schema"].tolist()

    # Create batches
    for batch_start in range(0, len(tokenized_schemas), BATCH_SIZE):
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


def calculate_cosine_distance(schema_embeddings, all_good_pairs):
    """
    Calculate the cosine distances between the embeddings of all paths.

    Args:
        schema_embeddings (dict): Dictionary containing paths and their corresponding embeddings.
        all_good_pairs (set): Set of good pairs of paths.
        df (pd.DataFrame): DataFrame containing paths and schemas.

    Returns:
        list: List of tuples containing path pairs and their cosine distance, sorted in ascending order of distance.
    """
    
    # Convert embeddings to a tensor
    paths_list = list(schema_embeddings.keys())
    embeddings_tensor = torch.stack([schema_embeddings[path] for path in paths_list])

    # Normalize embeddings
    normalized_embeddings = normalize(embeddings_tensor, p=2, dim=1)

    # Calculate cosine distances for all possible pairs
    all_pairs = list(itertools.combinations(range(len(paths_list)), 2))
    cosine_distances = []

    for path1_idx, path2_idx in all_pairs:
        path1 = paths_list[path1_idx]
        path2 = paths_list[path2_idx]

        # Skip pairs that are in the set of good pairs
        if (path1, path2) not in all_good_pairs and (path2, path1) not in all_good_pairs:
            # Calculate cosine similarity
            cosine_sim = torch.dot(normalized_embeddings[path1_idx], normalized_embeddings[path2_idx]).item()
            distance = 1 - cosine_sim
            cosine_distances.append(((path1, path2), distance))

    # Sort pairs by their cosine distance in ascending order
    cosine_distances.sort(key=lambda x: x[1])
    return cosine_distances


def label_samples(df, good_pairs, bad_pairs):
    """
    Label the samples in the DataFrame based on good and bad pairs.

    Args:
        df (pd.DataFrame): DataFrame containing paths, schemas, and filenames.
        good_pairs (set): Set of paths that should be groupe together.
        bad_pairs (set): Set of paths that should not be grouped together.

    Returns:
        pd.DataFrame: DataFrame containing labeled pairs, tokenized schemas, and filenames.
    """

    # Create lists to store data
    pairs = []
    labels = []
    tokenized_schemas1 = [] 
    tokenized_schemas2 = []  
    filenames = []  

    # Process good pairs: label them as 1 (positive)
    for pair in good_pairs:
        pairs.append(pair)
        labels.append(1)

        # Extract schemas and filename for both paths in the pair
        path1_row = df[df["path"] == pair[0]].iloc[0]
        path2_row = df[df["path"] == pair[1]].iloc[0]
        filenames.append(path1_row["filename"])
        tokenized_schemas1.append(path1_row["schema"])
        tokenized_schemas2.append(path2_row["schema"])
        
    # Process bad pairs: label them as 0 (negative)
    for pair in bad_pairs:
        pairs.append(pair)
        labels.append(0)
        
        # Extract schemas and filename for both paths in the pair
        path1_row = df[df["path"] == pair[0]].iloc[0]
        path2_row = df[df["path"] == pair[1]].iloc[0]
        filenames.append(path1_row["filename"])
        tokenized_schemas1.append(path1_row["schema"])
        tokenized_schemas2.append(path2_row["schema"])
        

    # Create a new DataFrame containing the labeled pairs, schemas, and filenames
    labeled_df = pd.DataFrame({"pairs": pairs,
                               "label": labels,
                               "filename": filenames,
                               "schema1": tokenized_schemas1,
                               "schema2": tokenized_schemas2
                               })

    return labeled_df


def get_samples(df, frequent_ref_defn_paths):
    """
    Generate labeled samples of good and bad pairs from the DataFrame based on ground truth definitions.

    Args:
        df (pd.DataFrame): DataFrame containing paths and schemas.
        frequent_ref_defn_paths (dict): Dictionary of frequent referenced definition and their paths.

    Returns:
        pd.DataFrame: Labeled dataFrame containing sample paths and schemas.
    """

    good_pairs = set()
    all_good_pairs = set()
    bad_pairs = set()
    ref_path_dict = {}

    # Get all paths in the schema
    paths = list(df["path"])
    
    # Generate good pairs from frequent referenced definition paths
    for ref_defn, good_paths in frequent_ref_defn_paths.items():
        good_paths_pairs = list(itertools.combinations(good_paths, 2))
        print(f"Number of good pairs for {ref_defn}: {len(good_paths_pairs)}")
        all_good_pairs.update(good_paths_pairs)
        good_pairs.update(itertools.islice(good_paths_pairs, 1000))

        # Map paths to their reference definition
        for path in good_paths:
            ref_path_dict[path] = ref_defn

    # Get non definition paths
    bad_paths = list(set(paths) - set(ref_path_dict.keys()))

    if bad_paths:
        # Calculate the embeddings of the tokenized schema
        schema_embeddings = calculate_embeddings(df)
        # Calculate cosine distances for all pairs
        cosine_distances = calculate_cosine_distance(schema_embeddings, all_good_pairs)
        
        # Select pairs with the smallest distances as bad pairs
        for pair, distance in cosine_distances:
            if len(bad_pairs) < len(good_pairs):
                bad_pairs.add(pair)
            else:
                break

    # Label data
    labeled_df = label_samples(df, good_pairs, bad_pairs)
    return labeled_df


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

    #print(f"Processing schema {schema_name}...")
    schema_path = os.path.join(SCHEMA_FOLDER, schema_name)

    # Load schema
    schema = load_schema(schema_path)
    if schema is None:
        failure_flags["load"] = 1
        print(f"Failed to load schema {schema_name}.")
        return None, None, schema_name, failure_flags

    #print("Get and clean referenced definitions")
    # Get and clean referenced definitions
    ref_defn_paths = clean_ref_defn_paths(schema)
    if not ref_defn_paths:
        failure_flags["ref_defn"] = 1
        print(f"No referenced definitions in {schema_name}.")
        return None, None, schema_name, failure_flags
    '''
    for ref, paths in ref_defn_paths.items():
        print(f"Reference: {ref} Paths: {paths}")
    print("_________________________________________________________________________________________________________________________")
    '''

    #print("Handle nested definitions")
    # Handle nested definitions
    new_ref_defn_paths = handle_nested_definitions(ref_defn_paths)
    cleaned_ref_defn_paths = remove_definition_keywords(new_ref_defn_paths)
    '''
    for ref, paths in cleaned_ref_defn_paths.items():
        print(f"Reference: {ref} Paths: {paths}")
    print("_________________________________________________________________________________________________________________________")
    '''

    paths_to_exclude = set()
    #print("get ref defn of type obj")
    get_ref_defn_of_type_obj(schema, cleaned_ref_defn_paths, paths_to_exclude)
    if not cleaned_ref_defn_paths:
        failure_flags["object_defn"] = 1
        print(f"No referenced definitions of type object in {schema_name}.")
        return None, None, schema_name, failure_flags
    '''
    for ref, paths in cleaned_ref_defn_paths.items():
        print(f"Reference: {ref} Paths: {paths}")
    print("_________________________________________________________________________________________________________________________")
    '''

    paths_dict = process_dataset(schema_name, filename)
    if paths_dict is None:
        failure_flags["path"] = 1
        print(f"No paths extracted from {schema_name}.")
        return None, None, schema_name, failure_flags

    #print("Check reference definition paths in the dataset")
    # Check reference definition paths in the dataset
    filtered_ref_defn_paths = check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, paths_dict)
    if not filtered_ref_defn_paths:
        failure_flags["schema_intersection"] = 1
        print(f"No paths of properties in referenced definitions found in {schema_name} dataset.")
        return None, None, schema_name, failure_flags
    '''
    for ref, paths in filtered_ref_defn_paths.items():
        print(f"Reference: {ref} Paths: {paths}")
    print("_________________________________________________________________________________________________________________________")
    '''

    #print("Find frequent definitions")
    # Find frequent definitions
    frequent_ref_defn_paths = find_frequent_definitions(filtered_ref_defn_paths, paths_to_exclude)
    if not frequent_ref_defn_paths:
        failure_flags["freq_defn"] = 1
        print(f"No frequent referenced definitions found in {schema_name}.")
        return None, None, schema_name, failure_flags
    '''
    for ref, paths in frequent_ref_defn_paths.items():
        print(f"Reference: {ref} Paths: {paths}")
    print("_________________________________________________________________________________________________________________________")
    '''
    
    print(f"Processing schema {schema_name}...")
    # Create DataFrame
    if filename == "baseline_test_data.csv":
        df = create_dataframe_baseline_model(paths_dict, paths_to_exclude)
    else:
        df = create_dataframe(paths_dict, paths_to_exclude)
        print(f"Number of paths in {schema_name}: {len(df)}")
        

    # Update reference definitions
    updated_ref_defn_paths = update_ref_defn_paths(frequent_ref_defn_paths, df)
    if not updated_ref_defn_paths:
        failure_flags["properties"] = 1
        print(f"Not enough properties found under refererenced definitions in {schema_name}.")
        return None, None, schema_name, failure_flags

    df["filename"] = schema_name
    df.reset_index(drop=True, inplace=True)

    return df, updated_ref_defn_paths, schema_name, failure_flags


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


def concatenate_dataframes(dfs):
    """
    Concatenate a list of pandas DataFrames efficiently using Dask.

    Args:
        dfs (list of pd.DataFrame): A list of pandas DataFrames to concatenate.

    Returns:
        pd.DataFrame: A single pandas DataFrame resulting from the concatenation
                      of the input DataFrames.
    """
     
    # Convert the list of DataFrames to Dask DataFrame
    dask_dfs = [dd.from_pandas(df, npartitions=4) for df in dfs]

    # Concatenate Dask DataFrames
    result_ddf = dd.concat(dask_dfs)
    result_df = result_ddf.compute()
    return result_df


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
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_schema, schema, filename): schema for schema in schemas}
        
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=1):  
            df = None
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
                        print(f"Sampling data for {schema_name}...")
                        df = get_samples(df, frequent_ref_defn_paths)
                    
                    # Append batch to CSV to avoid holding everything in memory
                    df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

            except Exception as e:
                print(f"Error processing schema {futures[future]}: {e}")
            finally:
                if df is not None:
                    del df

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
        train_size, random_value, m = sys.argv[-3:]
        train_ratio = float(train_size)
        random_value = int(random_value)


        # Split the data into training and testing sets
        train_set, test_set = split_data(train_ratio=train_ratio, random_value=random_value)

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

            # Preprocess the training and testing data
            preprocess_data(train_set, filename="sample_train_data.csv", ground_truth_file="train_ground_truth.json")
            preprocess_data(test_set, filename="sample_test_data.csv", ground_truth_file="test_ground_truth.json")

    except (ValueError, IndexError) as e:
        print(f"Error: {e}\nUsage: script.py <files_folder> <train_size> <random_value> <model>")
        sys.exit(1)
    

if __name__ == "__main__":
    main()