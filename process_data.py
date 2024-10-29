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
from collections import defaultdict
import concurrent.futures
from copy import copy, deepcopy
from functools import reduce
from jsonschema import ValidationError
from jsonschema.validators import validator_for
from sklearn.model_selection import GroupShuffleSplit
from torch.nn.functional import normalize
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

MODEL_NAME = "microsoft/codebert-base" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoAdapterModel.from_pretrained(MODEL_NAME)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)




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


def looks_like_object(schema):
    """
    Check if the schema looks like an object.

    Args:
        schema (dict): JSON schema object.

    Returns:
        bool: True if the schema looks like an object, False otherwise.
    """
    if not isinstance(schema, dict):
        return False
    elif "type" in schema:
        return schema["type"] == "object"
    else:
        return "properties" in schema


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
    """Check if schema or dereferenced subschemas are of object type.
    
    Args:
        defn_obj (dict): JSON schema object.
        defn_root (dict): JSON schema object.
        
    Returns:
        bool: True if the schema or dereferenced subschemas are of object type, False otherwise.
    """
    if not defn_obj:
        return False
    if looks_like_object(defn_obj):
        return True
    for keyword in ["oneOf", "anyOf", "allOf"]:
        for item in defn_obj.get(keyword, []):
            if "$ref" in item:
                item = resolve_ref(defn_root, item["$ref"])
            if looks_like_object(item):
                return True
    return False


def get_ref_defn_of_type_obj(json_schema, ref_defn_paths, paths_to_exclude):
    """Remove referenced definitions that are not of type object, including those in subschemas

    Args:
        json_schema (dict): JSON schema object
        ref_defn_paths (dict): Dictionary of referenced definitions
        paths_to_exclude (set): Paths to exclude from the JSON datasets

    Returns:
        set: Updated paths to exclude
    """
    
    defn_root = json_schema.get("$defs") or json_schema.get("definitions")
    ref_to_delete = []

    for ref in ref_defn_paths.keys():
        defn_name = ref.split("/")[-1]
        
        try:
            defn_obj = defn_root[defn_name]
        except (KeyError, AttributeError, TypeError):
            ref_to_delete.append(ref)
            continue

        # Skip if definition is not a dictionary
        if not isinstance(defn_obj, dict):
            ref_to_delete.append(ref)
            print(f"Excluded {ref}: Definition is not a dictionary.")
            continue

        # Exclude definitions with fewer than two properties, unless they resemble objects
        if len(defn_obj.get("properties", {})) <= 1:
            if not is_object_like(defn_obj, defn_root):
                ref_to_delete.append(ref)
                print(f"Excluded {ref}: Has fewer than two properties and does not resemble an object.")
                continue

        # Skip if the definition type is not object or object-like
        if not is_object_like(defn_obj, defn_root):
            ref_to_delete.append(ref)
            print(f"Excluded {ref}: Definition is not an object.")
            continue

    # Remove excluded references and update paths to exclude
    for ref in ref_to_delete:
        paths_to_exclude.update(ref_defn_paths[ref])
        del ref_defn_paths[ref]

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


def parse_document(doc, path = ('$',)):
    """Get the path of each key and its value from the json documents

    Args:
        doc (dict, list): json document
        paths (tuple): tuples to store paths found. Defaults to ().

    Raises:
        ValueError: raise an error if the json object is not a dict or list

    Yields:
        dict: dictionary of paths of each key and their values
    """

    if isinstance(doc, dict):
        for key, value in doc.items():
            yield path + (key,), value
            if isinstance(value, (dict, list)):
                yield from parse_document(value, path + (key,))
    elif isinstance(doc, list):
        for item in doc:
            yield path + ('*',), item
            if isinstance(item, (dict, list)):
                yield from parse_document(item, path + ('*',))


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


def process_dataset(dataset):
    """
    Process and extract data from the documents, and return a DataFrame.
    
    Args:
        dataset (str): The name of the dataset file.

    Returns:
        dict: Dictionary of paths and their values.
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
                if isinstance(doc, dict):
                    if match_properties(schema, doc):
                        matched_document_count += 1 
                        process_document(doc, paths_dict)
                        num_docs += 1
                elif isinstance(doc, list):
                    for item in doc:
                        if isinstance(item, dict):
                            if match_properties(schema, item):
                                matched_document_count += 1 
                                process_document(item, paths_dict)
                                num_docs += 1
            except Exception as e:
                print(f"Error processing line in {dataset}: {e}")
                continue

    if len(paths_dict) == 0:
        return None
    else:
        return paths_dict


def path_matches_with_wildkey(schema_path, json_path):
    """Check if schema_path with 'wildkey' matches json_path.
    
    Args:
        schema_path (list): List of keys in the schema path.
        json_path (list): List of keys in the JSON path.
        
    Returns:
        bool: True if the schema path matches the JSON path, False otherwise.
    """
    if len(schema_path) != len(json_path):
        return False

    # Allow 'wildkey' in schema_path to match any value at that position in json_path
    for schema_part, json_part in zip(schema_path, json_path):
        if schema_part != "wildkey" and schema_part != json_part:
            return False
    return True


def check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, paths_dict):
    """Check if the paths from JSON Schemas exist in JSON datasets.

    Args:
        cleaned_ref_defn_paths (dict): Dictionary of JSON definitions and their paths.
        paths_dict (dict): Dictionary of paths and their values.

    Returns:
        dict: Dictionary without paths that don't exist in the collection of JSON documents.
    """
    filtered_ref_defn_paths = {}

    for ref_defn, schema_paths in cleaned_ref_defn_paths.items():
        intersecting_paths = set()
        for schema_path in schema_paths:
            for json_path in paths_dict.keys():
                if path_matches_with_wildkey(schema_path, json_path):
                    intersecting_paths.add(json_path)
        
        filtered_ref_defn_paths[ref_defn] = intersecting_paths
    
    return filtered_ref_defn_paths



def find_frequent_definitions(filtered_ref_defn_paths, paths_to_exclude):
    """
    Find referenced definitions that are referenced more than once.

    Args:
        filtered_ref_defn_paths (dict): Dictionary of reference definitions and their paths.
        paths_to_exclude (set): Paths to remove from JSON files

    Returns:
        dict: Dictionary of frequently referenced definitions.
    """
    frequent_ref_defn_paths = {}

    # Find frequently referenced definitions and update paths_to_exclude
    for ref, paths in filtered_ref_defn_paths.items():
        if len(paths) > 1:
            frequent_ref_defn_paths[ref] = paths
        else:
            paths_to_exclude.update(paths)

    # To prevent removing paths that multiple definitions use Ex: buildkite.json
    paths_to_exclude_copy = paths_to_exclude.copy()
    for bad_path in paths_to_exclude_copy:
        for paths in frequent_ref_defn_paths.values():
            if bad_path in paths:
                paths_to_exclude.remove(bad_path)
                break

    return dict(sorted(frequent_ref_defn_paths.items()))


def merge_schemas(schema1, schema2):
    """
    Merges two schemas derived from JSON documents, focusing on structure
    rather than strict schema types.

    Args:
        schema1 (dict): The first schema derived from a JSON document.
        schema2 (dict): The second schema derived from a JSON document.

    Returns:
        dict: The merged schema reflecting the combined structure.
    """
    # Check if both schemas have the same type
    if schema1.get("type") == schema2.get("type"):
        new_schema = deepcopy(schema1)

        # If both schemas are objects, merge their properties
        if new_schema["type"] == "object":
            new_schema.setdefault("properties", {})
            if "properties" in schema2:
                for prop, value in schema2["properties"].items():
                    if prop in new_schema["properties"]:
                        # Recursively merge properties
                        new_schema["properties"][prop] = merge_schemas(new_schema["properties"][prop], value)
                    else:
                        new_schema["properties"][prop] = value

        # If both schemas are arrays, merge their items
        elif new_schema["type"] == "array":
            if "items" in schema2:
                new_schema["items"] = merge_schemas(schema1.get("items", {}), schema2["items"])

        # Merge required fields, handling boolean and list types
        required1 = schema1.get("required", [])
        required2 = schema2.get("required", [])
        
        # Convert booleans to lists if necessary and ensure "properties" exist before accessing
        if isinstance(required1, bool):
            required1 = list(schema1.get("properties", {}).keys()) if required1 else []
        if isinstance(required2, bool):
            required2 = list(schema2.get("properties", {}).keys()) if required2 else []
        
        # Merge the lists of required fields
        new_schema["required"] = list(set(required1) | set(required2))

        return new_schema

    # If types differ, return a oneOf schema to capture possible structures
    return {"oneOf": [schema1, schema2]} 


def discover_schema(value, parent_key=("$",), parent_key_frequency=1):
    """
    Discover the implicit structure of a JSON document, including field frequencies,
    value constraints, and nesting depth, and infer if a field may be required.

    Args:
        value: The JSON value to inspect.
        parent_key (tuple): The path to the parent key. Defaults to ("$",).
        parent_key_frequency (float): The frequency of the parent key. Defaults to 1.

    Returns:
        dict: A schema object capturing the value's type, constraints, frequency, depth,
              and whether the path is likely required.
    """
    nesting_depth = len(parent_key)
    required_threshold = 1  

    # Detect string type and add constraints
    if isinstance(value, str):
        return {
            "type": "string",
            "nesting_depth": nesting_depth,
            "relative_frequency": parent_key_frequency,
            "required": parent_key_frequency >= required_threshold
        }

    # Detect number (integer or float) and track range
    elif isinstance(value, (int, float)):
        return {
            "type": "number" if isinstance(value, float) else "integer",
            "nesting_depth": nesting_depth,
            "relative_frequency": parent_key_frequency,
            "required": parent_key_frequency >= required_threshold
        }

    # Detect boolean
    elif isinstance(value, bool):
        return {
            "type": "boolean",
            "nesting_depth": nesting_depth,
            "relative_frequency": parent_key_frequency,
            "required": parent_key_frequency >= required_threshold
        }

    # Detect arrays and discover schema for items
    elif isinstance(value, list):
        if not value:
            return {
                "type": "array",
                "items": discover_schema_from_values(value),
                "nesting_depth": nesting_depth,
                "relative_frequency": parent_key_frequency,
                "required": parent_key_frequency >= required_threshold
            }

    # Detect objects and recursively discover schema for properties
    elif isinstance(value, dict):
        schema = {
            "type": "object",
            "properties": {},
            "nesting_depth": nesting_depth,
            "relative_frequency": parent_key_frequency,
            "required": parent_key_frequency >= required_threshold
        }
        total_keys = len(value)

        for k, v in value.items():
            key_frequency = parent_key_frequency / total_keys
            full_key_path = parent_key + (k,)
            schema["properties"][k] = discover_schema(v, full_key_path, key_frequency)

        return schema

    # Null or unrecognized type
    return {
        "type": "null",
        "nesting_depth": nesting_depth,
        "relative_frequency": parent_key_frequency,
        "required": parent_key_frequency >= required_threshold
    }


def discover_schema_from_values(values, parent_key=("$",), parent_key_frequency=1):
    """
    Determine the schema for a list of values, including handling mixed types and null values.

    Args:
        values (list): The list of values to determine the schema for.
        parent_key (tuple): The parent key path for determining nesting depth and relative frequency.
        parent_key_frequency (float): Frequency of the parent key, used to calculate the relative 
                                      frequency of nested keys. Defaults to 1.

    Returns:
        dict: The schema representing the structure of the list of values.
    """
    if not values:
        # Empty lists should return a 'null' type schema
        return {"type": "null"}
    
    # Initialize the discovery process by mapping each value to its schema
    value_schemas = (discover_schema(v, parent_key, parent_key_frequency) for v in values)
    
    # Use reduce to merge the schemas for each value in the list
    merged_schema = reduce(merge_schemas, value_schemas)
    
    return merged_schema


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
        values = [json.loads(v) for v in values]
        schema = discover_schema_from_values(values)
        
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


def calculate_embeddings(df):
    """
    Calculate the embeddings for each path.

    Args:
        df (pd.DataFrame): DataFrame containing the paths and their corresponding schemas.

    Returns:
        dict: Dictionary containing paths and their corresponding embeddings.
    """
    
    schema_embeddings = {}

    # Process paths and calculate embeddings
    for index, row in df.iterrows():
        inputs = row["tokenized_schema"]
        bad_path = row["path"]

        # Move inputs to the device
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        
        # Calculate embeddings without computing gradients
        with torch.no_grad():
            outputs = model(**inputs)

        # Calculate the mean of the last hidden state to get the embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Store the embeddings in the dictionary
        schema_embeddings[bad_path] = embeddings.squeeze().cpu() 
    
    return schema_embeddings


def calculate_cosine_distance(schema_embeddings, all_good_pairs):
    """
    Calculate the cosine distances between the embeddings of all paths.

    Args:
        schema_embeddings (dict): Dictionary containing paths and their corresponding embeddings.
        all_good_pairs (set): Set of good pairs of paths.

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


def process_schema(schema_name):
    """
    Process a single schema and return the relevant dataframes and ground truths.

    Args:
        schema_name (str): The name of the schema file.

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

    print(f"Processing schema {schema_name}...")
    schema_path = os.path.join(SCHEMA_FOLDER, schema_name)

    # Load schema
    schema = load_schema(schema_path)
    if schema is None:
        failure_flags["load"] = 1
        print(f"Failed to load schema {schema_name}.")
        return None, None, schema_name, failure_flags

    print("Get and clean referenced definitions")
    # Get and clean referenced definitions
    ref_defn_paths = clean_ref_defn_paths(schema)
    if not ref_defn_paths:
        failure_flags["ref_defn"] = 1
        print(f"No referenced definitions in {schema_name}.")
        return None, None, schema_name, failure_flags
    for ref, paths in ref_defn_paths.items():
        print(f"Reference: {ref} Paths: {paths}")
    print("_________________________________________________________________________________________________________________________")

    print("Handle nested definitions")
    # Handle nested definitions
    new_ref_defn_paths = handle_nested_definitions(ref_defn_paths)
    cleaned_ref_defn_paths = remove_definition_keywords(new_ref_defn_paths)
    for ref, paths in cleaned_ref_defn_paths.items():
        print(f"Reference: {ref} Paths: {paths}")
    print("_________________________________________________________________________________________________________________________")

    paths_to_exclude = set()
    print("get ref defn of type obj")
    get_ref_defn_of_type_obj(schema, cleaned_ref_defn_paths, paths_to_exclude)
    if not cleaned_ref_defn_paths:
        failure_flags["object_defn"] = 1
        print(f"No referenced definitions of type object in {schema_name}.")
        return None, None, schema_name, failure_flags
    for ref, paths in cleaned_ref_defn_paths.items():
        print(f"Reference: {ref} Paths: {paths}")
    print("_________________________________________________________________________________________________________________________")

    # Process dataset
    paths_dict = process_dataset(schema_name)
    if paths_dict is None:
        failure_flags["path"] = 1
        print(f"No paths extracted from {schema_name}.")
        return None, None, schema_name, failure_flags

    print("Check reference definition paths in the dataset")
    # Check reference definition paths in the dataset
    filtered_ref_defn_paths = check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, paths_dict)
    if not filtered_ref_defn_paths:
        failure_flags["schema_intersection"] = 1
        print(f"No paths of properties in referenced definitions found in {schema_name} dataset.")
        return None, None, schema_name, failure_flags
    for ref, paths in filtered_ref_defn_paths.items():
        print(f"Reference: {ref} Paths: {paths}")
    print("_________________________________________________________________________________________________________________________")
   
    print("Find frequent definitions")
    # Find frequent definitions
    frequent_ref_defn_paths = find_frequent_definitions(filtered_ref_defn_paths, paths_to_exclude)
    if not frequent_ref_defn_paths:
        failure_flags["freq_defn"] = 1
        print(f"No frequent referenced definitions found in {schema_name}.")
        return None, None, schema_name, failure_flags
    for ref, paths in frequent_ref_defn_paths.items():
        print(f"Reference: {ref} Paths: {paths}")
    print("_________________________________________________________________________________________________________________________")

    # Create DataFrame
    df = create_dataframe(paths_dict, paths_to_exclude)
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
        key: {subkey: list(values) if isinstance(values, set) else values for subkey, values in subdict.items()}
        for key, subdict in ground_truths.items()
    }

    with open(ground_truth_file, "w") as json_file:
        for key, value in ground_truths_serializable.items():
            json_file.write(json.dumps({key: value}) + '\n')


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
        futures = {executor.submit(process_schema, schema): schema for schema in schemas}
        
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
                    print(f"Sampling data for {schema_name}...")
                    df = get_samples(df, frequent_ref_defn_paths)
                    
                    # Append batch to CSV to avoid holding everything in memory
                    df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)

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


def main():
    try:
        # Parse command-line arguments
        train_size, random_value = sys.argv[-2:]
        train_ratio = float(train_size)
        random_value = int(random_value)
        
        # Split the data into training and testing sets
        train_set, test_set = split_data(train_ratio=train_ratio, random_value=random_value)

        # Preprocess the training and testing data
        preprocess_data(train_set, filename="sample_train_data.csv", ground_truth_file="train_ground_truth.json")
        preprocess_data(test_set, filename="sample_test_data.csv", ground_truth_file="test_ground_truth.json")

    except (ValueError, IndexError) as e:
        print(f"Error: {e}\nUsage: script.py <files_folder> <train_size> <random_value>")
        sys.exit(1)
    

if __name__ == "__main__":
    main()