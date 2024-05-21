import collections
import io
import itertools
import json
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import sys
import torch
import tqdm
import warnings

from adapters import AutoAdapterModel
from collections import Counter, defaultdict
from copy import copy, deepcopy
from functools import reduce
from scipy.spatial.distance import cosine
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer

model_name = "microsoft/codebert-base" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoAdapterModel.from_pretrained(model_name)



warnings.filterwarnings("ignore")
sys.setrecursionlimit(30000) # I had to increase the recursion limit because of the discover_schema function

# Create constant variables
DISTINCT_SUBKEYS_UPPER_BOUND = 1000
RANDOM_VALUE = 101
TRAIN_RATIO = 0.80
REF_KEYWORD = "$ref"
ITEMS_KEYWORD = "items"
COMPLEX_PROPERTIES_KEYWORD = {"patternProperties", "additionalProperties"}
DEFINITION_KEYS = {"$defs", "definitions"}
JSON_SCHEMA_KEYWORDS = {"properties", "allOf", "oneOf", "anyOf"}
JSON_SUBSCHEMA_KEYWORDS = {"allOf", "oneOf", "anyOf"}

SCHEMA_FOLDER = "schemas"
JSON_FOLDER = "jsons"




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
            yield tuple(path) + (key,), value
            if isinstance(value, (dict, list)):
                yield from parse_document(value, tuple(path) + (key,))
    elif isinstance(doc, list):
        for item in doc:
            yield tuple(path) + ('*',), item
            if isinstance(item, (dict, list)):
                yield from parse_document(item, tuple(path) + ('*',))


def find_ref_paths(json_schema, current_path=('$',)):
    """Find the full paths of keys with $refs and append properties of the referenced definition to the path.

    Args:
        json_schema (dict): JSON schema object
        current_path (tuple, optional): path of the referenced definition. Defaults to ('$',).

    Yields:
        generator: full path and referenced definition name
    """
    if not isinstance(json_schema, dict):
        return

    for key, value in json_schema.items():
        updated_path = current_path + (key,)
        if key == REF_KEYWORD and isinstance(value, str):
            yield (value, current_path)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if item == ITEMS_KEYWORD:
                    list_path = updated_path + (item,)
                    yield from find_ref_paths(item, current_path=list_path)
                yield from find_ref_paths(item, current_path=updated_path)
        elif isinstance(value, list) or isinstance(value, dict):
            yield from find_ref_paths(value, current_path=updated_path)


def get_paths_and_values(json_schema, filename):
    """Get all the paths and their values. Values must be of type object and non empty

    Args:
        json_schema (dict): json schema
        filename (str): name of the json dataset
    Return:
        dict: dictionary of all the paths and values
    """
    paths_dict = defaultdict(set)
    #prefix_paths_dict = defaultdict(set)
    prefix_paths_dict = defaultdict(list)


    with open(filename, 'r') as f:
        for line in f:
            try:
                doc = json.loads(line)
            except:
                continue
            
            if isinstance(doc, dict) and len(doc) > 0 and match_properties(json_schema, doc):
                # Get the paths and values of type object and non-empty in the json document
                for path, value in parse_document(doc):
                    
                    if len(path) > 1:
                        prefix = path[:-1]
                        #prefix_paths_dict.setdefault(prefix, set()).add(path)
                        prefix_paths_dict.setdefault(prefix, []).append(path)

                    
                    if isinstance(value, dict) and len(value) > 0:
                        value = json.dumps(value)
                        paths_dict[path].add(value)                           

    return paths_dict, prefix_paths_dict


def match_properties(schema, document):
    """Check if there is an intersection between the properties in the schema with those from the document

    Args:
        schema (dict): JSON Schema object
        document (dict): JSON object

    Returns:
        boolean: True if there is a match, else False
    """
    # Extract schema properties #### Resolve references
    if "properties" in schema: 
        schema_properties = schema.get("properties", {})
        # Count the number of properties in the document that are defined in the schema
        matching_properties_count = sum(key in schema_properties for key in document)
        # Return true if there is a match, else false
        return matching_properties_count > 0
    else:
        return False


def merge_schemas(schema1, schema2):
    """
    Merges two JSON schemas recursively.

    Args:
        schema1 (dict): The first JSON schema.
        schema2 (dict): The second JSON schema.

    Returns:
        dict: The merged JSON schema.
    """
    # Check if both schemas have the same type
    if schema1.get("type") == schema2.get("type"):
        # Create a deep copy of the first schema to avoid modifying the original
        new_schema = deepcopy(schema1)
        
        # If both schemas are objects, merge their properties
        if schema1.get("type") == "object":
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
            if "items" in schema2:  # Ensure schema2 has items to merge
                new_schema["items"] = merge_schemas(schema1["items"], schema2["items"])
        
        return new_schema
    
    # If schemas have different types, return a oneOf schema
    else:
        return {"oneOf": [schema1, schema2]}


def discover_schema(value):
    """
    Determine the structure of the JSON key's value.

    Args:
        value: The value of the JSON key. It can be of any type.

    Raises:
        TypeError: Raised if the value does not have a recognized type.

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
        return {"type": "array", "items": discover_schema_from_values(value)}
    elif isinstance(value, dict):
        schema = {}
        for k, v in value.items():
            schema[k] = discover_schema(v)
        return {"type": "object", "properties": schema}
    elif value is None:
        return {"type": "null"}
    else:
        raise TypeError(f"Unsupported value type: {type(value)}")


def discover_schema_from_values(values):
    """
    Determine the schema for a list of values.

    Args:
        values (list): The list of values to determine the schema for.

    Returns:
        dict: The schema representing the structure of the list of values.
    """
    if not values:
        return {"type": "null"}
    else:
        return reduce(merge_schemas, (discover_schema(v) for v in values))


'''
def create_dataframe(prefix_paths_dict):
    """Create a DataFrame of paths, distinct subkeys, and numerical representations using FastText embeddings.

    Args:
        prefix_paths_dict (dict): Dictionary of paths and their prefixes.

    Returns:
        DataFrame: DataFrame containing paths, distinct subkeys, and numerical representations.
    """

    df_data = []

    # Iterate over paths under the current prefix to get all the unique nested keys
    for prefix, path_list in prefix_paths_dict.items():
        #distinct_subkeys = set()
        distinct_subkeys = []
        for path in path_list:
            if path[-1] == '*':
                continue

            distinct_subkeys.append(path[-1])
            #distinct_subkeys.add(path[-1])

            if len(distinct_subkeys) > DISTINCT_SUBKEYS_UPPER_BOUND:
                break

        if len(distinct_subkeys) == 0:
            # print(f"No distinct subkeys for path: {prefix}")
            continue

        if len(distinct_subkeys) == 1 and '*' in distinct_subkeys:
            continue

        distinct_subkeys = get_modes(distinct_subkeys)
        distinct_subkeys = sorted(list(distinct_subkeys))
        
        # Construct row data for DataFrame
        row_data = [prefix, distinct_subkeys]
        df_data.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(df_data, columns=["Path", "Schema"])
     # Sort the DataFrame by the "Path" column
    df_sorted = df.sort_values(by="Path")
    return df_sorted

'''
def create_dataframe(paths_dict):
    """Create a DataFrame of paths and their values schema
    Args:
        paths_dict (dict): Dictionary of paths and their values.

    Returns:
        DataFrame: DataFrame containing schema embeddings.
    """

    df_data = []
    
    for path, values in paths_dict.items():
        values = [json.loads(v) for v in values]
        schema = discover_schema_from_values(values)
        row_data = [path, schema] 
        df_data.append(row_data)
    columns = ["Path", "Schema"]
    df = pd.DataFrame(df_data, columns=columns)
    df_sorted = df.sort_values(by="Path")
    return df_sorted

    
def clean_ref_defn_paths(json_schema): 
    """Remove keywords associated with JSON Schema that do not exist in JSON documents format

    Args:
        json_schema (dict): JSON Schema object

    Returns:
        dict: dictionary of JSON paths without schem keywords
    """
    ref_defn_paths = defaultdict(set)

    # Loop through pairs of referenced definitions and their paths
    for ref, path in find_ref_paths(json_schema):
        # Prevent circular references
        if ref.split('/')[-1] in path:
            continue
        # Remove JSON Schema keywords from the paths
        cleaned_path = tuple('*' if key == ITEMS_KEYWORD else key for key in path if key not in JSON_SCHEMA_KEYWORDS)

        # XXX cannot handle paths with patternProperties or additionalProperties of tyoe object yet
        for keyword in COMPLEX_PROPERTIES_KEYWORD:
            if keyword in cleaned_path:
                break
        else:
            ref_defn_paths[ref].add(cleaned_path)

    return ref_defn_paths


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
            if defn_keyword in DEFINITION_KEYS:
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
            tuple(part for part in path if part not in DEFINITION_KEYS) for path in paths
        ]

    return updated_paths


def looks_like_object(schema):
    if not isinstance(schema, dict):
        return False
    elif "type" in schema:
        return schema["type"] == "object"
    else:
        return "properties" in schema


def get_ref_defn_of_type_obj(json_schema, ref_defn_paths, paths_to_exclude):
    """Remove referenced definitions that are not of type object

    Args:
        json_schema (dict): json schema object
        ref_defn_paths (dict): dictionary of referenced definitions 
        paths_to_exclude (list): paths to excludes from the json datasets
    Returns:
        dict: filtered dictionary
    """
  
    ref_to_delete = []
    
    for ref in ref_defn_paths.keys():
        # Get the name of the definition
        defn_name = ref.split("/")[-1]
        defn_root = json_schema.get("$defs") or json_schema.get("definitions")

        try:
            defn_obj = defn_root[defn_name]
        except (KeyError, AttributeError, TypeError):
            #print(ref)
            ref_to_delete.append(ref)
            continue

        # Skip if the definition object is not a dictionary
        if not isinstance(defn_obj, dict):
            #print("defn obj not dict", ref)
            ref_to_delete.append(ref)
            continue

        additional_properties_value = defn_obj.get("additionalProperties", False)

        # Skip if additionalProperties is True or not specified
        if additional_properties_value is not False:
            #print("addProp not false", ref)
            ref_to_delete.append(ref)
            continue

        # Skip if definition has less than two properties
        if "properties" not in defn_obj or len(defn_obj["properties"]) <= 1:
            #print("not enough prop", ref)
            ref_to_delete.append(ref)
            continue

        # Skip if the type is not object or oneOf/anyOf/allOf contain non-object types
        type_is_object = defn_obj.get("type") == "object"
        one_of_is_object = any(isinstance(item, dict) and item.get("type") == "object" for item in defn_obj.get("oneOf", []))
        any_of_is_object = any(isinstance(item, dict) and item.get("type") == "object" for item in defn_obj.get("anyOf", []))
        all_of_is_object = any(isinstance(item, dict) and item.get("type") == "object" for item in defn_obj.get("allOf", []))

        type_is_object = looks_like_object(defn_obj)
        one_of_is_object = any(looks_like_object(item) for item in defn_obj.get("oneOf", []))
        any_of_is_object = any(looks_like_object(item) for item in defn_obj.get("anyOf", []))
        all_of_is_object = any(looks_like_object(item) for item in defn_obj.get("allOf", []))
        if not (type_is_object or one_of_is_object or any_of_is_object or all_of_is_object):
            ref_to_delete.append(ref)

    for ref in ref_to_delete:
        paths_to_exclude.update(ref_defn_paths[ref])
        del ref_defn_paths[ref]
    
    return paths_to_exclude


def check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, jsonfile_paths):
    """Check if the paths from JSON Schemas exist in JSON datasets

    Args:
        cleaned_ref_defn_paths (dict): dictionary of JSON definitions and their paths
        jsonfile_paths (list): list of paths found in the collection of JSON documents associated with a schema

    Returns:
        dict: dictionary without paths that don't exist in the collection of JSON documents
    """
    # Use set intersection to find schema paths that exist in both json file
    filtered_ref_defn_paths = {ref_defn: set(paths) & set(jsonfile_paths) for ref_defn, paths in cleaned_ref_defn_paths.items()}
    return filtered_ref_defn_paths


def find_frequent_definitions(good_ref_defn_paths, paths_to_exclude):
    """Find referenced definitions that are referenced more than once.

    Args:
        good_ref_defn_paths (dict): Dictionary of reference definitions and their paths.
        paths_to_exclude (set): Paths to remove from JSON files
    Return:
        dict: Dictionary of frequently referenced definitions
    """
    frequent_ref_defn_paths = {}

    for ref, paths in good_ref_defn_paths.items():
        if len(paths) > 1:
            frequent_ref_defn_paths[ref] = paths
        else:
            paths_to_exclude.update(paths)

    # To prevent removing paths that multiple definitions use Ex: buildkite.json
    for bad_path in copy(paths_to_exclude):
        for paths in frequent_ref_defn_paths.values():
            if bad_path in paths:
                paths_to_exclude.remove(bad_path)
                break

    return frequent_ref_defn_paths


def print_items(dictionary):
    for k, v in dictionary.items():
        print("ref:", k, ", paths:", v)
    print()


def calculate_cosine_distance(a_df, all_paths, all_good_pairs):
    """
    Calculate the cosine distances between the embeddings of all paths.

    Args:
        a_df (pd.DataFrame): DataFrame containing the paths and their corresponding schemas.
        all_paths (list): List of all paths.
        all_good_pairs (set): Set of good pairs of paths.

    Returns:
        list: List of tuples containing path pairs and their cosine distance, sorted in ascending order of distance.
    """

    # Calculate embeddings for all paths
    path_embeddings = {}
    for path in all_paths:
        # Get the schema associated with the current path
        schema = a_df[a_df["Path"] == path].iloc[0]["Schema"]

        # Tokenize the schema
        inputs = tokenizer(json.dumps(schema), return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Calculate the embedding without computing gradients
        with torch.no_grad():
            outputs = model(**inputs)
    
        # Calculate the mean of the last hidden state to get the embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        path_embeddings[path] = embedding

    # Normalize embeddings
    for path in path_embeddings:
        path_embeddings[path] = path_embeddings[path] / np.linalg.norm(path_embeddings[path])

    # Calculate cosine distances for all possible pairs
    all_pairs = list(itertools.combinations(all_paths, 2))
    cosine_distances = []
    for path1, path2 in all_pairs:
        # Skip pairs that are in the set of good pairs
        if (path1, path2) not in all_good_pairs and (path2, path1) not in all_good_pairs:
            distance = cosine(path_embeddings[path1], path_embeddings[path2])
            cosine_distances.append(((path1, path2), distance))

    # Sort pairs by their cosine distance in ascending order
    cosine_distances.sort(key=lambda x: x[1])
    return cosine_distances


def get_samples(df, filename, ground_truth_dict):
    """
    Generate labeled samples of good and bad pairs from the DataFrame based on ground truth definitions.

    Args:
        df (pd.DataFrame): DataFrame containing paths and schemas.
        filename (str): Filename to save the resulting DataFrame.
        ground_truth_dict (dict): Dictionary with schema names as keys and frequent reference definition paths as values.
    """

    frames = []

    # Iterate over each schema and its corresponding frequent referenced definition paths
    for schema, frequent_ref_defn_paths in ground_truth_dict.items():
        good_pairs = set()
        all_good_pairs = set()
        bad_pairs = set()
        ref_path_dict = {}

        # Get the DataFrame for the current schema
        a_df = df[df["Filename"] == schema]

        # List of all paths in the current schema
        all_paths = list(a_df["Path"])

        # Generate good pairs from frequent referenced definition paths
        for ref_defn, paths in frequent_ref_defn_paths.items():
            # Generate all possible pairs and limited pairs (up to 1000)
            pairs_for_paths = itertools.combinations(paths, 2)
            all_good_pairs.update(pairs_for_paths)
            limited_pairs = itertools.islice(itertools.combinations(paths, 2), 1000)
            good_pairs.update(limited_pairs)

            # Map paths to their reference definition
            for path in paths:
                ref_path_dict[path] = ref_defn

        # Calculate cosine distances for all pairs
        cosine_distances = calculate_cosine_distance(a_df, all_paths, all_good_pairs)
        
        # Select pairs with the smallest distances as bad pairs
        for pair, distance in cosine_distances:
            if len(bad_pairs) < len(good_pairs):
                bad_pairs.add(pair)
            else:
                break

        # Label data and append to frames
        new_df = label_samples(a_df, good_pairs, bad_pairs)
        frames.append(new_df)

    # Merge data and save to parquet
    merged_df = pd.concat(frames, ignore_index=True)
    merged_df.to_parquet(filename, index=False)


def get_all_test_data(df, filename, ground_truth_dict):
    """
    Generate labeled good and bad pairs for all test data based on ground truth definitions.

    Args:
        df (pd.DataFrame): DataFrame containing paths and schemas.
        filename (str): Filename to save the resulting DataFrame.
        ground_truth_dict (dict): Dictionary with schema names as keys and frequent reference definition paths as values.
    """
    frames = []

    # Iterate over each schema and its corresponding frequent referenced definition paths
    for schema, frequent_ref_defn_paths in ground_truth_dict.items():
        good_pairs = set()
        bad_pairs = set()

        # Get the DataFrame for the current schema
        a_df = df[df["Filename"] == schema]

        # List of all paths in the current schema
        all_paths = list(a_df["Path"])

        # Generate good pairs from frequent referenced definition paths
        for ref_defn, paths in frequent_ref_defn_paths.items():
            pairs_for_paths = itertools.combinations(paths, 2)
            good_pairs.update(pairs_for_paths)

        # Generate bad pairs from all possible path combinations that are not in good pairs
        pairs_for_paths = itertools.combinations(all_paths, 2)
        for pair in pairs_for_paths:
            if pair not in good_pairs:
                bad_pairs.add(pair)
           
        # Label data and append to frames
        new_df = label_samples(a_df, good_pairs, bad_pairs)
        frames.append(new_df)

    # Merge all the DataFrames and save to a parquet file
    merged_df = pd.concat(frames, ignore_index=True)
    merged_df.to_parquet(filename, index=False)


def label_samples(train_df, good_pairs, bad_pairs):
    """
    Label the samples in the training DataFrame based on good and bad pairs.

    Args:
        train_df (pd.DataFrame): DataFrame containing paths, schemas, and filenames.
        good_pairs (set): Set of paths that should be groupe together.
        bad_pairs (set): Set of paths that should not be grouped together.

    Returns:
        pd.DataFrame: DataFrame containing labeled pairs, schemas, and filenames.
    """

    # Create lists to store data
    pairs = []
    labels = []
    schemas1 = [] 
    schemas2 = []  
    filenames = []  

    # Process good pairs: label them as 1 (positive)
    for pair in good_pairs:
        pairs.append(pair)
        labels.append(1)

        # Extract schemas and filename for both paths in the pair
        path1_row = train_df[train_df["Path"] == pair[0]].iloc[0]
        path2_row = train_df[train_df["Path"] == pair[1]].iloc[0]
        filenames.append(path1_row["Filename"])
        schemas1.append(path1_row["Schema"])
        schemas2.append(path2_row["Schema"])
        

    # Process bad pairs: label them as 0 (negative)
    for pair in bad_pairs:
        pairs.append(pair)
        labels.append(0)
        
        # Extract schemas and filename for both paths in the pair
        path1_row = train_df[train_df["Path"] == pair[0]].iloc[0]
        path2_row = train_df[train_df["Path"] == pair[1]].iloc[0]
        filenames.append(path1_row["Filename"])
        schemas1.append(path1_row["Schema"])
        schemas2.append(path2_row["Schema"])
        

    # Create a new DataFrame containing the labeled pairs, schemas, and filenames
    labeled_df = pd.DataFrame({"Pairs": pairs,
                               "Label": labels,
                               "Filename": filenames,
                               "Schema1": schemas1,
                               "Schema2": schemas2
                               })

    return labeled_df


def split_data():
    """
    Split the list of schemas into training and testing sets.

    Returns:
        tuple: A tuple containing the training set and testing set.
    """

    # Get the list of schema filenames
    schemas = os.listdir(SCHEMA_FOLDER)

    # Use GroupShuffleSplit to split the schemas into train and test sets
    gss = GroupShuffleSplit(train_size=TRAIN_RATIO, random_state=RANDOM_VALUE)

    # Make sure that schema names with the same first 3 letters are grouped together because they are likely from the same source
    train_idx, test_idx = next(gss.split(schemas, groups=[s[:3] for s in schemas]))

    # Create lists of filenames for the train and test sets
    train_set = [schemas[i] for i in train_idx]
    test_set = [schemas[i] for i in test_idx]

    return train_set, test_set


def preprocess_data(schemas, filename, ground_truth_file):
    """Process all the data from the JSON files to get their embeddings

    Args:
        schemas (str): list containing the schema names

    Returns:
        DataFrame: dataframe containing characteristics and embeddings of JSON paths and nested keys
    """

    frames = []
    ground_truths = defaultdict()
    
    for schema in tqdm.tqdm(schemas, position=1, leave=False, total=len(schemas)):

        schema_path = os.path.join(SCHEMA_FOLDER, schema)
        if os.path.exists(schema_path):

            if schema == "grunt-clean-task.json" or schema == "swagger-api-2-0.json":
                continue

            schema_path = os.path.join(SCHEMA_FOLDER, schema)
            with open(schema_path, 'r') as schema_file:
                try:
                    json_schema = json.load(schema_file)
                except json.JSONDecodeError as e:
                    print(e)
                    continue

            # Get the dataset associated to the schema
            dataset = os.path.join(JSON_FOLDER, schema)
            
            # Check if the schema contains definitions and the dataset exists
            if ("$defs" in json_schema or "definitions" in json_schema) and os.path.isfile(dataset):
        
                # Keep track of schemas with definitions
                #schemas_with_definitions.add(schema)

                #print("Clean paths")
                # Remove JSON Schema keywords from the paths
                ref_defn_paths = clean_ref_defn_paths(json_schema)
                #print_items(ref_defn_paths)
                #count_1 += len(ref_defn_paths.keys())

                #print("Handle paths")
                # Modify the paths of nested definitions
                new_ref_defn_paths = handle_nested_definitions(ref_defn_paths)
                #print_items(new_ref_defn_paths)

                # Remove the keyword definitions from paths that contain it
                cleaned_ref_defn_paths = remove_definition_keywords(new_ref_defn_paths)

                #print("Object paths")
                paths_to_exclude = set()
                # Get paths of referenced definitions of type object
                get_ref_defn_of_type_obj(json_schema, cleaned_ref_defn_paths, paths_to_exclude)
                #print_items(cleaned_ref_defn_paths)
                
                if len(cleaned_ref_defn_paths) > 0:
                    # Keep track schemas with object definitions
                    #schemas_with_object_definitions.add(schema)
                    #count_2 += len(cleaned_ref_defn_paths.keys())

                    # Get the paths from the json file
                    paths_dict, prefix_paths_dict = get_paths_and_values(json_schema, dataset)
                    
                    #print("In JSON")
                    # Check if the paths exist in the json files
                    filtered_ref_defn_paths = check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, list(paths_dict.keys()))
                    #print_items(filtered_ref_defn_paths)

                    if len(filtered_ref_defn_paths) > 0:
                        # Keep track schemas that exist in jsonfiles
                        #schemas_with_paths_in_jsonfiles.add(schema)
                        #count_3 += len(filtered_ref_defn_paths.keys())
                        # Find referenced definitions that are referenced more than once
                        frequent_ref_defn_paths = find_frequent_definitions(filtered_ref_defn_paths, paths_to_exclude)

                        if len(frequent_ref_defn_paths) > 0:
                            #print("Frequent paths")
                            #print_items(frequent_ref_defn_paths)
                            # Keep track schemas with frequent definitions
                            #schemas_with_frequent_definitions.add(schema)
                            #count_4 += len(frequent_ref_defn_paths.keys())

                            print(f"Creating a dataframe for {schema}...")
                            df = create_dataframe(paths_dict)
                            #df = create_dataframe(prefix_paths_dict)

                            # Remove the undesirable paths from the dataframe since we remove them from schema
                            filtered_df = df[~df["Path"].isin(paths_to_exclude)]
        
                            if not filtered_df.empty:
                                filtered_df["Filename"] = schema
                                # Reset the index
                                filtered_df.reset_index(drop=True, inplace=True)

                                # Add dataframe to list of frames that will be merge later
                                frames.append(filtered_df)
                                ground_truths[schema] = frequent_ref_defn_paths


    # Merge the dataframes of the datasets
    merged_df = pd.concat(frames, ignore_index=True)   

    # Sample the good and bad pair samples from the data
    get_samples(merged_df, filename, ground_truths) 

    if filename == "test_data.csv":   
        get_all_test_data(merged_df, "original_test_data.csv", ground_truths) 

    # Convert sets to lists in the inner dictionaries
    ground_truths_serializable = {
        key: {subkey: list(values) if isinstance(values, set) else values for subkey, values in subdict.items()}
        for key, subdict in ground_truths.items()
    }

    # Write each outer dictionary on a new line in the JSON file
    with open(ground_truth_file, "w") as json_file:
        for key, value in ground_truths_serializable.items():
            json_file.write(json.dumps({key: value}) + '\n')


def main():
    train_schemas, test_schemas = split_data()
    preprocess_data(train_schemas, filename="train_data.csv", ground_truth_file="train_ground_truth.json")
    preprocess_data(test_schemas, filename="test_data.csv", ground_truth_file="test_ground_truth.json")
    
    


if __name__ == "__main__":
    main()