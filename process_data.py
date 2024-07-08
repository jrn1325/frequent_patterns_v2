import dask.dataframe as dd
import itertools
import json
import os
import pandas as pd
import sys
import torch
import tqdm
import warnings

from adapters import AutoAdapterModel
from collections import defaultdict
from copy import copy, deepcopy
from functools import reduce
from sklearn.model_selection import GroupShuffleSplit
from torch.nn.functional import normalize
from transformers import AutoTokenizer


model_name = "microsoft/codebert-base" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoAdapterModel.from_pretrained(model_name)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



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
MAX_TOK_LEN = 512




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
                    '''
                    if len(path) > 1:
                        prefix = path[:-1]
                        #prefix_paths_dict.setdefault(prefix, set()).add(path)
                        prefix_paths_dict.setdefault(prefix, []).append(path)
                    '''

                    
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
    input_ids_tensor = tokenized_schema["input_ids"]
    input_ids_tensor = input_ids_tensor[input_ids_tensor != tokenizer.pad_token_id]

    # Remove the first and last tokens
    input_ids_tensor_sliced = input_ids_tensor[1:-1]

    # Convert tensor to a numpy array and then list
    input_ids_numpy = input_ids_tensor_sliced.cpu().numpy()
   
    return tokenized_schema, list(input_ids_numpy)

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


def check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, json_paths):
    """Check if the paths from JSON Schemas exist in JSON datasets

    Args:
        cleaned_ref_defn_paths (dict): dictionary of JSON definitions and their paths
        df (pd.DataFrame): DataFrame containing paths and schemas.
        jsonfile_paths (list): list of paths found in the collection of JSON documents associated with a schema

    Returns:
        dict: dictionary without paths that don't exist in the collection of JSON documents
    """
    # Use set intersection to find schema paths that exist in both json file
    filtered_ref_defn_paths = {}
    
    for ref_defn, paths in cleaned_ref_defn_paths.items():
        intersecting_paths = set(paths) & set(json_paths)
        filtered_ref_defn_paths[ref_defn] = intersecting_paths
        
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


def create_dataframe(paths_dict):
    """Create a DataFrame of paths and their values schema
    Args:
        paths_dict (dict): Dictionary of paths and their values.

    Returns:
        pd.DataFrame: DataFrame with tokenized schema added.
    """

    df_data = []
    
    for path, values in paths_dict.items():
        values = [json.loads(v) for v in values]
        schema = discover_schema_from_values(values)
        tokenized_schema, input_ids = tokenize_schema(json.dumps(schema))
        df_data.append([path, tokenized_schema, input_ids, schema])
        
    columns = ["Path", "Tokenized_schema", "Input_ids", "Schema"]
    df = pd.DataFrame(df_data, columns=columns)
    return df.sort_values(by="Path")


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
        inputs = row["Tokenized_schema"]
        bad_path = row["Path"]

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
    paths = list(df["Path"])
    
    # Generate good pairs from frequent referenced definition paths
    for ref_defn, good_paths in frequent_ref_defn_paths.items():
        good_paths_pairs = list(itertools.combinations(good_paths, 2))
        all_good_pairs.update(good_paths_pairs)

        limited_pairs = itertools.islice(good_paths_pairs, 1000)
        good_pairs.update(limited_pairs)

        # Map paths to their reference definition
        for path in good_paths:
            ref_path_dict[path] = ref_defn

    # Get non definition paths
    bad_paths = list(set(paths) - set(ref_path_dict.keys()))

    if bad_paths:
        # Filter the DataFrame for bad paths
        filtered_df = df[df["Path"].isin(bad_paths)]
        # Calculate the embeddings of the tokenized schema
        schema_embeddings = calculate_embeddings(filtered_df)
        # Calculate cosine distances for all pairs
        cosine_distances = calculate_cosine_distance(schema_embeddings, all_good_pairs)
        #for i in cosine_distances[:10]:
            #print(i)
        
        # Select pairs with the smallest distances as bad pairs
        for pair, distance in cosine_distances:
            if len(bad_pairs) < len(good_pairs):
                bad_pairs.add(pair)
            else:
                break

    # Label data
    labeled_df = label_samples(df, good_pairs, bad_pairs)
    return labeled_df




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
        path1_row = df[df["Path"] == pair[0]].iloc[0]
        path2_row = df[df["Path"] == pair[1]].iloc[0]
        filenames.append(path1_row["Filename"])
        tokenized_schemas1.append(path1_row["Input_ids"])
        tokenized_schemas2.append(path2_row["Input_ids"])
        

    # Process bad pairs: label them as 0 (negative)
    for pair in bad_pairs:
        pairs.append(pair)
        labels.append(0)
        
        # Extract schemas and filename for both paths in the pair
        path1_row = df[df["Path"] == pair[0]].iloc[0]
        path2_row = df[df["Path"] == pair[1]].iloc[0]
        filenames.append(path1_row["Filename"])
        tokenized_schemas1.append(path1_row["Input_ids"])
        tokenized_schemas2.append(path2_row["Input_ids"])
        

    # Create a new DataFrame containing the labeled pairs, schemas, and filenames
    labeled_df = pd.DataFrame({"Pairs": pairs,
                               "Label": labels,
                               "Filename": filenames,
                               "Tokenized_schema1": tokenized_schemas1,
                               "Tokenized_schema2": tokenized_schemas2
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


def process_schema(schema, json_folder, schema_folder):
    """
    Process a single schema and return the relevant dataframes and ground truths.

    Args:
        schema (str): The name of the schema file.
        json_folder (str): The folder containing JSON files.
        schema_folder (str): The folder containing schema files.

    Returns:
        tuple: A tuple containing the filtered DataFrame and frequent referenced definition paths,
               or (None, None) if processing failed.
    """
    schema_path = os.path.join(schema_folder, schema)
    if not os.path.exists(schema_path):
        return None, None

    json_schema = load_schema(schema_path)
    if json_schema is None:
        return None, None

    dataset = os.path.join(json_folder, schema)
    if not ("$defs" in json_schema or "definitions" in json_schema) or not os.path.isfile(dataset):
        return None, None

    ref_defn_paths = clean_ref_defn_paths(json_schema)
    new_ref_defn_paths = handle_nested_definitions(ref_defn_paths)
    cleaned_ref_defn_paths = remove_definition_keywords(new_ref_defn_paths)

    paths_to_exclude = set()
    get_ref_defn_of_type_obj(json_schema, cleaned_ref_defn_paths, paths_to_exclude)

    if not cleaned_ref_defn_paths:
        return None, None

    paths_dict, prefix_paths_dict = get_paths_and_values(json_schema, dataset)
    filtered_ref_defn_paths = check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, list(paths_dict.keys()))

    if not filtered_ref_defn_paths:
        return None, None
    
    frequent_ref_defn_paths = find_frequent_definitions(filtered_ref_defn_paths, paths_to_exclude)

    if not frequent_ref_defn_paths:
        return None, None

    df = create_dataframe(paths_dict)

    filtered_df = df[~df["Path"].isin(paths_to_exclude)]
    if filtered_df.empty:
        return None, None

    filtered_df["Filename"] = schema
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df, frequent_ref_defn_paths


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
    Process all the data from the JSON files to get their embeddings.

    Args:
        schemas (list): List of schema filenames.
        filename (str): Filename to save the resulting DataFrame.
        ground_truth_file (str): Filename to save the ground truth definitions.
    """
    frames = []
    ground_truths = defaultdict(dict)

    for schema in tqdm.tqdm(schemas, position=2, leave=False, total=len(schemas)):
        if schema in ["grunt-clean-task.json", "swagger-api-2-0.json"]:#, "web-types.json"]:#, "openrpc-json.json"]:
            continue

        filtered_df, frequent_ref_defn_paths = process_schema(schema, JSON_FOLDER, SCHEMA_FOLDER)
        if filtered_df is not None and frequent_ref_defn_paths is not None:
            ground_truths[schema] = frequent_ref_defn_paths
            print(f"Sampling data for {schema}...")
            df = get_samples(filtered_df, frequent_ref_defn_paths)
            frames.append(df)
        
    #merged_df = concatenate_dataframes(frames)
    #merged_df[["Path", "Schema"]].to_csv("df.csv", index=False)
    
    if frames:
        print("Merging dataframes...")
        merged_df = concatenate_dataframes(frames)
        merged_df.to_parquet(filename, index=False)
        save_ground_truths(ground_truths, ground_truth_file)
    


def main():
    train_schemas, test_schemas = split_data()
    preprocess_data(train_schemas, filename="sample_train_data.parquet", ground_truth_file="train_ground_truth.json")
    preprocess_data(test_schemas, filename="sample_test_data.parquet", ground_truth_file="test_ground_truth.json")

    

if __name__ == "__main__":
    main()