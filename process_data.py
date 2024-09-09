import dask.dataframe as dd
import itertools
import json
import jsonschema
import os
import pandas as pd
import random
import sys
import torch
import torch.nn.functional as F
import tqdm
import warnings


from adapters import AutoAdapterModel
from collections import defaultdict
from copy import copy, deepcopy
from functools import reduce
from jsonschema import ValidationError
from jsonschema.validators import validator_for
from sklearn.model_selection import GroupShuffleSplit
from torch.nn.functional import normalize
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset





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

MODEL_NAME = "microsoft/codebert-base" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoAdapterModel.from_pretrained(MODEL_NAME)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



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


def match_properties(schema, document):
    """Check if there is an intersection between the properties in the schema with those from the document

    Args:
        schema (dict): JSON Schema object
        document (dict): JSON object

    Returns:
        boolean: True if there is a match, else False
    """
    # Extract schema properties #### Resolve references
    schema_properties = schema.get("properties", {})
    # Count the number of properties in the document that are defined in the schema
    matching_properties_count = sum(1 for key in document if key in schema_properties)
    # Return true if there is a match, else false
    return matching_properties_count > 0


def extract_definitions(schema):
    """
    Extract definitions from the schema.

    Args:
        schema (dict): The JSON schema.

    Returns:
        dict: A dictionary of definitions.
    """
    if "definitions" in schema:
        return {(k,): v for k, v in schema["definitions"].items()}
    elif "$defs" in schema:
        return {(k,): v for k, v in schema["$defs"].items()}
    return {}


def handle_reference(item, current_path, paths, definitions, visited):
    """
    Handle schema references and replace them with actual paths.
    Avoids circular references by tracking visited schemas.

    Args:
        item (dict): The current item with a reference.
        current_path (tuple): The current path being processed.
        paths (set): A set to store the extracted paths.
        definitions (dict): A dictionary of schema definitions.
        visited (set): Set of visited schema IDs to avoid circular references.
    """
    ref_path = item["$ref"]

    if ref_path.startswith("#/definitions/"):
        definition_key = ref_path[len("#/definitions/"):]
    elif ref_path.startswith("#/$defs/"):
        definition_key = ref_path[len("#/$defs/"):]
    else:
        return

    definition_path = (definition_key,)
    if definition_path in definitions:
        definition_schema = definitions[definition_path]

        if definition_path not in visited:
            visited.add(definition_path)
            process_item(definition_schema, current_path, paths, definitions, visited)
            visited.remove(definition_path) 


def process_item(item, current_path, paths, definitions, visited):
    """
    Recursively process items in the schema to extract all paths.

    Args:
        item (dict): The current schema item.
        current_path (tuple): The current path being processed.
        paths (set): A set to store the extracted paths.
        definitions (dict): A dictionary of schema definitions.
        visited (set): Set of visited schema to avoid circular references.
    """
    if isinstance(item, dict):
        if "properties" in item:
            for prop, subschema in item["properties"].items():
                new_path = current_path + (prop,)
                paths.add(new_path)
                process_item(subschema, new_path, paths, definitions, visited)
        if "items" in item:
            new_path = current_path + ('*',)
            paths.add(new_path)
            process_item(item["items"], new_path, paths, definitions, visited)
        if "$ref" in item:
            handle_reference(item, current_path, paths, definitions, visited)
        for key in ["oneOf", "anyOf", "allOf"]:
            if key in item:
                for subschema in item[key]:
                    process_item(subschema, current_path, paths, definitions, visited)


def extract_schema_paths(schema, path=('$',)):
    """
    Extract all possible paths from the JSON schema and replace references to definitions with actual paths.
    Replace the 'items' keyword with '*' in paths that contain array items.

    Args:
        schema (dict): The JSON schema.
        path (tuple): The current path being processed.

    Returns:
        set: A set of paths in the schema.
    """
    paths = set()
    definitions = extract_definitions(schema)
    visited = set()

    process_item(schema, path, paths, definitions, visited)
    return sorted(paths)


def add_additional_properties_false(schema):
    """
    Add "additionalProperties": false to all objects in a JSON schema
    where "additionalProperties" is not explicitly declared.

    Args:
        schema (dict or list): The JSON schema to modify. Can be a dictionary representing
                               the schema or a list of schemas.

    Returns:
        None
    """
    if isinstance(schema, dict):
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = {"not": {}}
    elif isinstance(schema, list):
        for item in schema:
            add_additional_properties_false(item)


def delete_key(data, key_path):
    """
    Delete a key from a nested dictionary or list using a key path.

    Args:
    data (dict or list): The JSON data.
    key_path (list): The path to the key to delete.
    """
    
    sub_data = data
    for key in key_path[:-1]:
        sub_data = sub_data[key]
    del sub_data[key_path[-1]]
    '''
    if not key_path:
        return

    if isinstance(data, dict):
        if len(key_path) == 1:
            data.pop(key_path[0], None)
        elif key_path[0] in data:
            delete_key(data[key_path[0]], key_path[1:])
    elif isinstance(data, list):
        index = key_path[0]
        if len(key_path) == 1:
            if 0 <= index < len(data):
                data.pop(index)
        elif 0 <= index < len(data):
            delete_key(data[index], key_path[1:])
    '''

    
def validate_and_clean(data, schema, max_iterations=100):
    """
    Recursively validate and clean JSON data according to a schema.
    The function iteratively removes parts of the document that make it invalid until it passes validation.

    Args:
    data (dict or list): The JSON data to validate and clean.
    schema (dict): The JSON schema to validate against.
    max_iterations (int): Maximum number of iterations to prevent infinite loops.

    Returns:
    dict or list: The cleaned JSON data with invalid parts removed, or None if the process fails.
    """
    try:
        # Apply the modification to the schema
        add_additional_properties_false(schema)
        
        cls = validator_for(schema)
        cls.check_schema(schema) # If no error on this line, the schema is valid
        validator = cls(schema)

        iteration = 0
        while iteration < max_iterations:
            errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
            # Exit the loop if there are no errors
            if not errors:
                break  
            
            for error in errors:
                error_path = list(error.path)
                #print(f"Removing invalid data at path: {error_path} due to error: {error.message}")
                try:
                    delete_key(data, error_path)
                except (KeyError, IndexError) as e:
                    #print(f"Failed to delete key at path: {error_path}. Error: {e}")
                    return None
            
            iteration += 1

        if iteration == max_iterations:
            print("Warning: Maximum iterations reached. Data may still be invalid.")
            return None

        return data
    except ValidationError as ve:
        print(f"Schema validation error: {ve}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def get_paths_and_values(json_schema, filename):
    """Get all the paths and their values. Values must be of type object and non empty

    Args:
        json_schema (dict): json schema
        filename (str): name of the json dataset
    Return:
        dict: dictionary of all the paths and values
    """
    paths_dict = defaultdict(set)
    prefix_paths_dict = defaultdict(set)
    total_docs = 0
    valid_docs = 0

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
        
            total_docs += 1
            # Validate the document
            doc = validate_and_clean(doc, json_schema)
            
            if isinstance(doc, dict) and len(doc) > 0 and match_properties(json_schema, doc):
                valid_docs += 1
                # Get the paths and values of type object and non-empty in the json document
                for path, value in parse_document(doc):
                    
                    if len(path) > 1 and isinstance(value, dict) and len(value) > 0:
                        prefix = path[:-1]
                        prefix_paths_dict[prefix].add(path)
                        value = json.dumps(value)
                        paths_dict[path].add(value)  
    try:
        print(f"Schema{filename} has {total_docs}, of which {valid_docs/total_docs} % are valid.")
    except ZeroDivisionError as e:
        print("Empty dataset.")

    # Sorting the dictionary by the size of the tuple keys
    paths_dict = dict(sorted(paths_dict.items(), key=lambda item: len(item[0])))
    
    return paths_dict, prefix_paths_dict, total_docs, valid_docs


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

        # Merge frequencies if they exist
        if "frequency" in schema1 and "frequency" in schema2:
            new_schema["frequency"] = schema1["frequency"] + schema2["frequency"]
        
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
    #input_ids_tensor = tokenized_schema["input_ids"]
    #input_ids_tensor = input_ids_tensor[input_ids_tensor != tokenizer.pad_token_id]

    # Remove the first and last tokens
    #input_ids_tensor_sliced = input_ids_tensor[1:-1]

    # Convert tensor to a numpy array and then list
    #input_ids_numpy = input_ids_tensor_sliced.cpu().numpy()
   
    return tokenized_schema#, list(input_ids_numpy)


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

        # XXX cannot handle paths with patternProperties or additionalProperties of type object yet
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
            ref_to_delete.append(ref)
            continue

        # Skip if the definition object is not a dictionary
        if not isinstance(defn_obj, dict):
            ref_to_delete.append(ref)
            continue

        additional_properties_value = defn_obj.get("additionalProperties", False)

        # Skip if additionalProperties is True or not specified
        if additional_properties_value is not False:
            ref_to_delete.append(ref)
            continue

        # Skip if definition has fewer than two properties
        if "properties" not in defn_obj or len(defn_obj["properties"]) <= 1:
            #print("not enough prop", ref)
            ref_to_delete.append(ref)
            continue

        # Skip if the type is not object or oneOf/anyOf/allOf contain non-object types
        if not (
            looks_like_object(defn_obj) or
            any(looks_like_object(item) for item in defn_obj.get("oneOf", [])) or
            any(looks_like_object(item) for item in defn_obj.get("anyOf", [])) or
            any(looks_like_object(item) for item in defn_obj.get("allOf", []))
        ):
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
    Returns:
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


def load_data_ambiguity_model():
    """
    Load the model and adapter from the specified path.

    Returns:
        PreTrainedModel: The model with the loaded adapter.
    """
    model_path = "/home/stu5/s17/jrn1325/Desktop/data_ambiguity/data_ambiguity_v2/adapter-model"
    
    m = AutoAdapterModel.from_pretrained(model_path)

    # Load the adapter
    adapter_name = m.load_adapter(model_path)

    # Activate the adapter
    m.set_active_adapters(adapter_name)

    return m


def is_dynamic(distinct_subkeys, tokenizer, data_ambiguity_model, device):
    # Convert the relevant data to tensors
    tokens = tokenizer(json.dumps(distinct_subkeys), return_tensors="pt", padding="max_length", truncation=True)
    input_ids = tokens["input_ids"].squeeze(0).to(device)
    attention_mask = tokens["attention_mask"].squeeze(0).to(device)

    data_ambiguity_model.to(device)
    data_ambiguity_model.eval()

    with torch.no_grad():
        # Forward pass for the single path
        outputs = data_ambiguity_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Get the prediction
        prediction = torch.argmax(logits, dim=1).item()

    return prediction


def get_distinct_subkeys(prefix_paths_dict):
    result_dict = {}

    # Iterate over paths under the current prefix to get all the unique nested keys
    for prefix, path_list in prefix_paths_dict.items():
        distinct_subkeys = set()
        for path in path_list:
            distinct_subkeys.add(path[-1])

            if len(distinct_subkeys) > DISTINCT_SUBKEYS_UPPER_BOUND:
                break

        if len(distinct_subkeys) == 0:
            continue

        # Sort and store the distinct subkeys in the dictionary
        result_dict[prefix] = sorted(list(distinct_subkeys))

    return result_dict


def update_new_paths(group_paths, new_paths_dict):
    """
    Merge new paths values into the existing ones.

    Args:
        new_paths_dict (list): The list of new path values to be merged.

    Returns:
        dict: The merged path values.
    """

    for path, values in group_paths.items():
        merged_paths_values = {}
        for value in values:
            dict_set = new_paths_dict[value]
            for d in dict_set:
                for k, v in json.loads(d).items():
                    merged_paths_values[k] = v
        new_paths_dict[value] = merged_paths_values


def handle_sub_paths(wildcard_path, prefix_paths_dict, group_paths):
    """
    Find sub_paths, update them with wildcard paths, and group paths values.

    Args:
        wildcard_path (tuple): The wildcard path to replace sub_paths with.
        prefix_paths_dict (dict): Dictionary of prefix paths and their sub_paths.
        group_paths (dict): Dictionary of merged paths.

    Returns:
        None
    """
   
    prefix = wildcard_path[:-1]
    sub_paths = prefix_paths_dict.get(prefix, set())

    if group_paths:
        for key, value in copy(group_paths).items():
            if prefix in value:
                group_paths[key + (prefix[-1],)] = sub_paths
    else:
        group_paths[wildcard_path] = sub_paths


def update_new_paths(group_paths, new_paths_dict):
    """
    Merge new paths values into the existing ones.

    Args:
        group_paths (dict): The dictionary of group paths.
        new_paths_dict (dict): The dictionary of new path values.
    """
    for wild_path, values in group_paths.items():
        merged_paths_values = defaultdict(set)
        for value in values:
            dict_set = new_paths_dict.get(value, set()) 
            for d in dict_set:
                for k, v in json.loads(d).items():
                    merged_paths_values[k].add(json.dumps(v))
        
        # Convert merged_paths_values from defaultdict(set) to a regular dictionary
        merged_dict = {k: list(v) for k, v in merged_paths_values.items()}
        
        # Add merged paths to new_paths_dict
        if wild_path not in new_paths_dict:
            new_paths_dict[wild_path] = set() 
        new_paths_dict[wild_path].add(json.dumps(merged_dict))

        
def delete_sub_paths(group_paths, new_paths_dict):
    # Delete original sub_paths from the new paths
    sub_paths_to_delete = group_paths.values()
    for sub_path_list in sub_paths_to_delete:
        for sub_path in sub_path_list:
            if sub_path in new_paths_dict:
                del new_paths_dict[sub_path]


def incorporate_dynamic_paths(paths_dict, prefix_paths_dict, paths_to_exclude, nested_keys_dict, data_ambiguity_model):
    """
    Incorporate dynamic paths by merging schemas for wildcard paths and updating the paths dictionary.

    Args:
        paths_dict (dict): Dictionary of paths and their values.
        paths_to_exclude (set): Set of paths to be excluded from the final dictionary.
        nested_keys_dict (dict): Dictionary of nested keys related to paths.
        data_ambiguity_model (model): Model used for prediction (placeholder for actual logic).

    Returns:
        tuple: A tuple containing:
            - dict: Updated dictionary with incorporated dynamic paths.
            - set: Updated set of paths to exclude.
    """
    # Sort paths by length
    paths_dict = dict(sorted(paths_dict.items(), key=lambda item: len(item[0])))

    new_paths_dict = {}
    group_paths = {}

    while paths_dict:
        # Take the first path
        path = next(iter(paths_dict))
        distinct_subkeys = nested_keys_dict.get(path, [])

        # Determine if the path is dynamic
        prediction = is_dynamic(distinct_subkeys, tokenizer, data_ambiguity_model, device)
        #prediction = 0
        #if path == ('$','a','b'):
        #    prediction = 1
    
        if prediction == 1:
            print(f"Prediction for path {path}: {prediction}")
            wildcard_path = path + ("*",)

            # Group the sub_paths by each wildcard path
            handle_sub_paths(wildcard_path, prefix_paths_dict, group_paths)
            
        # Move the path from paths_dict to new_paths_dict
        new_paths_dict[path] = paths_dict.pop(path)

    #for wild_path, values in group_paths.items():
    #    print("wild_path",wild_path)
    #    print("values",values)

    #print()
    #print("new_paths_dict",new_paths_dict.keys())
    # Merge the paths values into new_paths_dict based on group_paths
    update_new_paths(group_paths, new_paths_dict)
    #print("new_paths_dict",new_paths_dict)

    # Delete the original sub_paths from new_paths_dict
    delete_sub_paths(group_paths, new_paths_dict)
    #print("new_paths_dict",new_paths_dict)
    
    return new_paths_dict, paths_to_exclude



def create_dataframe(paths_dict, paths_to_exclude):
    """Create a DataFrame of paths and their values schema
    Args:
        paths_dict (dict): Dictionary of paths and their values.
        paths_to_exclude (set): Paths to remove from JSON files.

    Returns:
        pd.DataFrame: DataFrame with tokenized schema added.
    """

    df_data = []
    
    for path, values in paths_dict.items():
        values = [json.loads(v) for v in values]
        schema = discover_schema_from_values(values)
        if len(schema["properties"]) > 1:
            tokenized_schema = tokenize_schema(json.dumps(schema))
            df_data.append([path, tokenized_schema, json.dumps(schema)])
        else:
            paths_to_exclude.update(path)
        
    columns = ["Path", "Tokenized_schema", "Schema"]
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
        tokenized_schemas1.append(path1_row["Schema"])
        tokenized_schemas2.append(path2_row["Schema"])
        

    # Process bad pairs: label them as 0 (negative)
    for pair in bad_pairs:
        pairs.append(pair)
        labels.append(0)
        
        # Extract schemas and filename for both paths in the pair
        path1_row = df[df["Path"] == pair[0]].iloc[0]
        path2_row = df[df["Path"] == pair[1]].iloc[0]
        filenames.append(path1_row["Filename"])
        tokenized_schemas1.append(path1_row["Schema"])
        tokenized_schemas2.append(path2_row["Schema"])
        

    # Create a new DataFrame containing the labeled pairs, schemas, and filenames
    labeled_df = pd.DataFrame({"Pairs": pairs,
                               "Label": labels,
                               "Filename": filenames,
                               "Schema1": tokenized_schemas1,
                               "Schema2": tokenized_schemas2
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


def process_schema(schema, json_folder, schema_folder, data_ambiguity_model):
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
        return None, None, None, None

    json_schema = load_schema(schema_path)
    if json_schema is None:
        return None, None, None, None

    dataset = os.path.join(json_folder, schema)
    if not ("$defs" in json_schema or "definitions" in json_schema) or not os.path.isfile(dataset):
        return None, None, None, None

    ref_defn_paths = clean_ref_defn_paths(json_schema)
    new_ref_defn_paths = handle_nested_definitions(ref_defn_paths)
    cleaned_ref_defn_paths = remove_definition_keywords(new_ref_defn_paths)

    paths_to_exclude = set()
    get_ref_defn_of_type_obj(json_schema, cleaned_ref_defn_paths, paths_to_exclude)

    if not cleaned_ref_defn_paths:
        return None, None, None, None

    paths_dict, prefix_paths_dict, total_docs, valid_docs = get_paths_and_values(json_schema, dataset)
    filtered_ref_defn_paths = check_ref_defn_paths_exist_in_jsonfiles(cleaned_ref_defn_paths, list(paths_dict.keys()))

    if not filtered_ref_defn_paths:
        return None, None, None, None

    frequent_ref_defn_paths = find_frequent_definitions(filtered_ref_defn_paths, paths_to_exclude)

    if not frequent_ref_defn_paths:
        return None, None, None, None

    # Get distinct nested keys
    df_nested_keys = get_distinct_subkeys(prefix_paths_dict)
    new_paths_dict, paths_to_exclude = incorporate_dynamic_paths(paths_dict, prefix_paths_dict, paths_to_exclude, df_nested_keys, data_ambiguity_model)
    
    print("Number of old paths",len(paths_dict))
    print("Number of new paths",len(new_paths_dict))

    df = create_dataframe(new_paths_dict, paths_to_exclude)

    filtered_df = df[~df["Path"].isin(paths_to_exclude)]
    if filtered_df.empty:
        return None, None, None, None

    updated_ref_defn_paths = {}
    for ref_defn, paths in frequent_ref_defn_paths.items():
        intersecting_paths = set(paths) & set(filtered_df["Path"])
        if len(intersecting_paths) >= 2:
            updated_ref_defn_paths[ref_defn] = intersecting_paths

    filtered_df["Filename"] = schema
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df, updated_ref_defn_paths, total_docs, valid_docs


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


def preprocess_data(data_ambiguity_model,schemas, filename, ground_truth_file):
    """
    Process all the data from the JSON files to get their embeddings.

    Args:
        schemas (list): List of schema filenames.[]
        filename (str): Filename to save the resulting DataFrame.
        ground_truth_file (str): Filename to save the ground truth definitions.
    """
    frames = []
    ground_truths = defaultdict(dict)
    total = 0
    valid = 0

    for schema in tqdm.tqdm(schemas, position=2, leave=False, total=len(schemas)):
        #if schema != "pyproject.json":
        #    continue
        if schema in ["grunt-clean-task.json", "swagger-api-2-0.json"]:#, "web-types.json"]:#, "openrpc-json.json"]:
            continue

        filtered_df, frequent_ref_defn_paths, total_docs, valid_docs = process_schema(schema, JSON_FOLDER, SCHEMA_FOLDER, data_ambiguity_model)
    
        if filtered_df is not None and frequent_ref_defn_paths is not None:
            total += total_docs
            valid += valid_docs
            #filtered_df[["Path"]].to_csv(schema)
            
            ground_truths[schema] = frequent_ref_defn_paths
            print(f"Sampling data for {schema}...")
            df = get_samples(filtered_df, frequent_ref_defn_paths)
            frames.append(df)
    
    if frames:
        print("Merging dataframes...")
        merged_df = concatenate_dataframes(frames)
        merged_df.to_parquet(filename, index=False)
        save_ground_truths(ground_truths, ground_truth_file)
        print(f"Total valid docs: {valid}/{total}")
    

def main():
    # Load data ambiguity model
    data_ambiguity_model = load_data_ambiguity_model()
    train_schemas, test_schemas = split_data()
    preprocess_data(data_ambiguity_model, train_schemas, filename="sample_train_data.parquet", ground_truth_file="train_ground_truth.json")
    preprocess_data(data_ambiguity_model, test_schemas, filename="sample_test_data.parquet", ground_truth_file="test_ground_truth.json")

    

if __name__ == "__main__":
    main()