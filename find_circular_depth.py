import json
import sys

from collections import defaultdict

JSON_SCHEMA_KEYWORDS = {"definitions", "$defs", "properties", "allOf", "oneOf", "anyOf", "not", "if", "then", "else"}


def read_json_file(file_path):
    """
    Reads the JSON file and returns the parsed data.
    
    Args:
        file_path: The path to the JSON file to read.
    
    Returns:
        The parsed JSON data.
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def extract_circular_definitions(schema, path=(), refs=None):
    """
    Recursively extracts paths to all `$ref` properties within the "definitions" or "$defs" objects
    of a JSON schema, preserving the path as a tuple. Returns the results as a list of tuples.

    Args:
        schema (dict or list): The schema object (could be a dictionary or list) to search.
        path (tuple, optional): The current path of the schema element being processed. Defaults to an empty tuple.
        refs (set, optional): A set to store the paths to all `$ref` properties. Defaults to None.

    Returns:
        set: A set of paths to all `$ref` definitions within the schema.
    """
    if refs is None:
        refs = set()

    # If the schema is a dictionary, process each key-value pair
    if isinstance(schema, dict):
        # Only process paths within the "definitions" or "$defs" object
        if path and path[0] in ("definitions", "$defs"):
            # If the dictionary contains a "$ref", record the path and reference
            if "$ref" in schema and schema["$ref"].split('/')[-1] in path:
                refs.add(schema["$ref"])

        # Recurse over each key in the dictionary
        for key, value in schema.items():
            if key == "items":
                new_path = path + ('*',)
            else:
                new_path = path + (key,)
            extract_circular_definitions(value, new_path, refs)
    
    # If the schema is a list, process each item
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            new_path = path + ('*',) 
            extract_circular_definitions(item, new_path, refs)
    
    return refs


def extract_ref_paths_for_definitions(schema, definitions, path=(), refs=None):
    """
    Recursively extracts paths to all `$ref` properties that refer to a given set of definitions.
    The full paths where those definitions are being referenced throughout the schema are captured.
    
    Args:
        schema (dict or list): The schema object (could be a dictionary or list) to search.
        definitions (list): A list of definition names to search for in `$ref` values.
        path (tuple, optional): The current path of the schema element being processed. Defaults to an empty tuple.
        refs (dict, optional): A dictionary to accumulate paths where definitions are referenced. Defaults to an empty dictionary.

    Returns:
        dict: A dictionary where:
            - Keys are `$ref` values (definition names).
            - Values are sets of tuples representing the paths where those definitions are referenced.
    """
    if refs is None:
        refs = defaultdict(set)

    # If the schema is a dictionary, process each key-value pair
    if isinstance(schema, dict):
        # Recurse over each key in the dictionary
        for key, value in schema.items():
            # For "items" key (in arrays), represent with "*"
            if key == "items":
                new_path = path + ('*',)
            else:
                new_path = path + (key,)
            # Check if there's a $ref and if it points to a definition we're tracking
            if "$ref" in schema:
                if schema["$ref"] in definitions:
                    refs[schema["$ref"]].add(path)

            # Continue the recursion
            extract_ref_paths_for_definitions(value, definitions, new_path, refs)
    
    # If the schema is a list, process each item
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            new_path = path + ('*',)  # Use "*" to represent items in a list
            extract_ref_paths_for_definitions(item, definitions, new_path, refs)
    
    return refs


def clean_ref_paths(ref_paths_dict):
    """
    Cleans ref_paths_dict by removing JSON schema keywords from the paths.

    Args:
        ref_paths_dict (dict): Dictionary of $ref paths to clean.

    Returns:
        Dict: A cleaned dictionary with JSON schema keywords removed.

    """
    cleaned_ref_paths = {}
    for ref, paths in ref_paths_dict.items():
        cleaned_paths = []
        for path in paths:
            cleaned_path = tuple([k for k in path if k not in JSON_SCHEMA_KEYWORDS])
            cleaned_paths.append(cleaned_path)
        cleaned_ref_paths[ref] = cleaned_paths
    return cleaned_ref_paths


def extract_all_paths(doc, path=()):
    """
    Extracts all paths from the JSON document, excluding the root symbol '$', 
    and only includes paths where the value is an object or a list.

    Args:
        doc (dict or list): The JSON document to traverse.
        path (tuple): The current path being traversed.

    Returns:
        list: A list of paths where values are either objects or lists.
    """
    paths = []
    
    if isinstance(doc, dict):
        for key, value in doc.items():
            new_path = path + (key,)
            if isinstance(value, (dict, list)): 
                paths.append(new_path)
                paths.extend(extract_all_paths(value, new_path))
    
    elif isinstance(doc, list):
        for index, item in enumerate(doc):
            new_path = path + ('*',)
            if isinstance(item, (dict, list)):
                paths.append(new_path)
                paths.extend(extract_all_paths(item, new_path))
    
    return paths


def contains_path(doc_path, ref_path):
    """
    Counts how many times a reference path is found in a document path.

    Args:
        doc_path (tuple): The full path from the document.
        ref_path (tuple): The reference path to search for.

    Returns:
        count: The number of times the reference path is found in the document path.
    """
    count = 0

    # A reference path should match a subsequence of a document path
    ref_len = len(ref_path)
    for i in range(len(doc_path) - ref_len + 1):
        if doc_path[i:i + ref_len] == ref_path:
            count += 1
    return count


def count_references_in_data(all_paths, ref_paths_dict):
    """
    Counts how many times each reference path appears in the JSON document and returns the maximum count
    for each reference path.

    Args:
        all_paths (set): A set of paths extracted from the JSON document.
        ref_paths_dict (dict): A dictionary where keys are reference strings (e.g., '#/definitions/child')
                                and values are lists of paths to check within the JSON document.

    Returns:
        dict: A dictionary where each key is a reference and each value is the maximum count of occurrences
              of that reference path in the document.
    """
    
    # Loop over each reference and its paths
    for ref, ref_paths in ref_paths_dict.items():
        max_counts = {}
        # Loop over the referenced paths
        for ref_path in ref_paths:
            # Track the highest count for this reference path across all document paths for each reference
            max_counts[ref_path] = max(contains_path(doc_path, ref_path) for doc_path in all_paths)

    return max_counts


def process_documents(schema, dataset_path):
    """
    Processes a JSON dataset where each document is separated by a newline.
    It checks each document for circular references based on the given schema,
    and tracks the unique circular reference paths, their maximum depth, and count.

    Args:
        schema (dict): The JSON schema object.
        dataset_path (str): The path to the JSON dataset.
    """
    # Get the format of the definitions in the schema
    ref_paths = extract_circular_definitions(schema)
    print(ref_paths)
    if not ref_paths:
        print("No circular references found in the schema.")
        return
    
    # Extract the reference paths for the definitions
    ref_paths_dict = extract_ref_paths_for_definitions(schema, ref_paths)
    print(ref_paths_dict)
    print()
    # Clean the reference paths by removing JSON schema keywords
    cleaned_ref_paths_dict = clean_ref_paths(ref_paths_dict)
    print(cleaned_ref_paths_dict)
    
    # Open the dataset file and read each line (document)
    all_paths = set() 
    with open(dataset_path, 'r') as file:
        for line in file:
            doc = json.loads(line)
            all_paths.update(extract_all_paths(doc)) 

    reference_counts_dict = count_references_in_data(all_paths, cleaned_ref_paths_dict)
    
    print(f"\nUnique circular reference paths with their maximum depths, counts, and referenced definitions:")
    for ref_name, depth in reference_counts_dict.items():
        print(f"Referenced Definition: {ref_name} | Max Depth: {depth}")





def main():
    schema_path, dataset_path = sys.argv[1], sys.argv[2]

    schema = read_json_file(schema_path)
    process_documents(schema, dataset_path)


if __name__ == "__main__":
    main()
