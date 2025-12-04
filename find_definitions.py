import argparse
import ast
import json
import os
import time
from tqdm import tqdm

ARRAY_WILDCARD = "<ARRAY_ITEM>"


def load_schema(path):
    """
    Load a JSON schema from a file.

    Args:
        path (str): Path to the JSON schema file.
    Returns:
        dict: The loaded JSON schema.
    """
    with open(path, "r") as f:
        return json.load(f)
    
def resolve_ref(ref, root):
    """
    Resolve a local $ref within the root schema.
    Only supports '#/definitions/...' or '#/$defs/...'
    """
    if not ref.startswith("#/"):
        return None
    parts = ref.lstrip("#/").split("/")
    node = root
    for p in parts:
        node = node.get(p)
        if node is None:
            return None
    return node

def path_references_definition(schema, path, root):
    """
    Return the $ref string if the path references a definition, else None.
    Supports implicit intermediate keys (e.g., additionalProperties = true).
    """

    # Handle end of path: check for $ref
    if not path:
        ref = schema.get("$ref")
        if isinstance(ref, str) and (
            ref.startswith("#/definitions/") or ref.startswith("#/$defs/")
        ):
            return ref
        return None

    key = path[0]
    remaining = path[1:]

    # Skip artificial "$"
    if key == "$":
        return path_references_definition(schema, remaining, root)

    # Handle $ref before anything else
    if "$ref" in schema:
        resolved = resolve_ref(schema["$ref"], root)
        if resolved:
            # IMPORTANT: descend using *remaining*, not full path
            result = path_references_definition(resolved, path, root)
            if result:
                return result

    # Handle Object
    is_object = (
        schema.get("type") == "object"
        or "properties" in schema
        or "additionalProperties" in schema
    )

    if is_object:
        props = schema.get("properties", {})

        # Case A: Key exists literally in properties
        if key in props:
            result = path_references_definition(props[key], remaining, root)
            if result:
                return result

        # Case B: additionalProperties lets the key exist logically
        addl = schema.get("additionalProperties", True)

        # B1: additionalProperties = true or {}  → ANY key allowed
        if addl is True or addl == {}:
            # Create a blank "virtual" schema node
            result = path_references_definition({}, remaining, root)
            if result:
                return result

        # B2: additionalProperties = {schema} → descend into it
        if isinstance(addl, dict) and key not in props:
            result = path_references_definition(addl, remaining, root)
            if result:
                return result

    # Handle Arrays
    if schema.get("type") == "array" and key == ARRAY_WILDCARD:

        # prefixItems
        if "prefixItems" in schema:
            for subschema in schema["prefixItems"]:
                result = path_references_definition(subschema, remaining, root)
                if result:
                    return result

        # items
        if "items" in schema:
            result = path_references_definition(schema["items"], remaining, root)
            if result:
                return result

    # Handle combinators (anyOf / oneOf / allOf)
    for comb in ("anyOf", "oneOf", "allOf"):
        if comb in schema:
            # First check if any subschema is a $ref
            for subschema in schema[comb]:
                if "$ref" in subschema:
                    ref = subschema["$ref"]
                    if ref.startswith("#/definitions/") or ref.startswith("#/$defs/"):
                        return ref

            # Then recursively check the path in each subschema
            for subschema in schema[comb]:
                result = path_references_definition(subschema, path, root)
                if result:
                    return result

    return None



def load_json(path, schemas_dir):
    """
    Load a JSON file containing mappings:

    Args:
        path (str): Path to the JSON file.
        schemas_dir (str): Directory containing the JSON schemas.
    """
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)

            for schema_name, mapping in data.items():
                schema_path = os.path.join(schemas_dir, schema_name)
                print(f"\nLoading schema from {schema_path}")
                schema = load_schema(schema_path)

                for defn, paths in mapping.items():
                    print()
                    print(f"Processing Schema: {schema_name}, Definition: {defn}")
    
                    # Change definition from $defs to definitions if needed
                    if defn.startswith("#/$defs/"):
                        defn = defn.replace("#/$defs/", "#/definitions/")

                    for p in paths:
                        ref = path_references_definition(schema, p, schema)
                        if ref:
                            print(f"Path {p} references {ref}")
                        else:
                            print(f"Path {p} does not reference any definition")


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("ReCG_schemas", type=str, help="Directory for inferred schemas")
    parser.add_argument("ground_truth_input", type=str, help="File with ground truth paths")
    args = parser.parse_args()

    load_json(args.ground_truth_input, args.ReCG_schemas)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} sec", flush=True)
        
if __name__ == "__main__":
    main()