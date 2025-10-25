import json
import model
import os
import pandas as pd
import sys
import time
import torch.multiprocessing as mp

from collections import defaultdict



def evaluate_baseline_model():
    
    df = pd.read_csv("baseline_test_data.csv", sep=';')

    test_ground_truth = {}
    with open("test_ground_truth.json", 'r') as json_file:
        for line in json_file:
            test_ground_truth.update(json.loads(line))

    model.group_paths(df, test_ground_truth)

def evaluate_schema():
    """
    Evaluate the schema of the model.
    This function is currently a placeholder and does not perform any operations.
    """
    # Load CSV
    test_df = pd.read_csv("sample_test_data.csv", sep=';')

    schemas_by_file = defaultdict(list)

    # Collect all path-schema pairs per file
    for _, row in test_df.iterrows():
        filename = row["filename"]
        for path_col, schema_col in [("path1", "schema1"), ("path2", "schema2")]:
            path = model.parse_path(row[path_col])
            try:
                schema = json.loads(row[schema_col])
            except Exception:
                continue
            schemas_by_file[filename].append((path, schema))

    # Build and write schemas
    output_dir = "global_schemas"
    os.makedirs(output_dir, exist_ok=True)

    for filename, path_schema_pairs in schemas_by_file.items():
        seen = set()
        global_schema = {"type": "object", "properties": {}}
        for path, schema in path_schema_pairs:
            key = (tuple(path), json.dumps(schema, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            partial = {"type": "object", "properties": {}}
            model.insert_path(partial, path, schema)
            model.merge_dicts(global_schema, partial)
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w") as f:
            json.dump(global_schema, f, indent=2)
 




def main():
    """
    Main function to train or evaluate the model based on the provided mode from the command-line arguments. 
    If the mode is 'train', it calls the load_data function to train the model. If the mode is 'test', 
    it calls the evaluate_model function to evaluate the model. If the mode is unknown, it prints 
    an error message and exits.
    """

    try:
        train_data, test_data, mode, *extra = sys.argv[-4:]
        training_mode = extra[0].lower() if extra else "adapter"
        if training_mode not in {"adapter", "full"}:
            raise ValueError("Invalid training mode. Use 'adapter' or 'full'.")
        if mode not in {"train", "eval"}:
            raise ValueError("Invalid mode. Use 'train' or 'eval'.")
        
        if mode == "train":
            train_df = pd.read_csv(train_data, delimiter=';')
            test_df = pd.read_csv(test_data, delimiter=';')
            model.train_model(train_df, test_df, training_mode=training_mode)
        elif mode == "eval":
            start_time = time.time()
            test_ground_truth = {}
            with open("test_ground_truth.json", 'r') as json_file:
                for line in json_file:
                    test_ground_truth.update(json.loads(line))
            model.evaluate_model(test_ground_truth, eval_mode=training_mode)
            print(time.time() - start_time)
        elif mode == "baseline":
            start_time = time.time()
            evaluate_baseline_model()
            print(time.time() - start_time)
        elif mode == "size":
            #model.get_json_schema_size(ori)
            deref_schemas_dir = "global_schemas_deref"
            original_schemas_dir = "global_schemas_ori"
            abstracted_schemas_dir = "global_schemas_abstracted"
            model.compare_schema_sizes(deref_schemas_dir, original_schemas_dir, abstracted_schemas_dir)
        
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 
  

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()