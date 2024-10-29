import json
import model
import multiprocessing
import pandas as pd
import sys
import time



def load_data():
    """
    Load sampled training and testing datasets from Parquet files and train the model.
    """
    train_df = pd.read_csv("sample_train_data.csv") 
    test_df = pd.read_csv("sample_test_data.csv") 
    model.train_model(train_df, test_df)


def evaluate_model():
    """
    Evaluate the model on the test set.
    """

    test_ground_truth = {}
    with open("test_ground_truth.json", 'r') as json_file:
        for line in json_file:
            test_ground_truth.update(json.loads(line))

    model.evaluate_data(test_ground_truth)


def evaluate_simple_model():
    
    df = pd.read_csv("test_data.csv")

    test_ground_truth = {}
    with open("test_ground_truth.json", 'r') as json_file:
        for line in json_file:
            test_ground_truth.update(json.loads(line))

    model.group_paths(df, test_ground_truth)


def deref_schemas():
    test_ground_truth = {}
    with open("modified_ground_truth.json", 'r') as json_file:
        for line in json_file:
            test_ground_truth.update(json.loads(line))

    for schema, ground_truth in test_ground_truth.items():
        definitions_to_keep = list(ground_truth.keys())

        # Calculate the size of the dereferenced schema in bytes
        schema_size = model.dereference_and_calculate_schema_size(schema, definitions_to_keep)
        print(f"The size of the dereferenced schema {schema} is {schema_size} bytes")
    
                        
def main():
    """
    Main function to train or evaluate the model based on the provided mode from the command-line arguments. 
    If the mode is 'train', it calls the load_data function to train the model. If the mode is 'test', 
    it calls the evaluate_model function to evaluate the model. If the mode is unknown, it prints 
    an error message and exits.
    """
    mode = sys.argv[-1]
    if mode == "train":
        # Train the model
        load_data()
    elif mode == "test":
        start_time = time.time()
        # Evaluate the model
        evaluate_model()
        print(time.time() - start_time)
    elif mode == "simple":
        # Evaluate the simple model
        evaluate_simple_model()
    elif mode == "size":
        deref_schemas()
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'train' or 'test'.")
        sys.exit(1)

  

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()