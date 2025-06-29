import json
import model
import os
import pandas as pd
import sys
import time
import torch
import torch.multiprocessing as mp



def load_data(ori):
    """
    Load sampled training and testing datasets from CSV files and train the model.
    """
    train_df = pd.read_csv("sample_train_data.csv",sep=';') 
    test_df = pd.read_csv("sample_test_data.csv", sep=';') 

    if ori == "ori":
        model.train_model(train_df, test_df, ori=True)
    else:
        model.train_model(train_df, test_df, ori=False, train_mode="adapter")



def evaluate_model(ori):
    """
    Evaluate the model on the test set.

    If ori is True, it uses the original data; otherwise, it uses the sampled data.
    """
    test_ground_truth = {}
    with open("test_ground_truth.json", 'r') as json_file:
        for line in json_file:
            test_ground_truth.update(json.loads(line))
    if ori == "ori":
        model.evaluate_data(test_ground_truth, True)
    else:
        model.evaluate_data(test_ground_truth, False)



def evaluate_baseline_model():
    
    df = pd.read_csv("baseline_test_data.csv", sep=';')

    test_ground_truth = {}
    with open("test_ground_truth.json", 'r') as json_file:
        for line in json_file:
            test_ground_truth.update(json.loads(line))

    model.group_paths(df, test_ground_truth)

                        
def main():
    """
    Main function to train or evaluate the model based on the provided mode from the command-line arguments. 
    If the mode is 'train', it calls the load_data function to train the model. If the mode is 'test', 
    it calls the evaluate_model function to evaluate the model. If the mode is unknown, it prints 
    an error message and exits.
    """
    mode, ori = sys.argv[-2:]
    if mode == "train":
        load_data(ori)
    elif mode == "test":
        start_time = time.time()
        evaluate_model(ori)
        print(time.time() - start_time)
    elif mode == "baseline":
        start_time = time.time()
        evaluate_baseline_model()
        print(time.time() - start_time)
    elif mode == "size":
        model.get_json_schema_size(ori)
    elif mode == "info":
        train_df = pd.read_csv("sample_train_data.csv",sep=';') 
        train_df = pd.read_csv("sample_train_data.csv", sep=';')
        train_df['path1'] = train_df['path1'].apply(eval)  # Converts string representations of tuples to actual tuples
        train_df['path2'] = train_df['path2'].apply(eval)

        # Combine paths from both columns into a single Series and extract unique paths
        unique_paths = pd.concat([train_df['path1'], train_df['path2']]).drop_duplicates()

        # Calculate the length of each unique path
        unique_paths_with_lengths = unique_paths.apply(lambda x: (x, len(x)))

        # Create a DataFrame for unique paths with lengths
        unique_paths_df = pd.DataFrame(unique_paths_with_lengths.tolist(), columns=['path', 'length'])

        # Melt the DataFrame to include 'path1', 'path2', 'cosine_similarity', and 'label'
        similarity_records = train_df[['path1', 'path2', 'cosine_similarity', 'label']].melt(
            id_vars=['cosine_similarity', 'label'],
            value_name='path'
        ).drop_duplicates()

        # Merge unique paths with lengths and similarity/label data
        result_df = unique_paths_df.merge(similarity_records, on='path', how='left')

        # Display the result
        print(result_df)

        # Optionally save to a CSV file
        result_df.to_csv("unique_paths_with_lengths_similarity_and_labels.csv", index=False, sep=';')
        
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'train' or 'test'.")
        sys.exit(1)

  

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()