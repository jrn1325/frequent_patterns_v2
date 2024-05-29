import json
import model
import pandas as pd
import sys

from transformers import AutoTokenizer



def load_data():
    """
    Load sampled training and testing datasets from Parquet files and train the model.
    """
    train_df = pd.read_parquet("sample_train_data.parquet") 
    test_df = pd.read_parquet("sample_test_data.parquet") 
    model.train_model(train_df, test_df)


def evaluate_model():
    """
    Load the model and tokenizer, retrieve the testing data and ground truth, and evaluate the model.
    """

    m = model.load_model_and_adapter()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    test_ground_truth = {}
    with open("test_ground_truth.json", 'r') as json_file:
        for line in json_file:
            test_ground_truth.update(json.loads(line))

    model.evaluate_data(test_ground_truth, m, tokenizer)

                        
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
        # Evaluate the model
        evaluate_model()
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'train' or 'test'.")
        sys.exit(1)

  

if __name__ == "__main__":
    main()