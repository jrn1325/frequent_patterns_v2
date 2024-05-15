import json
import model
import pandas as pd
import sys


from transformers import AutoTokenizer



def load_data():
    # Get the training data and testing sets
    train_df = pd.read_csv("train_data.csv") 
    test_df = pd.read_csv("test_data.csv") 
        
    # Train the model
    model.train_model(train_df, test_df)


def evaluate_model():
    
    # Load the model and adapter
    m = model.load_model_and_adapter()

    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

     # Get the testing data
    test_df = pd.read_csv("test_data.csv") 

    # Get the ground truth
    test_ground_truth = {}
    with open("test_ground_truth.json", "r") as json_file:
        for line in json_file:
            test_ground_truth.update(json.loads(line))

    # Evaluate the model
    model.evaluate_data(test_df, test_ground_truth, m, tokenizer)

                
              
def main():
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
  
    


if __name__ == '__main__':
    main()