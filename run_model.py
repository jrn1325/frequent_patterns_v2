import json
import model
import pandas as pd


from transformers import AutoTokenizer



def load_data():
    # Get the training data
    train_df = pd.read_csv("train_data.csv") 
    #values_to_filter = ["label-commenter-config-yml.json"]
    #train_filtered_df = train_df.loc[train_df["Filename"].isin(values_to_filter)]


    # Get the testing data
    #test_df = pd.read_csv("test_data_tokens.csv") 
    #values_to_filter = ["gitpod-configuration.json"]
    #test_filtered_df = test_df.loc[test_df["Filename"].isin(values_to_filter)]
    #test_df = test_df[test_df["Filename"] != "openrpc-json.json"]
    #values_to_filter = ["electron-builder-configuration-file.json", "plagiarize-me-yaml.json"]
    #test_filtered_df = test_df.loc[test_df["Filename"].isin(values_to_filter)]
    '''
    test_ground_truth = {}
    # Open the JSON file in read mode
    with open("test_ground_truth.json", "r") as json_file:
        # Read each line of the file
        for line in json_file:
            # Parse the JSON object in the line and update the loaded_data dictionary
            test_ground_truth.update(json.loads(line))
    '''
        
    # Train the model
    model.train_model(train_df)


def evaluate_model():
    

    # Load the model and adapter
    m = model.load_model_and_adapter()

    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

     # Get the testing data
    test_df = pd.read_csv("test_data.csv") 
    sample_df = test_df[test_df["Filename"] == "img-catapult-psp.json"]

    # Get the ground truth
    test_ground_truth = {}
    with open("test_ground_truth.json", "r") as json_file:
        for line in json_file:
            test_ground_truth.update(json.loads(line))

    # Evaluate the model
    model.evaluate_data(sample_df, test_ground_truth, m, tokenizer)

                
              
def main():
    # Tokenize training schemas
    #model.tokenize_schemas()
    # Train the model
    load_data()
    # Evaluate the model
    #evaluate_model()
    #model.test()
    


if __name__ == '__main__':
    main()