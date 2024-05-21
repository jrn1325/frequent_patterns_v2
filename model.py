import ast
import collections
import json
import math
import networkx as nx
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn.functional as F
import tqdm
import wandb

from adapters import AdapterTrainer, AutoAdapterModel
from collections import defaultdict
from itertools import combinations

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoConfig, AutoTokenizer, get_scheduler, TrainingArguments, EvalPrediction


MAX_TOK_LEN = 512
ADAPTER_PATH = "./adapter"

# https://stackoverflow.com/a/73704579
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def initialize_model(model_name, adapter_name="mrpc"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)
    model.add_classification_head(adapter_name, num_labels=2)
    model.add_adapter(adapter_name, config="seq_bn")
    model.set_active_adapters(adapter_name)
    model.train_adapter(adapter_name)
    wandb.watch(model)
    return model, tokenizer


def merge_schema_tokens(tokenized_schema1, tokenized_schema2, tokenizer):
    """Merge the tokens of the schemas of the paths with in pair

    Args:
        tokenized_schema1 (Tensor): Tokenized schema of path1.
        tokenized_schema2 (Tensor): Tokenized schema of path2.
        tokenizer (_type_): Tokenizer.

    Returns:
        list: list of merged tokens.
    """
    
    newline_token = tokenizer("\n")["input_ids"][1:-1]

    total_len = len(tokenized_schema1) + len(tokenized_schema2)
    max_tokenized_len = MAX_TOK_LEN - 1 - 2 # Account for BOS, EOS, and newline token lengths

    # Make sure both tokenized schemas are proportionally truncated and represented
    if total_len > max_tokenized_len:
        truncate_len = total_len - max_tokenized_len
        truncate_len1 = math.ceil(len(tokenized_schema1) / (total_len) * truncate_len)
        truncate_len2 = math.ceil(len(tokenized_schema2) / (total_len) * truncate_len)
        return [tokenizer.bos_token_id] + tokenized_schema1[:-truncate_len1].tolist() + newline_token + tokenized_schema2[:-truncate_len2].tolist() + [tokenizer.eos_token_id]

    else:
        return [tokenizer.bos_token_id] + tokenized_schema1.tolist() + newline_token + tokenized_schema2.tolist() + [tokenizer.eos_token_id]


def tokenize_schemas(df, tokenizer):
    """Tokenize schemas and add tokenized versions as new columns to the DataFrame. Remove first and last token.

    Args:
        df (pd.DataFrame): DataFrame containing pairs, labels, filenames, and schemas of each path in pair
        tokenizer: Microsoft CodeBERT tokenizer.

    Returns:
        DataFrame: dafaframe with tokenized schemas
    """
    tokenized_schemas = []

    # Loop over the schemas of the pairs of paths
    for idx, (schema1, schema2) in df[["Schema1", "Schema2"]].iterrows():

        # Tokenize the schemas, removing the first and last tokens
        tokenized_schema1 = tokenizer(schema1, return_tensors="pt", max_length=MAX_TOK_LEN, padding="max_length", truncation=True)["input_ids"][0][1:-1]
        tokenized_schema2 = tokenizer(schema2, return_tensors="pt", max_length=MAX_TOK_LEN, padding="max_length", truncation=True)["input_ids"][0][1:-1]

        # Merge the tokenized schemas
        tokenized_schema = merge_schema_tokens(tokenized_schema1, tokenized_schema2, tokenizer)
        tokenized_schemas.append(tokenized_schema)
    
    # Add a new column for tokenized schemas
    df["Tokenized_schema"] = tokenized_schemas

    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def transform_data(df, tokenizer, device):
    """
    Transforms the input training DataFrame into a format suitable for a transformer model.

    Args:
        df (pd.DataFrame): The training DataFrame with columns "Tokenized_schema" and "Label".
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing text data.
        device (torch.device): The device (CPU or GPU) to which tensors should be moved.

    Returns:
        list: A list of dictionaries, each containing "input_ids", "attention_mask", and "labels" tensors.
    """
    # Rename the target column to "labels"
    df = df.rename(columns={"Label": "labels"})

    # Get the maximum length of tokenized schemas
    max_length = max(len(schema) for schema in df["Tokenized_schema"])

    pad_token_id = tokenizer.pad_token_id

    dataset = []
    for schema, label in zip(df["Tokenized_schema"], df["labels"]):
        schema_tensor = torch.tensor(schema, device=device)
        padded_schema = torch.nn.functional.pad(schema_tensor, (0, max_length - len(schema)), value=pad_token_id)
        attention_mask = (padded_schema != pad_token_id).long()
        label_tensor = torch.tensor(label, device=device)

        dictionary = {
            "input_ids": padded_schema,
            "attention_mask": attention_mask.to(device),
            "labels": label_tensor
        }
        dataset.append(dictionary)

    return dataset


def train_model(train_df, test_df):
    model_name = "microsoft/codebert-base" 
    accumulation_steps = 4
    batch_size = 8
    learning_rate = 1e-6
    num_epochs = 25

    # Start a new wandb run to track this script
    wandb.init(
        project="custom-codebert_frequent_patterns",
        config={
            "accumulation_steps": accumulation_steps,
            "batch_size": batch_size,
            "dataset": "json-schemas",
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "model_name": model_name,
        }
    )

    # Initialize tokenizer, model with adapter and classification head
    model, tokenizer = initialize_model(model_name)

    # Tokenize the train and test schemas
    train_df_with_tokens = tokenize_schemas(train_df, tokenizer)
    test_df_with_tokens = tokenize_schemas(test_df, tokenizer)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Transform data into dict
    train_dataset = transform_data(train_df_with_tokens, tokenizer, device)
    test_dataset = transform_data(test_df_with_tokens, tokenizer, device)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

     # Calculate total steps
    total_steps = len(train_dataset) // batch_size * num_epochs

    # Define warm-up steps
    warmup_steps = int(0.1 * total_steps)

    # Initialize scheduler with warm-up steps
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Setup the training arguments
    training_args = TrainingArguments(
        report_to="wandb",
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=10,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        output_dir="./training_output",
        overwrite_output_dir=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=accumulation_steps,
    )

    # Create AdapterTrainer instance
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_accuracy,
        optimizers=(optimizer, lr_scheduler),
    )

    # Train the model
    trainer.train()
    trainer.evaluate()
        
    # Save the adapter
    save_adapter(model)
    wandb.save("./adapter/*")
  

def compute_accuracy(p: EvalPrediction):
  """
    Compute accuracy for the given predictions.

    Args:
        p (EvalPrediction): An EvalPrediction object containing predictions and label IDs.

    Returns:
        dict: A dictionary containing the accuracy.
    """
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}
    

def save_adapter(model):
    """
    Save the adapter and log it as a W&B artifact.

    Args:
        model (PreTrainedModel): The model with the adapter to be saved.
        adapter_name (str): The name of the adapter to save.
        save_path (str): The directory path where the adapter will be saved.
    """
    # Save the adapter
    model.save_adapter(ADAPTER_PATH, "mrpc", with_head=True)

    # Log the adapter artifact
    artifact = wandb.Artifact("customized_codebert_frequent_patterns", type="model")
    artifact.add_dir(ADAPTER_PATH)
    wandb.log_artifact(artifact)


def load_model_and_adapter():
    """
    Load the model and adapter from the specified path.

    Args:
        adapter_path (str): The directory path where the adapter is saved.

    Returns:
        PreTrainedModel: The model with the loaded adapter.
    """
    # Load the adapter configuration from file
    config_file_path = f"{ADAPTER_PATH}/adapter_config.json"
    with open(config_file_path, "r") as config_file:
        adapter_config = json.load(config_file)
    
    # Initialize the model with the same configuration as during saving
    model_name = adapter_config["model_name"]
    adapter_name = adapter_config["name"]

    config = AutoConfig.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name, config=config)

    # Load the adapter parameters
    model.load_adapter(ADAPTER_PATH, config=adapter_config)

     # Activate the adapter
    model.set_active_adapters(adapter_name)

    return model


def get_predicted_label(model, tokenized_pair, device):
    """
    Get the predicted label for a tokenized pair using the given model.

    Args:
        model (PreTrainedModel): The model used for prediction.
        tokenized_pair (list[int]): The tokenized input pair.
        device (torch.device): The device (CPU or GPU) to run the model on.

    Returns:
        int: The predicted label.
    """
    input_ids = torch.tensor([tokenized_pair]).to(device)
    attention_mask = torch.tensor([[1] * len(tokenized_pair)]).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

    return int(predictions[0])


def build_definition_graph(group, model, device):
    """
    Build a definition graph from tokenized schema pairs using the given model.

    Args:
        group (pd.DataFrame): A DataFrame containing pairs and tokenized schemas.
        model (PreTrainedModel): The model used for predicting connections.
        device (torch.device): The device (CPU or GPU) to run the model on.

    Returns:
        nx.Graph: A graph with edges representing predicted connections between pairs.
    """
    # Create a graph
    graph = nx.Graph()

    # Loop over tokenized schemas
    for _, row in group.iterrows():
        pair_str = row["Pairs"]
        # Convert string representation of tuple to tuple
        try:
            pair = ast.literal_eval(pair_str)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing pair: {pair_str} - {e}")
            continue

        tokenized_schema = row["Tokenized_schema"]

        # If the predicted label is 1, add an edge between the paths' nodes
        predicted_label = get_predicted_label(model, tokenized_schema, device)
        if predicted_label:  
            graph.add_edge(pair[0], pair[1])

    return graph


def find_definitions_from_graph(graph):
    """
    Find definitions from a graph representing schema connections.

    Args:
        graph (nx.Graph): A graph representing schema connections.

    Returns:
        List[List]: A list of lists, each containing nodes representing a definition.
    """
    #print("Graph edges")
    #print(graph.number_of_edges())
    #for e in graph.edges():
    #    print(e)

    # Get all the cliques from the graph and sort them based on their lengths in ascending order
    cliques = list(nx.algorithms.find_cliques(graph))
    cliques.sort(key=lambda a: len(a))
    
    # Make sure a path only appears in one definition
    processed_cliques = []
    while cliques:
        # Remove the last clique
        largest_clique = cliques.pop()
        # Create a set of vertices in the largest clique
        clique_set = set(largest_clique)
        # Filter out vertices from other cliques that are in the largest clique
        filtered_cliques = []
        for clique in cliques:
            filtered_clique = [c for c in clique if c not in clique_set]
            if len(filtered_clique) > 1:
                filtered_cliques.append(filtered_clique)
        processed_cliques.append(list(clique_set))
        cliques = filtered_cliques

    return processed_cliques


def evaluate_data(test_df, test_ground_truth, model, tokenizer):
    """
    Evaluate the model on test data.

    Args:
        test_df (pd.DataFrame): The DataFrame containing test data.
        test_ground_truth (dict): Dictionary containing ground truth information for test data.
        model (): The trained model to be evaluated.
        tokenizer: The tokenizer used for processing schema.
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the schema
    test_df = tokenize_schemas(test_df, tokenizer)

    # Clear CUDA cache
    torch.cuda.empty_cache()
    
     # Evaluate the model
    model.eval()

    # Group by "Filename" columns
    grouped = test_df.groupby(["Filename"])
    
    # Loop over each group
    for filename, group in tqdm.tqdm(grouped, position=0, leave=False, total=len(grouped)):
        filename = filename[0]
        
        # Build definition graph
        graph = build_definition_graph(group, model, device)
   
        # Predict clusters
        predicted_clusters = find_definitions_from_graph(graph)

        # Get the ground truth clusters
        ground_truth_dict = test_ground_truth.get(filename, {})
        actual_clusters = [[tuple(inner_list) for inner_list in outer_list] for outer_list in ground_truth_dict.values()]

        # Calculate precision, recall, and F1-score
        precision, recall, f1_score = calc_scores(actual_clusters, predicted_clusters)

        # Print evaluation metrics
        print(f"Schema: {filename}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")

        # Print actual and predicted clusters
        print("Actual clusters")
        for actual_cluster in actual_clusters:
            print(actual_cluster)

        print()

        print("Predicted clusters:")
        for predicted_cluster in predicted_clusters:
            print(predicted_cluster)
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print()
        
  

def calc_jaccard_index(actual_cluster, predicted_cluster):
    """Measure the similarity between actual and predicted clusters

    Args:
        actual_cluster (list): clusters from the json schemas
        predicted_cluster (list): clusters from the json files

    Returns:
        float: jaccard index
    """
 
    intersection = len(actual_cluster.intersection(predicted_cluster))
    union = len(actual_cluster) + len(predicted_cluster) - intersection
    return intersection / union
    

def calc_scores(actual_clusters, predicted_clusters, threshold=0.5):
    """Use schema definition properties (actuals) to evaluate the predicted clusters
    TP: Definition from the schema that exists in the file
    FP: Definition from the file that does not exist in schema
    FN: Definition from the schema that does not exist in the file
    TN: All the definitions that do not exist in the schema or json_files
    # Precision: for all things you found, how many of them you should have found?
    # Recall: for all things you should have found, how many did you find?

    Args:
        actual_clusters (list): clusters found in the json schemas
        predicted_clusters (_type_): clusters found in the json files
        threshold (float): minimum jaccard similarity score

    Returns:
        precision (float): Precision score
        recall (float): Recall score
        f1_score (float): F1-score
    """
 
    if not predicted_clusters:
        return 0, 0, 0
    
    TP = 0
    FP = 0
    FN = 0
        
    for actual_cluster in actual_clusters:
        found_match = False
        for predicted_cluster in predicted_clusters:
            jaccard_index = calc_jaccard_index(set(actual_cluster), set(predicted_cluster))
            if jaccard_index >= threshold:
                found_match = True
                TP += 1
                break
        if not found_match:
            FN += 1

    for predicted_cluster in predicted_clusters:
        found_match = False
        for actual_cluster in actual_clusters:
            jaccard_index = calc_jaccard_index(set(actual_cluster), set(predicted_cluster))
            if jaccard_index >= threshold:
                found_match = True
                break
        if not found_match:
            FP += 1

    
    
    try:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        precision, recall, f1_score = 0, 0, 0
    
    return precision, recall, f1_score
