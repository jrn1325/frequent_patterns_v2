import ast
import collections
import json
import math
#import networkx as nx
import pandas as pd
import sys
import torch
import torch.nn.functional as F

import wandb

from adapters import AutoAdapterModel
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, get_scheduler

MAX_TOK_LEN = 512

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
    wandb.watch(model)
    return model, tokenizer


def tokenize_schemas(train_df, tokenizer):
    """Tokenize schemas and add tokenized versions as new columns to the DataFrame. Remove first and last token

    Args:
        train_df (DataFrame): train dataframe containing pairs, label, filename, and schema of each path in pair
        tokenizer: Microsoft CodeBERT tokenizer

    Returns:
        DataFrame: train dafaframe with tokenized schemas
    """
    tokenized_schemas = []

    # Loop over the schemas of the pairs of paths
    for idx, (schema1, schema2) in train_df[["Schema1", "Schema2"]].iterrows():

        # Tokenize the schemas
        tokenized_schema1 = tokenizer(schema1, return_tensors="pt", max_length=MAX_TOK_LEN, padding=True)["input_ids"][0][1:-1]
        tokenized_schema2 = tokenizer(schema2, return_tensors="pt", max_length=MAX_TOK_LEN, padding=True)["input_ids"][0][1:-1]

        # Merge the tokenized schemas
        tokenized_schema = merge_schema_tokens(tokenized_schema1, tokenized_schema2, tokenizer)
        tokenized_schemas.append(tokenized_schema)
    
    # Add a new column for tokenized schemas
    train_df["Tokenized_schema"] = tokenized_schemas

    return train_df


def tokenize_test_schemas(test_df, tokenizer):
    """Tokenize schemas and add tokenized versions as new columns to the DataFrame. Remove first and last token

    Args:
        test_df (DataFrame): test dataframe containing path, label, filename, and schema of path
        tokenizer: Microsoft CodeBERT tokenizer

    Returns:
        DataFrame: test dafaframe with tokenized schemas
    """

    test_df["Tokenized_schema"] = test_df["Schema"].apply(
        lambda schema: tokenizer(schema, return_tensors="pt", max_length=MAX_TOK_LEN, padding=True, truncation=True)["input_ids"][0][1:-1]
    )

    return test_df


def merge_schema_tokens(tokenized_schema1, tokenized_schema2, tokenizer):
    """Merge the tokens of the schemas of the paths with in pair

    Args:
        tokenized_schema1 (Tensor): tokenized schema of path1
        tokenized_schema2 (Tensor): tokenized schema of path2
        tokenizer (_type_): tokenizer

    Returns:
        list: list of merged tokens
    """
    
    newline_token = tokenizer("\n")["input_ids"][1:-1]

    total_len = len(tokenized_schema1) + len(tokenized_schema2)
    max_tokenized_len = MAX_TOK_LEN - 1 - 2

    # Make sure both tokenized schemas are proportionally represented
    if total_len > max_tokenized_len:
        remove_len = total_len - max_tokenized_len
        remove1 = math.ceil(len(tokenized_schema1) / (len(tokenized_schema1) + len(tokenized_schema2)) * remove_len)
        remove2 = math.ceil(len(tokenized_schema2) / (len(tokenized_schema1) + len(tokenized_schema2)) * remove_len)
        return [tokenizer.bos_token_id] + tokenized_schema1[:-remove1].tolist() + newline_token + tokenized_schema2[:-remove2].tolist() + [tokenizer.eos_token_id]

    else:
        return [tokenizer.bos_token_id] + tokenized_schema1.tolist() + newline_token + tokenized_schema2.tolist() + [tokenizer.eos_token_id]


def train_model(train_df):
    model_name = "microsoft/codebert-base"
    accumulation_steps = 4
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 10

    # Start a new wandb run to track this script
    wandb.init(
        project="custom-codebert_frequent_patterns_all_" + str(num_epochs),
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
    train_df_with_tokens = tokenize_schemas(train_df, tokenizer)
    #train_df_with_tokens.to_csv("sample_data.csv")

    # Shuffle the DataFrame
    train_df_with_tokens = train_df_with_tokens.sample(frac=1).reset_index(drop=True)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up scheduler to adjust the learning rate during training
    num_training_steps = num_epochs * len(train_df_with_tokens) // batch_size
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Get the maximum length of tokenized schemas
    max_length = max(len(schema) for schema in train_df_with_tokens["Tokenized_schema"])

    # Tokenize and pad schemas outside the epoch loop
    padded_schemas = []
    for schema in train_df_with_tokens["Tokenized_schema"]:
        padded_schema = torch.nn.functional.pad(torch.tensor(schema), (0, max_length - len(schema)), value=tokenizer.pad_token_id)
        padded_schemas.append(padded_schema)

    # Stack the padded schemas into a tensor
    input_ids = torch.stack(padded_schemas)

    # Create attention masks based on the presence of tokens
    attention_masks = torch.nn.functional.pad(torch.ones_like(input_ids), (0, max_length - input_ids.shape[1]), value=0).to(device)

    # Process labels outside the epoch loop
    labels = torch.tensor(train_df_with_tokens["Label"].values).to(device)

    # Train the model
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Epoch", position=0):
        total_loss = 0
        
        # Iterate over batches
        for i in tqdm(range(0, len(input_ids), batch_size), desc="Batch", position=0, leave=True):
            # Get the current batch
            batch_input_ids = input_ids[i:i + batch_size].to(device)
            batch_attention_masks = attention_masks[i:i + batch_size].to(device)
            batch_labels = labels[i:i + batch_size].to(device)
            print(batch_input_ids)
            print(batch_attention_masks)
            print(batch_labels)
            print()

            # Forward pass
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
  
            # Calculate the training loss
            training_loss = outputs.loss
            training_loss.backward()
            
            # Gradient accumulation and optimization step
            if (i + 1) % accumulation_steps == 0 or i == len(input_ids) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += training_loss.item()
            torch.cuda.empty_cache()
        
        average_loss = total_loss / len(train_df_with_tokens)
        wandb.log({"training_loss": average_loss})
        
    # Save the adapter
    #save_adapter(model)
    #wandb.save("./adapter/*")


def save_adapter(model):
    # Save the adapter
    adapter_path = "./adapter"
    model.save_adapter(adapter_path, "mrpc", with_head=True)

    # Log the adapter artifact
    artifact = wandb.Artifact("customized_codebert_frequent_patterns", type="model")
    artifact.add_dir(adapter_path)
    wandb.log_artifact(artifact)


def load_model_and_adapter():
    model_path = "model.pth"
    adapter_path = "./adapter"

    # Load the adapter configuration from file
    with open(f"{adapter_path}/adapter_config.json", "r") as config_file:
        adapter_config = json.load(config_file)
    
    # Initialize the model with the same configuration as during saving
    config = AutoConfig.from_pretrained(adapter_config["model_name"])
    model = AutoAdapterModel.from_pretrained(adapter_config["model_name"], from_tf=False, config=config)

    # Load the adapter parameters
    model.load_adapter(adapter_path, model_name=adapter_config["model_name"])

     # Activate the adapter
    model.set_active_adapters(adapter_config["name"])

    # Load the model state dictionary
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    return model


def get_predicted_label(model, tokenized_pair):
    outputs = model(input_ids=torch.tensor([tokenized_pair]), attention_mask=torch.tensor([[1] * len(tokenized_pair)]))
    logits = outputs.logits
    print(outputs.logits)
    predictions = torch.argmax(logits, dim=1)
    return int(predictions[0])


def test():
    # Create a graph
    graph = nx.Graph()
    paths = [('A') , ('B'), ('C')]

    # Add edges
    graph.add_edge(('A') , ('B'))
    graph.add_edge(('A') , ('C'))
    graph.add_edge(('B') , ('C'))
    graph.add_edge(('A') , ('D'))
    graph.add_edge(('C') , ('D'))
    graph.add_edge(('D') , ('E'))

    # Get all the cliques from the graph and sort them based on their lengths in ascending order
    #cliques = list(nx.algorithms.find_cliques(graph))
    cliques = [clique for clique in nx.algorithms.find_cliques(graph)]

    print("original cliques")
    print(cliques)
    #cliques = cliques[0]
    cliques.sort(key=lambda a: len(a))
    print()

    print("Sorted cliques")
    print(cliques)
    print()
    
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
    print("Processed cliques")
    print(processed_cliques)
    return processed_cliques


def build_definition_graph(group, model, tokenizer):
    # Create a graph
    graph = nx.Graph()
    
    # Generate unique pairs of paths
    path_pairs = combinations(group["Path"], 2)

    # Loop over unique pairs of paths
    for path1, path2 in path_pairs:
        
        # Get tokenized schemas
        tokenized_schema1 = group[group["Path"] == path1]["Tokenized_schema"].iloc[0]
        tokenized_schema2 = group[group["Path"] == path2]["Tokenized_schema"].iloc[0]
            
        # Merge the two tokenized schemas
        tokenized_pair = merge_schema_tokens(tokenized_schema1, tokenized_schema2, tokenizer)
        print(path1, path2)
        print(tokenized_pair)
        print(tokenizer.decode(tokenized_pair))

        # If the predicted label is 1 make a connection edge between the paths nodes
        predicted_label = get_predicted_label(model, tokenized_pair)
        if predicted_label:
            # Convert string representation of tuple to tuple
            path1 = ast.literal_eval(path1)
            path2 = ast.literal_eval(path2)
            graph.add_edge(path1, path2)

    return graph


def find_definitions_from_graph(graph):
    print("Graph edges")
    print(graph.number_of_edges())
    for e in graph.edges():
        print(e)
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
    # Tokenize the schema
    test_df = tokenize_test_schemas(test_df, tokenizer)

     # Evaluate the model
    model.eval()

    # Group by "Filename" columns
    grouped = test_df.groupby(["Filename"])

    # Loop over each group
    for (filename), group in grouped:
        print("Filename:" , filename)
        
        graph = build_definition_graph(group, model, tokenizer)
   
        predicted_clusters = find_definitions_from_graph(graph)

        # Get the ground truth of a schema
        ground_truth_dict = test_ground_truth[filename]
        actual_clusters = list(ground_truth_dict.values())
        # Convert list of lists of lists into list of lists of tuples
        actual_clusters = [[tuple(inner_list) for inner_list in outer_list] for outer_list in actual_clusters]
        
        precision, recall, f1_score = calc_scores(actual_clusters, predicted_clusters)

        print(f"schema: {filename}, precision: {precision}, recall: {recall}, f1-score: {f1_score}")
        print("Actual clusters")
        print(actual_clusters)
        print()
        print("Predicted clusters:")
        print(predicted_clusters)
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
    

def calc_scores(actual_clusters, predicted_clusters, threshold=0.0):
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
