import ast
import concurrent.futures
import json
import jsonref
import math
import networkx as nx
import numpy as np
import os
import subprocess
import sys
import torch
import torch.nn as nn
import tqdm
import wandb

from adapters import AdapterTrainer, AutoAdapterModel
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
from torch.optim import AdamW
from transformers import AutoTokenizer, get_scheduler, TrainingArguments, EvalPrediction

import process_data


MAX_TOK_LEN = 512
MODEL_NAME = "microsoft/codebert-base" 
PATH = "./adapter-model"
ADAPTER_NAME = "frequent_patterns"
SCHEMA_FOLDER = "processed_schemas"
JSON_FOLDER = "processed_jsons"
BATCH_SIZE = 32
JSON_SUBSCHEMA_KEYWORDS = {"allOf", "oneOf", "anyOf", "not"}

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


def initialize_model():
    """
    Initializes a pre-trained model with an adapter for classification tasks.
   
    Returns:
        tuple: A tuple containing the initialized model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)
    model.add_classification_head(ADAPTER_NAME, num_labels=2)
    model.add_adapter(ADAPTER_NAME, config="seq_bn")
    model.set_active_adapters(ADAPTER_NAME)
    model.train_adapter(ADAPTER_NAME)
    wandb.watch(model)
    return model, tokenizer


def merge_schema_tokens(df, tokenizer):
    """Merge the tokens of the schemas of the paths with in pair.

    Args:
        df (pd.DataFrame): DataFrame containing pairs, labels, filenames, and tokenized schema of each path in pair
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing schemas.

    Returns:
        DataFrame: dafaframe with merged tokenized schemas
    """
    
    newline_token = tokenizer("\n")["input_ids"][1:-1]
    tokenized_schemas = []

    # Loop over the schemas of the pairs of tokenized schemas
    for idx, (schema1, schema2) in tqdm.tqdm(df[["schema1", "schema2"]].iterrows(), position=4, leave=False, total=len(df), desc="merge tokens"):
        #pair = df.iloc[idx]["Pairs"]
        #schema1 = schema1.replace("'", '"')
        #schema2 = schema2.replace("'", '"')
        schema1 = json.loads(schema1)
        schema2 = json.loads(schema2)

        ordered_schema1, ordered_schema2 = order_properties_by_commonality(schema1, schema2)

        tokenized_schema1 = tokenize_schema(ordered_schema1, tokenizer)
        tokenized_schema2 = tokenize_schema(ordered_schema2, tokenizer)

        total_len = len(tokenized_schema1) + len(tokenized_schema2)
        max_tokenized_len = MAX_TOK_LEN - 1 - 2  # Account for BOS, EOS, and newline token lengths
    
        # Proportionally truncate tokenized schemas if they exceed the maximum token length
        if total_len > max_tokenized_len:
            truncate_len = total_len - max_tokenized_len
            truncate_len1 = math.ceil(len(tokenized_schema1) / total_len * truncate_len)
            truncate_len2 = math.ceil(len(tokenized_schema2) / total_len * truncate_len)
            tokenized_schema1 = tokenized_schema1[:-truncate_len1]
            tokenized_schema2 = tokenized_schema2[:-truncate_len2]

        merged_tokenized_schema = (
            [tokenizer.bos_token_id] + tokenized_schema1 + newline_token + tokenized_schema2 + [tokenizer.eos_token_id]
        )

        tokenized_schemas.append(merged_tokenized_schema)

    # Add a new column for tokenized schemas and drop old ones
    df["tokenized_schema"] = tokenized_schemas
    df = df.drop(["schema1", "schema2", "filename"], axis=1)

    # Shuffle all the rows in the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def transform_data(df, tokenizer, device):
    """
    Transforms the input training DataFrame into a format suitable for our model.

    Args:
        df (pd.DataFrame): The training DataFrame with columns "Tokenized_schema" and "Label".
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing schemas.
        device (torch.device): The device (CPU or GPU) to which tensors should be moved.

    Returns:
        list: A list of dictionaries, each containing "input_ids", "attention_mask", and "labels" tensors.
    """

    max_length = max(len(schema) for schema in df["tokenized_schema"])
    pad_token_id = tokenizer.pad_token_id

    dataset = []
    for idx in range(len(df)):
        schema = df["tokenized_schema"].iloc[idx]
        label = int(df["label"].iloc[idx])

        schema_tensor = torch.tensor(schema)
        padded_schema = torch.nn.functional.pad(schema_tensor, (0, max_length - len(schema)), value=pad_token_id)
        attention_mask = (padded_schema != pad_token_id).long()
        label_tensor = torch.tensor(label)

        dictionary = {
            "input_ids": padded_schema.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": label_tensor.to(device)
        }

        dataset.append(dictionary)

    return dataset


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
    

def save_model_and_adapter(model):
    """
    Save the model's adapter and log it as a WandB artifact.

    Args:
        model: The model with the adapter to save.
    """
    path = os.path.join(os.getcwd(), "adapter-model")
    
    model = model.module if isinstance(model, nn.DataParallel) else model
    # Save the entire model
    model.save_pretrained(path)

    # Save the adapter
    model.save_adapter(path, ADAPTER_NAME)


def train_model(train_df, test_df):
    """
    Train a model
    
    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The testing dataset.

    Returns:
        None
    """

    accumulation_steps = 4
    learning_rate = 2e-6
    num_epochs = 100

    # Start a new wandb run to track this script
    wandb.require("core")
    wandb.init(
        project="custom-codebert_frequent_patterns",
        config={
            "accumulation_steps": accumulation_steps,
            "BATCH_SIZE": BATCH_SIZE,
            "dataset": "json-schemas",
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "model_name": MODEL_NAME,
        }
    )

    # Initialize tokenizer, model with adapter and classification head
    model, tokenizer = initialize_model()

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Use all available GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Merge tokenized schemas
    train_df_with_tokens = merge_schema_tokens(train_df, tokenizer) 
    test_df_with_tokens = merge_schema_tokens(test_df, tokenizer)

    # Transform data into dict
    train_data = transform_data(train_df_with_tokens, tokenizer, device)
    test_data = transform_data(test_df_with_tokens, tokenizer, device)

    # Set up scheduler to adjust the learning rate during training
    num_training_steps = num_epochs * len(train_data) // BATCH_SIZE 
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Setup the training arguments
    training_args = TrainingArguments(
        report_to="wandb",
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE, 
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
        model=model.module if isinstance(model, nn.DataParallel) else model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_accuracy,
        optimizers=(optimizer, lr_scheduler),
    )

    # Train the model
    trainer.train()
    trainer.evaluate()
        
    # Save the adapter
    save_model_and_adapter(model)
    

def load_model_and_adapter():
    """
    Load the model and adapter from the specified path.

    Returns:
        PreTrainedModel: The model with the loaded adapter.
    """
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoAdapterModel.from_pretrained(PATH)

    # Load the adapter from the saved path and activate it
    adapter_name = model.load_adapter(PATH)
    model.set_active_adapters(adapter_name)
    print(f"Loaded and activated adapter: {adapter_name}")
    
    return model, tokenizer


def get_schema_paths(schema, current_path=('$',)):
    """
    Recursively traverse the JSON schema and return the full paths of all nested properties,
    excluding schema keywords from the path.

    Args:
        schema (dict): The JSON schema object.
        current_path (tuple): The current path tuple, starting from the root ('$').

    Yields:
        tuple: A tuple representing the path of each nested property as it would appear in a JSON document.
    """
    if not isinstance(schema, dict):
        return

    # Yield the current path if it's a dictionary (nested property)
    yield current_path

    for key, value in schema.items():
        # Skip 'definitions', '$defs', and 'additionalProperties'
        if key in {"definitions", "$defs", "additionalProperties"}:
            continue

        # Update the path for nested structures
        if key == "properties":
            for prop_key, prop_value in value.items():
                # Recursively yield paths for each sub-property
                yield from get_schema_paths(prop_value, current_path + (prop_key,))

        elif key in {"allOf", "anyOf", "oneOf"}:
            for item in value:
                yield from get_schema_paths(item, current_path)

        elif key == "items":
            # Represent 'items' in the path with '*'
            yield from get_schema_paths(value, current_path + ('*',))

        # Recursively handle other nested dictionaries
        elif isinstance(value, dict):
            yield from get_schema_paths(value, current_path + (key,))


def in_schema(path, schema_paths):
    """
    Check if the given path matches exactly any path in schema_paths.

    Args:
        path (tuple): The path to check.
        schema_paths (set): A set of paths representing allowed schema paths.

    Returns:
        bool: True if the path matches exactly any schema path, False otherwise.
    """
    return path in schema_paths


def remove_additional_properties(filtered_df, schema):
    """
    Remove paths from the DataFrame that are not explicitly defined in the schema.

    Args:
        filtered_df (pd.DataFrame): DataFrame containing paths and associated data.
        schema (str): The JSON schema name.

    Returns:
        pd.DataFrame: Filtered DataFrame with paths that are explicitly defined in the schema.
    """
    # Load the schema
    schema_path = os.path.join(SCHEMA_FOLDER, schema)
    try:
        with open(schema_path, 'r') as schema_file:
            schema = jsonref.load(schema_file)
    except Exception as e:
        print(f"Error loading and dereferencing schema {schema_path}: {e}")
        return filtered_df
    
    # Get all paths in the schema
    schema_paths = set(get_schema_paths(schema))
    print(f"Schema paths: {schema_paths}")

    # Only keep rows where the 'path' exists in schema_paths
    filtered_df = filtered_df[
        filtered_df["path"].apply(lambda path: in_schema(path, schema_paths))
    ]

    return filtered_df


def find_pairs_with_common_properties(df):
    """
    Generate pairs of paths from a DataFrame where the schemas have at least two properties in common.

    Args:
        df (pd.DataFrame): A DataFrame containing path information.

    Yields:
        tuple: A tuple containing the indices of the two paths (i, j) and a set of common properties between their schemas.
    """
    properties_list = []

    # Extract properties for each row
    for i in range(len(df)):
        try:
            schema = json.loads(df.at[i, "schema"])
            properties = schema.get("properties", {})
            properties_list.append(set(properties.keys()))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping row {i} due to error: {e}")
            properties_list.append(set()) 

    # Compare each pair of schemas
    for i in range(len(df)):
        properties_i = properties_list[i]

        for j in range(i + 1, len(df)):
            properties_j = properties_list[j]
            
            # Find the common properties between the two schemas
            common_properties = properties_i & properties_j
            
            # Yield the pair if there are more than two common properties
            if len(common_properties) > 1:
                yield i, j, common_properties


def extract_properties(schema):
    """
    Extracts the top-level properties from a JSON schema.
    
    Args:
        schema (dict): The JSON schema.

    Returns:
        dict: A dictionary containing the top-level properties.
    """
    if "properties" in schema:
        return schema["properties"]
    return {}


def order_properties_by_commonality(schema1, schema2):
    """
    Orders the properties of two schemas based on their common properties.

    Args:
        schema1 (dict): The first JSON schema.
        schema2 (dict): The second JSON schema.

    Returns:
        tuple: Two dictionaries representing the schemas with properties ordered by commonality.
    """

    # Extract properties from both schemas
    properties1 = extract_properties(schema1)
    properties2 = extract_properties(schema2)

    # Find common properties
    common_properties = set(properties1.keys()) & set(properties2.keys())

    # Order properties by commonality
    ordered_properties1 = {prop: properties1[prop] for prop in common_properties}
    ordered_properties1.update({prop: properties1[prop] for prop in properties1 if prop not in common_properties})

    ordered_properties2 = {prop: properties2[prop] for prop in common_properties}
    ordered_properties2.update({prop: properties2[prop] for prop in properties2 if prop not in common_properties})

    # Create new ordered schemas
    ordered_schema1 = {**schema1, "properties": ordered_properties1}
    ordered_schema2 = {**schema2, "properties": ordered_properties2}

    return ordered_schema1, ordered_schema2


def tokenize_schema(schema, tokenizer):
    """Tokenize schema.

    Args:
        schema (dict): DataFrame containing pairs, labels, filenames, and schemas of each path in pair
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing schemas.

    Returns:
        torch.tensor: inputs_ids tensor
    """

    # Tokenize the schema
    tokenized_schema = tokenizer(json.dumps(schema), return_tensors="pt", max_length=MAX_TOK_LEN, padding="max_length", truncation=True)
    input_ids_tensor = tokenized_schema["input_ids"]
    input_ids_tensor = input_ids_tensor[input_ids_tensor != tokenizer.pad_token_id]

    # Remove the first and last tokens
    input_ids_tensor_sliced = input_ids_tensor[1:-1]

    # Convert tensor to a numpy array and then list
    input_ids_numpy = input_ids_tensor_sliced.cpu().numpy()
    input_ids_list = input_ids_numpy.tolist()
   
    return input_ids_list


def merge_eval_schema_tokens(tokenized_schema1, tokenized_schema2, tokenizer):#, truncations):
    """Merge the tokens of the schemas of the paths with in pair.

    Args:
        df (pd.DataFrame): DataFrame containing paths, tokenized schemas, filenames
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing schemas.

    Returns:
        list: List of merged tokenized schemas.
    """
    
    newline_token = tokenizer("\n")["input_ids"][1:-1]

    total_len = len(tokenized_schema1) + len(tokenized_schema2)
    max_tokenized_len = MAX_TOK_LEN - 1 - 2  # Account for BOS, EOS, and newline token lengths
    #print("total length:", total_len, "max length:", max_tokenized_len)
    # Proportionally truncate tokenized schemas if they exceed the maximum token length
    if total_len > max_tokenized_len:
        truncate_len = total_len - max_tokenized_len
        truncate_len1 = math.ceil(len(tokenized_schema1) / total_len * truncate_len)
        truncate_len2 = math.ceil(len(tokenized_schema2) / total_len * truncate_len)
        tokenized_schema1 = tokenized_schema1[:-truncate_len1]
        tokenized_schema2 = tokenized_schema2[:-truncate_len2]
        #truncations += 1

    merged_tokenized_schema = (
        [tokenizer.bos_token_id] + tokenized_schema1 + newline_token + tokenized_schema2 + [tokenizer.eos_token_id]
    )

    return merged_tokenized_schema#, truncations

    
def process_pairs(pairs, df, model, device, tokenizer):
    """
    Process schema pairs and return edges for connected pairs.

    Args:
        pairs (list[tuple]): A list of tuples, where each tuple contains the indices of the two schemas (i1, i2) and a set of common properties.
        df (pd.DataFrame): DataFrame containing the paths and schemas.
        model (PreTrainedModel): The model used for predicting connections.
        device (torch.device): The device to run the model on.
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing schemas.

    Returns:
        list[tuple]: A list of tuples where each tuple contains the paths of two schemas that are predicted to be connected.
    """
    edges = []
    
    for i1, i2, prop in pairs:
        schema1 = json.loads(df["schema"].iloc[i1])
        schema2 = json.loads(df["schema"].iloc[i2])
        ordered_schema1, ordered_schema2 = order_properties_by_commonality(schema1, schema2)
        
        # Tokenize schemas (returns lists of token IDs)
        tokenized_schema1 = tokenize_schema(ordered_schema1, tokenizer)
        tokenized_schema2 = tokenize_schema(ordered_schema2, tokenizer)
        
        # Merge the tokenized schemas
        tokenized_schema = merge_eval_schema_tokens(tokenized_schema1, tokenized_schema2, tokenizer)
        
        # Convert the list of token IDs to a tensor and create attention mask
        input_ids = torch.tensor(tokenized_schema, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)

        # Predict labels
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=-1).item()

        # Add edge if prediction is positive
        if pred == 1:
            edges.append((df["path"].iloc[i1], df["path"].iloc[i2]))

    return edges


def build_definition_graph(df, m, device, tokenizer):
    """
    Build a definition graph from tokenized schema pairs using the given model, without batching.

    Args:
        df (pd.DataFrame): A DataFrame containing paths and their tokenized schemas.
        m (PreTrainedModel): The model used for predicting connections.
        device (torch.device): The device (CPU or GPU) to run the model on.
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing schemas.

    Returns:
        nx.Graph: A graph with edges representing predicted connections between pairs.
    """
    graph = nx.Graph()
    pairs = list(find_pairs_with_common_properties(df))
    
    with tqdm.tqdm(total=len(pairs), desc="Processing pairs", position=0, leave=True) as pbar:
        edges = process_pairs(pairs, df, m, device, tokenizer)

        # Add edges to the graph
        for edge in edges:
            graph.add_edge(*edge)

        pbar.update(len(pairs))

    return graph


def find_definitions_from_graph(graph):
    """
    Find definitions from a graph representing schema connections.

    Args:
        graph (nx.Graph): A graph representing schema connections.

    Returns:
        List[List]: A list of lists, each containing nodes representing a definition.
    """
    # Get all the cliques from the graph and sort them based on their lengths in descending order
    cliques = list(nx.algorithms.find_cliques(graph))
    cliques.sort(key=lambda a: len(a), reverse=True)
    
    # Make sure a path only appears in one definition
    processed_cliques = []
    while cliques:
        # Remove the largest clique
        largest_clique = cliques.pop(0)
        # Create a set of vertices in the largest clique
        clique_set = sorted(set(largest_clique))

        # Filter out vertices from other cliques that are in the largest clique
        filtered_cliques = []
        for clique in cliques:
            filtered_clique = [c for c in clique if c not in clique_set]
            if len(filtered_clique) > 1:
                filtered_cliques.append(filtered_clique)
        
        processed_cliques.append(list(clique_set))
        cliques = filtered_cliques

    return processed_cliques
 

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
    

def calc_scores(actual_clusters, predicted_clusters, threshold=1.0):
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


def evaluate_single_schema(schema, device, test_ground_truth):
    """
    Helper function to evaluate a single schema.

    Args:
        schema (str): The schema filename to be evaluated.
        device (torch.device): The device on which to run the model.
        test_ground_truth (dict): Ground truth clusters for test schemas.

    Returns:
        tuple: Contains schema name, precision, recall, F1 score, sorted actual clusters, and sorted predicted clusters.
        None: If the schema could not be processed.
    """

    # Load the model and tokenizer
    m, tokenizer = load_model_and_adapter()

    # Enable multiple GPU usage with DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for model parallelism.")
        m = nn.DataParallel(m)

    m.to(device)
    m.eval()

    filtered_df, frequent_ref_defn_paths, schema_name, failure_flags = process_data.process_schema(schema, "")
        
    if filtered_df is not None and frequent_ref_defn_paths is not None:
        # Build a definition graph
        graph = build_definition_graph(filtered_df, m, device, tokenizer)
        
        # Predict clusters and sort each predicted cluster alphabetically
        predicted_clusters = [sorted(cluster) for cluster in find_definitions_from_graph(graph)]

        # Get the ground truth clusters and sort each actual cluster alphabetically
        ground_truth_dict = test_ground_truth.get(schema_name, {})
        actual_clusters = [[sorted(tuple(inner_list)) for inner_list in outer_list] for outer_list in ground_truth_dict.values()]

        print(f"Schema: {schema_name}")
        print(f"Actual clusters: {actual_clusters}")
        print(f"Predicted clusters: {predicted_clusters}")
        # Calculate the precision, recall, and F1-score
        precision, recall, f1_score = calc_scores(actual_clusters, predicted_clusters)

        # Return evaluation metrics and sorted clusters for the schema
        return schema, precision, recall, f1_score, actual_clusters, predicted_clusters
        
    return None


def evaluate_data(test_ground_truth, output_file="evaluation_results.json"):
    """
    Evaluate the model on the entire test data using multiple CPUs and store results in a JSON file.

    Args:
        test_ground_truth (dict): Dictionary containing ground truth information for test data.
        output_file (str): Path to the JSON file to store the evaluation results.
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the test schemas
    test_schemas = list(test_ground_truth.keys())
    f1_scores = []
    results = []

    # Process schemas in parallel using ProcessPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(evaluate_single_schema, schema, device, test_ground_truth): schema 
            for schema in test_schemas
        }

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=1):
            result = future.result()
            if result is not None:
                schema, precision, recall, f1_score, actual_clusters, predicted_clusters = result
                f1_scores.append(f1_score)

                # Collect results for each schema
                results.append({
                    "schema": schema,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "actual_clusters": actual_clusters,
                    "predicted_clusters": predicted_clusters
                })

    # Calculate average F1-score and add it to results
    if f1_scores:
        avg_f1_score = sum(f1_scores) / len(f1_scores)
        results.append({"average_f1_score": avg_f1_score})

    # Write results to JSON file
    with open(output_file, "w") as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {output_file}")


def group_paths(df, test_ground_truth, min_common_keys=2, output_file="evaluation_results_baseline_model.json"):
    """
    Group paths that have at least a specified number of distinct nested keys in common per filename and evaluate.

    Args:
        df (pd.DataFrame): DataFrame containing columns "path", "distinct_nested_keys", and "filename".
        test_ground_truth (dict): Dictionary containing ground truth clusters for test data.
        min_common_keys (int, optional): Minimum number of distinct nested keys required for paths to be grouped together. Default is 2.
        output_file (str, optional): Path to the output JSON file. Default is "evaluation_results_baseline_model.json".

    Returns:
        None
    """
    all_results = []  # To accumulate results for each filename
    f1_scores = []  # To store F1 scores for each schema

    # Iterate through each filename group
    for filename, group in tqdm.tqdm(df.groupby("filename"), position=1, leave=False, total=len(df["filename"].unique()), desc="Grouping paths"):
        paths = group["path"].tolist()
        distinct_keys = group["distinct_nested_keys"].tolist()

        # Dictionary to track paths based on shared keys
        group_dict = defaultdict(set)

        # Process each path and its associated keys
        for path, keys in zip(paths, distinct_keys):
            # Convert the strings to actual sets
            path = ast.literal_eval(path)
            keys = frozenset(ast.literal_eval(keys))
            
            added_to_group = False

            # Only start checking if there are groups in group_dict
            if group_dict:
                # Attempt to find or create a group for the current path
                for existing_keys in list(group_dict.keys()):
                    # Check if this path has sufficient keys in common with an existing group
                    if len(keys & existing_keys) >= min_common_keys:
                        # Add the path to the existing group
                        group_dict[existing_keys].append(path)
                        added_to_group = True
                        break

            # If no existing group was found, add a new group for this path
            if not added_to_group:
                group_dict[keys] = [path]
        
        # Filter groups to keep only valid ones with more than one path
        predicted_clusters = [sorted(group) for group in group_dict.values() if len(group) > 1]
        predicted_clusters = sorted(predicted_clusters)


        # Get the actual clusters for the current filename
        defn_paths_dict = test_ground_truth.get(filename, {})
        actual_clusters = [sorted([tuple(path) for path in cluster]) for cluster in sorted(defn_paths_dict.values())]

        # Evaluate the predicted clusters against the ground truth
        precision, recall, f1_score = calc_scores(actual_clusters, predicted_clusters)

        # Store individual F1 scores for average calculation later
        f1_scores.append(f1_score)

        # Collect results for the current schema
        results = {
            "schema": filename,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "actual_clusters": actual_clusters,
            "predicted_clusters": predicted_clusters
        }
        all_results.append(results)

    # Calculate average F1 score if there are any scores
    if f1_scores:
        avg_f1_score = sum(f1_scores) / len(f1_scores)
        all_results.append({"average_f1_score": avg_f1_score})

    # Write all results to the specified JSON file
    with open(output_file, "w") as json_file:
        json.dump(all_results, json_file, indent=4)


    

def get_definition(ref, schema):
    """
    Retrieve the object definition from a JSON schema based on a `$ref` reference string.

    Args:
        ref (str): The `$ref` reference string pointing to a specific definition within the schema.
        schema (dict): The full JSON schema containing definitions.

    Returns:
        dict: The dereferenced object from the schema corresponding to the `$ref` string.

    Raises:
        KeyError: If the `$ref` path does not exist in the schema.
    """
    try:
        if ref.startswith("#/definitions/"):
            parts = ref.split("/")
            definition = schema["definitions"]
            for part in parts[2:]:
                if part:
                    definition = definition[part]
            return definition

        elif ref.startswith("$defs/"):
            parts = ref.split("/")
            definition = schema["$defs"]
            for part in parts[1:]:
                if part:
                    definition = definition[part]
            return definition

    except KeyError as e:
        raise KeyError(f"Reference {ref} could not be resolved in schema.") from e

    # Unhandled reference format
    return None


def dereference_schema(schema_path, excluded_definitions):

     # Convert the excluded_definitions set to a string or JSON representation
    excluded_definitions_str = json.dumps(list(excluded_definitions))

    # Build the command to call the Node.js script with the file paths
    command = ["node", "dereference_schema.js", schema_path, excluded_definitions_str]

    # Run the Node.js script and capture the output
    result = subprocess.run(command, capture_output=True, text=True)

    # Handle the output and extract the size in kilobytes
    if result.returncode == 0:
        output = result.stdout.strip()
        if output: 
            return output
        else:
            print("Error: Node.js script returned an empty output")
            return None
    else:
        print(f"Error executing Node.js script: {result.stderr}")
        return None


def get_dereferenced_schema_size():
    """
    Calls the Node.js script to get the dereferenced schema size while excluding specified definitions.

    Args:
        schema (dict): The schema to be dereferenced.
        excluded_definitions (list): List of definitions to exclude (in the format #/definitions/DefinitionName).

    Returns:
        float: The size of the dereferenced schema in kilobytes.
    """
    try:
        with open("test_ground_truth.json", 'r') as json_file:
            test_ground_truth_list = [json.loads(line) for line in json_file]

    except json.JSONDecodeError as e:
        print(f"Error loading JSON file: {e}")
        return

    for test_ground_truth in test_ground_truth_list:
        for schema_name, ground_truth in test_ground_truth.items():
            ref_definitions = list(ground_truth.keys())
            schema_path = os.path.join(SCHEMA_FOLDER, schema_name)
            print(dereference_schema(schema_path, ref_definitions))

                