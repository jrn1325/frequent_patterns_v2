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
import wandb

from adapters import AdapterTrainer, AutoAdapterModel
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler, TrainingArguments, EvalPrediction

import process_data


MAX_TOK_LEN = 512
MODEL_NAME = "microsoft/codebert-base" 
PATH = "./adapter-model"
ADAPTER_NAME = "frequent_patterns"
SCHEMA_FOLDER = "processed_schemas"
JSON_FOLDER = "processed_jsons"
BATCH_SIZE = 16
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
    return model, tokenizer


def tokenize_schema(schema, tokenizer):
    """
    Tokenizes the input schema text with a maximum length constraint.

    Args:
        schema (str): The schema text to be tokenized.
        tokenizer (PreTrainedTokenizer): The tokenizer for processing the schema text.

    Returns:
        list: A list of token IDs representing the tokenized schema.
    """
    tokens = tokenizer(schema, add_special_tokens=False, truncation=True, max_length=MAX_TOK_LEN)["input_ids"]
    return tokens


def merge_schema_tokens(df, tokenizer):
    """
    Merges tokenized schemas of pairs into a single sequence with a newline token in between.
    Truncates proportionally if the tokenized length exceeds the maximum token length.

    Args:
        df (pd.DataFrame): DataFrame containing schema pairs to be tokenized and merged.
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing schemas.

    Returns:
        pd.DataFrame: Updated DataFrame with a single 'tokenized_schema' column containing merged tokens.
    """
    # Special tokens
    cls_token_id = tokenizer.cls_token_id  # [CLS]
    sep_token_id = tokenizer.sep_token_id  # [SEP]
    eos_token_id = tokenizer.eos_token_id  # [EOS]
    tokenized_schemas = []

    for idx, (schema1, schema2) in df[["schema1", "schema2"]].iterrows():
        schema1 = json.loads(schema1)
        schema2 = json.loads(schema2)

        # Ensure that properties are ordered by commonality if needed (implementation dependent)
        ordered_schema1, ordered_schema2 = order_properties_by_commonality(schema1, schema2)

        tokenized_schema1 = tokenize_schema(ordered_schema1, tokenizer)
        tokenized_schema2 = tokenize_schema(ordered_schema2, tokenizer)

        total_len = len(tokenized_schema1) + len(tokenized_schema2)
        max_tokenized_len = MAX_TOK_LEN - 3  # Account for BOS, EOS, and newline token lengths

        if total_len > max_tokenized_len:
            truncate_len = total_len - max_tokenized_len
            truncate_len1 = math.ceil(len(tokenized_schema1) / total_len * truncate_len)
            truncate_len2 = math.ceil(len(tokenized_schema2) / total_len * truncate_len)
            tokenized_schema1 = tokenized_schema1[:-truncate_len1]
            tokenized_schema2 = tokenized_schema2[:-truncate_len2]

        merged_tokenized_schema = (
            [cls_token_id] +  # Add [CLS] at the beginning
            tokenized_schema1 +
            [sep_token_id] +  # Add [SEP] between schemas
            tokenized_schema2 +
            [eos_token_id]  # Add [EOS] at the end
        )

        tokenized_schemas.append(merged_tokenized_schema)

    df["tokenized_schema"] = tokenized_schemas
    df = df.drop(["schema1", "schema2", "filename"], axis=1)

    return df


def compute_metrics(pred: EvalPrediction):
    """
    Computes accuracy, precision, recall, and F1 score for evaluation.
    
    Args:
        pred (EvalPrediction): An EvalPrediction object containing predictions and labels.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    logits, labels = pred.predictions, pred.label_ids
    predictions = logits.argmax(-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    
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


class SchemaDataset(Dataset):
    """
    Custom dataset class for tokenized schema data.

    Args:
        df (pd.DataFrame): The DataFrame containing tokenized schemas and labels.
        tokenizer (PreTrainedTokenizer): The tokenizer for padding and encoding.
    """
    def __init__(self, df, tokenizer):
        self.tokenizer = tokenizer
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves the tokenized schema and its label for the given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing input_ids, attention_mask, and label.
        """
        tokenized_schema = self.df.iloc[idx]["tokenized_schema"]
        label = self.df.iloc[idx]["label"]

        # Pad the tokenized schema to the max length using the tokenizer's pad_token_id
        encoding = self.tokenizer.pad(
            {"input_ids": [tokenized_schema], 
             "attention_mask": [[1] * len(tokenized_schema)]}, 
            padding="max_length",  
            max_length=MAX_TOK_LEN, 
            return_tensors="pt" 
        )
        # Return the encoded data along with the label
        return {
            "input_ids": encoding["input_ids"].squeeze(0), 
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label)
        }


def create_dataset(df, tokenizer):
    """
    Creates a custom dataset from the provided DataFrame with tokenized schema and labels.
    
    Args:
        df (pd.DataFrame): DataFrame containing the tokenized schema and labels.
        tokenizer (PreTrainedTokenizer): The tokenizer used to process the schemas.
    
    Returns:
        SchemaDataset: Custom dataset ready for training.
    """
    return SchemaDataset(df, tokenizer)


def train_model(train_df, test_df):
    """
    Trains the model using an AdapterTrainer, logging metrics with wandb and evaluating at each epoch.

    Args:
        train_df (pd.DataFrame): DataFrame with training data.
        test_df (pd.DataFrame): DataFrame with validation data.
    """
    # Hyperparameters
    accumulation_steps = 4
    learning_rate = 1e-6
    num_epochs = 50

    # Initialize wandb logging
    wandb.init(
        project="custom-codebert_frequent_patterns",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": BATCH_SIZE
        }
    )

    # Initialize model and tokenizer
    model, tokenizer = initialize_model()

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Use all available GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Move the model to the appropriate device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Merge tokens for each schema in the training and testing DataFrames
    train_df = merge_schema_tokens(train_df, tokenizer)
    test_df = merge_schema_tokens(test_df, tokenizer)

    # Create datasets with dynamic padding and batching
    train_dataset = create_dataset(train_df, tokenizer)
    test_dataset = create_dataset(test_df, tokenizer)

    # Set up scheduler to adjust the learning rate during training
    num_training_steps = num_epochs * len(train_dataset) // BATCH_SIZE 
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

    trainer = AdapterTrainer(
        model=model.module if isinstance(model, nn.DataParallel) else model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler),
    )

    # Start training
    trainer.train()
    trainer.evaluate()

    # Save the model and adapter after training
    save_model_and_adapter(model)

    # Finish wandb logging
    wandb.finish()


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
    return schema.get("properties", {})


def find_common_properties(properties1, properties2):
    """
    Finds common properties between two dictionaries.

    Args:
        properties1 (dict): The properties of the first schema.
        properties2 (dict): The properties of the second schema.

    Returns:
        set: A set of common property keys.
    """
    return set(properties1.keys()) & set(properties2.keys())


def order_properties(properties, common_keys):
    """
    Orders a dictionary of properties by prioritizing common keys.

    Args:
        properties (dict): The properties to order.
        common_keys (set): A set of keys that are common between schemas.

    Returns:
        OrderedDict: The ordered properties.
    """
    ordered = OrderedDict((key, properties[key]) for key in common_keys if key in properties)
    ordered.update((key, properties[key]) for key in properties if key not in common_keys)
    return ordered


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
    common_properties = find_common_properties(properties1, properties2)

    # Order properties
    ordered_properties1 = order_properties(properties1, common_properties)
    ordered_properties2 = order_properties(properties2, common_properties)

    # Create updated schemas with reordered properties
    ordered_schema1 = {**schema1, "properties": dict(ordered_properties1)}
    ordered_schema2 = {**schema2, "properties": dict(ordered_properties2)}

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


def merge_eval_schema_tokens(pairs, df, tokenizer):
    """
    Merge the tokens of two schemas for evaluation, with truncation if necessary to fit the maximum token length.

    Args:
        pairs (list): A list of tuples containing the indices of the two schemas and a set of common properties.
        df (pd.DataFrame): DataFrame containing the paths and schemas.
        tokenizer (PreTrainedTokenizer): Tokenizer used for processing schemas.

    Returns:
        list: A list of merged tokenized schemas, adhering to token length constraints.
    """
    
    # Special tokens
    cls_token_id = tokenizer.cls_token_id  # [CLS]
    sep_token_id = tokenizer.sep_token_id  # [SEP]
    eos_token_id = tokenizer.eos_token_id  # [EOS]

     # Loop through the entire batch passed in as 'pairs'
    for i1, i2, _ in pairs:
        schema1 = json.loads(df["schema"].iloc[i1])
        schema2 = json.loads(df["schema"].iloc[i2])

    # Order properties by commonality
    ordered_schema1, ordered_schema2 = order_properties_by_commonality(schema1, schema2)
    
    tokenized_schema1 = tokenize_schema(ordered_schema1, tokenizer)
    tokenized_schema2 = tokenize_schema(ordered_schema2, tokenizer)
    
    # Calculate the total length of the merged tokenized schemas
    total_len = len(tokenized_schema1) + len(tokenized_schema2)
    max_tokenized_len = MAX_TOK_LEN - 1 - 2  # Subtract BOS, EOS, and newline token lengths

    # Truncate the schemas proportionally if they exceed the max token length
    if total_len > max_tokenized_len:
        truncate_len = total_len - max_tokenized_len
        truncate_len1 = math.ceil(len(tokenized_schema1) / total_len * truncate_len)
        truncate_len2 = math.ceil(len(tokenized_schema2) / total_len * truncate_len)
        tokenized_schema1 = tokenized_schema1[:-truncate_len1]
        tokenized_schema2 = tokenized_schema2[:-truncate_len2]

    merged_tokenized_schema = (
        [cls_token_id] +  # Add [CLS] at the beginning
        tokenized_schema1 +
        [sep_token_id] +  # Add [SEP] between schemas
        tokenized_schema2 +
        [eos_token_id]  # Add [EOS] at the end
    )

    return merged_tokenized_schema


def process_pairs(pairs, df, model, device, tokenizer):
    """
    Process a batch of schema pairs and return edges for connected pairs.

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
    batch_input_ids = []
    batch_attention_masks = []

    # Tokenize and merge schemas
    tokenized_schema = merge_eval_schema_tokens(pairs, df, tokenizer)

    # Convert tokens to tensors 
    input_ids = torch.tensor(tokenized_schema, dtype=torch.long).to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    batch_input_ids.append(input_ids)
    batch_attention_masks.append(attention_mask)

    # Pad batch tensors to have the same length, then stack
    batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True)
    batch_attention_masks = torch.nn.utils.rnn.pad_sequence(batch_attention_masks, batch_first=True)

    # Run the batch through the model
    with torch.no_grad():
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
        preds = torch.argmax(outputs.logits, dim=-1).tolist()

    # Append edges based on predictions
    for idx, pred in enumerate(preds):
        if pred == 1:
            i1, i2, _ = pairs[idx]
            edges.append((df["path"].iloc[i1], df["path"].iloc[i2]))

    return edges


def build_definition_graph(df, model, device, tokenizer, schema_name):
    """
    Build a definition graph from tokenized schema pairs using the given model, with batching.

    Args:
        df (pd.DataFrame): A DataFrame containing paths and their tokenized schemas.
        model (PreTrainedModel): The model used for predicting connections.
        device (torch.device): The device (CPU or GPU) to run the model on.
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing schemas.
        schema_name (str): The name of the schema.

    Returns:
        nx.Graph: A graph with edges representing predicted connections between pairs.
    """
    graph = nx.Graph()
    pairs = list(find_pairs_with_common_properties(df))
    
    edges = []
    with tqdm(total=len(pairs), desc="Processing pairs for " + schema_name, position=0, leave=True) as pbar:
        for batch_start in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[batch_start: batch_start + BATCH_SIZE]
            batch_edges = process_pairs(batch, df, model, device, tokenizer)
            edges.extend(batch_edges)
            pbar.update(len(batch))

    # Add edges to the graph
    graph.add_edges_from(edges)
    
    return graph


def find_definitions_from_graph(graph):
    """
    Find definitions from a graph representing schema connections.

    Args:
        graph (nx.Graph): A graph representing schema connections.

    Returns:
        List[List]: A list of lists, each containing nodes representing a definition.
    """
    # Find all maximal cliques in the graph
    cliques = list(nx.algorithms.find_cliques(graph))
    cliques.sort(key=len, reverse=True)  # Sort cliques by size in descending order
    
    # Track used nodes to ensure each node is part of only one definition
    processed_cliques = []
    used_nodes = set()

    for clique in cliques:
        # Exclude nodes already processed
        unique_clique = [node for node in clique if node not in used_nodes]

        if unique_clique:
            processed_cliques.append(unique_clique)
            used_nodes.update(unique_clique)  # Mark nodes as processed

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
    """Use schema definition properties (actuals) to evaluate the predicted clusters.
    TP: Definition from the schema that exists in the file.
    FP: Definition from the file that does not exist in schema.
    FN: Definition from the schema that does not exist in the file.
    # Precision: for all things you found, how many of them you should have found?
    # Recall: for all things you should have found, how many did you find?

    Args:
        actual_clusters (list): clusters found in the json schemas.
        predicted_clusters (_type_): clusters found in the json files.
        threshold (float): minimum jaccard similarity score.

    Returns:
        precision (float): Precision score.
        recall (float): Recall score.
        f1_score (float): F1-score.
        matched_definitions (list): List of matched definitions.
        matched_paths (set): Set of matched paths.
    """
 
    if not predicted_clusters:
        return 0, 0, 0, [], set()

    TP = 0
    FP = 0
    FN = 0
    matched_definitions = []  # To store matched clusters and their keys
    matched_paths = set()  # To store matched paths
        
    for actual_cluster in actual_clusters:
        found_match = False
        for predicted_cluster in predicted_clusters:
            jaccard_index = calc_jaccard_index(set(actual_cluster), set(predicted_cluster))
            if jaccard_index >= threshold:
                found_match = True
                TP += 1

                #matched_definitions.append(definition)
                for path in actual_cluster:
                    matched_paths.add(path)
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
    
    return precision, recall, f1_score, matched_definitions, matched_paths


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
        graph = build_definition_graph(filtered_df, m, device, tokenizer, schema_name)
        
        # Predict clusters and sort each predicted cluster alphabetically
        predicted_clusters = [sorted(cluster) for cluster in find_definitions_from_graph(graph)]

        # Get the ground truth clusters and sort each actual cluster alphabetically
        defn_paths_dict = test_ground_truth.get(schema_name, {})
        actual_clusters = [sorted([tuple(path) for path in cluster]) for cluster in sorted(defn_paths_dict.values())]
     
        print(f"Schema: {schema_name}")
        print(f"Actual clusters: {actual_clusters}")
        print(f"Predicted clusters: {predicted_clusters}")
        # Calculate the precision, recall, and F1-score
        precision, recall, f1_score, _, _ = calc_scores(actual_clusters, predicted_clusters)
        print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")

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

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=1):
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
        min_common_keys (int, optional): Minimum number of distinct nested keys required for paths to be grouped together. Default is 1.
        output_file (str, optional): Path to the output JSON file. Default is "evaluation_results_baseline_model.json".

    Returns:
        None
    """
    all_results = []  # To accumulate results for each filename
    f1_scores = []  # To store F1 scores for each schema

    # Iterate through each filename group
    for filename, group in tqdm(df.groupby("filename"), position=1, leave=False, total=len(df["filename"].unique()), desc="Grouping paths"):
        paths = group["path"].tolist()
        distinct_keys = group["distinct_nested_keys"].tolist()

        # Dictionary to track paths based on shared keys
        group_dict = defaultdict(list)

        # Process each path and its associated keys
        for path, keys in zip(paths, distinct_keys):
            path = ast.literal_eval(path)  # Convert string to list
            keys = frozenset(ast.literal_eval(keys))  # Convert string to frozenset
            
            added_to_group = False

            # Attempt to add path to an existing group
            for existing_keys in list(group_dict.keys()):
                if len(keys & existing_keys) >= min_common_keys:  # Check for key overlap
                    group_dict[existing_keys].append(path)
                    added_to_group = True
                    break

            # Create a new group if no existing group matches
            if not added_to_group:
                group_dict[keys] = [path]

        # Filter groups with more than one path and sort paths within groups
        predicted_clusters = [sorted(group) for group in group_dict.values() if len(group) > 1]
        predicted_clusters = sorted(predicted_clusters)  # Sort groups for consistency

        # Get the actual clusters for the current filename
        defn_paths_dict = test_ground_truth.get(filename, {})
        actual_clusters = [(sorted([tuple(path) for path in cluster]), defn) for defn, cluster in defn_paths_dict.items()]

        # Evaluate the predicted clusters against the ground truth
        precision, recall, f1_score, matched_definitions, matched_paths = calc_scores(actual_clusters, predicted_clusters)

        # Store individual F1 scores for average calculation later
        f1_scores.append(f1_score)

        # Collect results for the current schema
        results = {
            "schema": filename,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "actual_clusters": actual_clusters,
            "predicted_clusters": predicted_clusters,
            "matched_definitions": matched_definitions,
            "matched_paths": list(matched_paths),
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


def dereference_schema(schema_path, excluded_paths):
    # Convert the excluded_paths to a JSON representation
    excluded_paths_str = json.dumps(excluded_paths)
    
    # Log the paths for debugging
    print(f"Running with schema_path: {schema_path} and excluded_paths: {excluded_paths_str}")

    # Build the command to call the Node.js script with the file paths
    command = ["node", "dereference_schema.js", schema_path, excluded_paths_str]

    # Run the Node.js script and capture the output
    result = subprocess.run(command, capture_output=True, text=True)

    # Handle the output and extract the sizes
    if result.returncode == 0:
        sizes = result.stdout.strip()
        print(sizes)
        return sizes

    else:
        print(f"Error executing Node.js script: {result.stderr}")
        return None


def get_dereferenced_schema_size():
    """
    Calls the Node.js script to get the dereferenced schema size while excluding specified paths.

        Returns:
        float: The size of the dereferenced schema in kilobytes.
    """
    try:
        with open("evaluation_results_baseline_model.json", 'r') as json_file:
            json_data = json.load(json_file)

    except json.JSONDecodeError as e:
        print(f"Error loading JSON file: {e}")
        return
    
    for entry in json_data[:-1]:  # Ensure the loop handles all the entries except the last one
        schema_name = entry["schema"]
        matched_paths = entry["matched_definitions"]
        schema_path = os.path.join(SCHEMA_FOLDER, schema_name)

        # Calling the dereference_schema function with matched_paths (partial definition)
        result = dereference_schema(schema_path, matched_paths)
        

