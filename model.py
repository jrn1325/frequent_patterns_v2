import ast
import concurrent.futures
import json
import jsonref
import math
import networkx as nx
import os
import pandas as pd
import subprocess
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from adapters import AutoAdapterModel
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, EvalPrediction

import process_data

MAX_TOK_LEN = 512
MODEL_NAME = "microsoft/codebert-base" 
ADAPTER_NAME = "frequent_patterns"
SCHEMA_FOLDER = "processed_schemas"
JSON_FOLDER = "processed_jsons"
BATCH_SIZE = 120
HIDDEN_SIZE = 768
JSON_SUBSCHEMA_KEYWORDS = {"allOf", "oneOf", "anyOf", "not"}
SCHEMA_KEYWORS = ["$defs", "definitions", "properties", "additionalProperties", "patternProperties", "items", "allOf", "anyOf", "oneOf"]


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


def order_properties(properties, common_properties):
    """
    Orders a dictionary of properties by prioritizing common properties.

    Args:
        properties (dict): The properties to order.
        common_properties (set): A set of properties that are common between schemas.

    Returns:
        OrderedDict: The ordered properties.
    """
    ordered = OrderedDict((property, properties[property]) for property in common_properties if property in properties)
    ordered.update((property, properties[property]) for property in properties if property not in common_properties)
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

    # Get common properties
    common_properties = find_common_properties(properties1, properties2)

    # Order properties
    ordered_properties1 = order_properties(properties1, common_properties)
    ordered_properties2 = order_properties(properties2, common_properties)

    # Create updated schemas with reordered properties
    ordered_schema1 = {**schema1, "properties": dict(ordered_properties1)}
    ordered_schema2 = {**schema2, "properties": dict(ordered_properties2)}

    return ordered_schema1, ordered_schema2


def tokenize_schema(schema, tokenizer):
    """
    Tokenizes the input schema text with a maximum length constraint.

    Args:
        schema (str): The schema text to be tokenized.
        tokenizer (PreTrainedTokenizer): The tokenizer for processing the schema text.

    Returns:
        list: A list of token IDs representing the tokenized schema.
    """
    tokens = tokenizer(json.dumps(schema), return_tensors="pt")["input_ids"]
    tokens = tokens.squeeze(0).tolist()
    tokens = tokens[1:-1]
    return tokens


def merge_schema_tokens(df, tokenizer):
    """
    Merges tokenized schemas and includes a similarity indicator based on the second element of their paths
    in the merged tokenized schema. Truncates proportionally if the tokenized length exceeds the maximum token length.

    Args:
        df (pd.DataFrame): DataFrame containing schema pairs and their paths.
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing schemas.

    Returns:
        pd.DataFrame: Updated DataFrame with a single 'tokenized_schema' column.
    """
    # Special tokens
    bos_token_id = tokenizer.bos_token_id  # [BOS]
    sep_token_id = tokenizer.sep_token_id  # [SEP]
    eos_token_id = tokenizer.eos_token_id  # [EOS]
    tokenized_schemas = []

    # Add tqdm progress bar for the iteration
    for idx, (schema1, schema2) in tqdm(df[["schema1", "schema2"]].iterrows(), leave=False, total=len(df), desc="Merging schema tokens"):
        schema1 = json.loads(schema1)
        schema2 = json.loads(schema2)

        # Ensure that properties are ordered by commonality
        ordered_schema1, ordered_schema2 = order_properties_by_commonality(schema1, schema2)

        # Tokenize schemas
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

        # Merge tokens including the similarity token
        merged_tokenized_schema = (
            [bos_token_id] +  # Add [BOS] at the beginning
            tokenized_schema1 +
            [sep_token_id] +  # Add [SEP] between schemas
            tokenized_schema2 +
            [eos_token_id]  # Add [EOS] at the end
        )

        tokenized_schemas.append(merged_tokenized_schema)

    # Add tokenized schema to DataFrame
    df["tokenized_schema"] = tokenized_schemas
    df = df.drop(["schema1", "schema2", "filename"], axis=1)

    return df


def compare_prefixes(path1, path2):
    """
    Compares the prefixes (all elements except the last) of two paths.

    Args:
        path1 (list or tuple): The first path, e.g., a list or tuple of elements.
        path2 (list or tuple): The second path, e.g., a list or tuple of elements.

    Returns:
        int: 1 if the prefixes are the same, 0 otherwise.
    """

    # Extract prefixes
    prefix1 = path1[:-1]
    prefix2 = path2[:-1]

    # Compare prefixes
    return 1 if prefix1 == prefix2 else 0


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset class for training and testing the model.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the tokenized schemas and labels.

    Returns:
        dict: A dictionary containing the input IDs, attention mask, label, and nesting depths of the schemas.
    """

    
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized_schema = self.data.iloc[idx]["tokenized_schema"]
        label = self.data.iloc[idx]["label"]
        path1_freq = self.data.iloc[idx]["path1_freq"] # Maybe consioder the difference of the nesting level
        path2_freq = self.data.iloc[idx]["path2_freq"]
        nested_key_freq1 = self.data.iloc[idx]["nested_key_freq1"]
        nested_key_freq2 = self.data.iloc[idx]["nested_key_freq2"]
        nesting_depth1 = self.data.iloc[idx]["nesting_depth1"]
        nesting_depth2 = self.data.iloc[idx]["nesting_depth2"]
        similarity = self.data.iloc[idx]["cosine_similarity"]
        
        return {
            "input_ids": tokenized_schema,
            "attention_mask": [1] * len(tokenized_schema),
            "label": label,
            "extra_features": [nesting_depth1, nesting_depth2, similarity, path1_freq, path2_freq],# nested_key_freq1, nested_key_freq2]
        }


def collate_fn(batch, tokenizer):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    extra_features = torch.tensor([item["extra_features"] for item in batch])

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "label": labels,
        "extra_features": extra_features
    }


class CustomEvalDataset(Dataset):
    """
    Custom PyTorch Dataset class for Evaluating the model.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the tokenized schemas.

    Returns:
        dict: A dictionary containing the input IDs, attention mask and nesting depths of the schemas.
    """

    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized_schema = self.data.iloc[idx]["tokenized_schema"]
        path1_freq = self.data.iloc[idx]["path1_freq"]
        path2_freq = self.data.iloc[idx]["path2_freq"]
        nested_key_freq1 = self.data.iloc[idx]["nested_key_freq1"]
        nested_key_freq2 = self.data.iloc[idx]["nested_key_freq2"]
        nesting_depth1 = self.data.iloc[idx]["nesting_depth1"]
        nesting_depth2 = self.data.iloc[idx]["nesting_depth2"]
        similarity = self.data.iloc[idx]["cosine_similarity"]
        path1 = self.data.iloc[idx]["path1"]
        path2 = self.data.iloc[idx]["path2"]

        return {
            "input_ids": tokenized_schema,
            "attention_mask": [1] * len(tokenized_schema),
            "extra_features": [nesting_depth1, nesting_depth2, similarity, path1_freq, path2_freq],# nested_key_freq1, nested_key_freq2],
            "path1": path1,
            "path2": path2
        }
    

def collate_eval_fn(batch, tokenizer):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    extra_features = torch.tensor([item["extra_features"] for item in batch])
    path1 = [item["path1"] for item in batch]
    path2 = [item["path2"] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "extra_features": extra_features,
        "path1": path1,
        "path2": path2
    }


class CustomBERTModel(nn.Module):
    def __init__(self):
        super(CustomBERTModel, self).__init__()
        
        # Load pre-trained CodeBERT with adapters
        self.codebert = AutoAdapterModel.from_pretrained(MODEL_NAME)

        # Add the adapter and classification head
        self.codebert.add_adapter(ADAPTER_NAME, config="seq_bn")
        self.codebert.set_active_adapters(ADAPTER_NAME)
        self.codebert.add_classification_head(ADAPTER_NAME, num_labels=2) 
        self.codebert.train_adapter(ADAPTER_NAME)

        # Custom layers for processing additional inputs
        self.extra_features = nn.Linear(5, HIDDEN_SIZE) 

        # Final classifier to process concatenated logits and custom features
        self.modified_classifier = nn.Linear(HIDDEN_SIZE + 2, 2)


    def forward(self, input_ids, attention_mask, extra_features):
        # Pass input through the pre-trained CodeBERT with the adapter
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the logits from the classification head
        logits = outputs.logits

        # Process the additional input features with the custom layer
        extra_features_emb = F.relu(self.extra_features(extra_features.float()))

        # Concatenate the CodeBERT logits and the processed feature embeddings
        combined_output = torch.cat([logits, extra_features_emb], dim=-1)
        
        # Pass the combined output through the final classification layer
        logits = self.modified_classifier(combined_output)
    
        return logits
            

def normalize_nesting_depths(df, eval_mode, max_depth=6):
    """
    Calculates and normalizes the nesting depths by dividing by max_depth for a DataFrame, with optional capping at max_depth.

    Args:
        df (pd.DataFrame): DataFrame containing path1 and path2 columns.
        max_depth (int): Maximum allowable depth for normalization.
        eval_mode (bool): Whether the DataFrame is for evaluation or training.

    Returns:
        pd.DataFrame: DataFrame with normalized nesting depths.
    """

    if eval_mode:
        df["nesting_depth1"] = df["path1"].apply(lambda path: min(len(path), max_depth) / max_depth)
        df["nesting_depth2"] = df["path2"].apply(lambda path: min(len(path), max_depth) / max_depth)
    else:
        df["nesting_depth1"] = df["path1"].apply(lambda path: min(len(ast.literal_eval(path)), max_depth) / max_depth)
        df["nesting_depth2"] = df["path2"].apply(lambda path: min(len(ast.literal_eval(path)), max_depth) / max_depth)
        
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

    
def save_model_and_adapter(model, save_path="frequent_pattern_model"):
    """
    Save the custom model's state_dict and the adapter.

    Args:
        model (CustomBERTModel): The custom model to save.
    """

    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path = os.path.join(os.getcwd(), save_path)

    # Handle multi-GPU models
    model = model.module if isinstance(model, nn.DataParallel) else model

    # Save the state_dict for the custom layers
    torch.save(model.state_dict(), os.path.join(path, "frequent_patterns_model_state_dict.pth"))

    # Save the CodeBERT adapter and base model
    model.codebert.save_pretrained(path)
    model.codebert.save_adapter(path, ADAPTER_NAME)  


def train_model(train_df, test_df):
    """
    Trains the model using an AdapterTrainer, logging metrics with wandb and evaluating at each epoch.

    Args:
        train_df (pd.DataFrame): DataFrame with training data.
        test_df (pd.DataFrame): DataFrame with validation data.
    """
    # Hyperparameters
    accumulation_steps = 4
    learning_rate = 2e-6
    num_epochs = 25

    # Initialize wandb logging
    wandb.init(
        project="custom-codebert_frequent_patterns",
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": BATCH_SIZE
        }
    )

    # Initialize the custom model and tokenizer
    model = CustomBERTModel()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Use all available GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Move the model to the appropriate device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Process and normalize nesting depths for training and testing
    train_df = normalize_nesting_depths(train_df, False, max_depth=6)
    test_df = normalize_nesting_depths(test_df, False, max_depth=6)

    # Merge tokens for each schema in the training and testing DataFrames
    train_df = merge_schema_tokens(train_df, tokenizer)
    test_df = merge_schema_tokens(test_df, tokenizer)

    # Create datasets with dynamic padding and batching
    train_dataset = CustomDataset(train_df)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer))
   
    # Set up scheduler to adjust the learning rate during training
    num_training_steps = num_epochs * len(train_dataloader) // accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    early_stopper = EarlyStopper(patience=1, min_delta=0.005) # training steps if the validation loss does not decrease by at least 0.005 for 1 consecutive epochs

    # Train the model
    model.train()
    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    pbar = tqdm(range(num_epochs), position=0, desc="Epoch")
    for epoch in pbar:
        total_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader, position=1, total=len(train_dataloader), leave=False, desc="Training")):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids, attention_mask, labels, extra_features = batch["input_ids"], batch["attention_mask"], batch["label"], batch["extra_features"]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask, extra_features=extra_features)
  
            # Calculate the training loss
            training_loss = loss_fn(logits, labels)
       
            # Need to average the loss if we are using DataParallel
            if training_loss.dim() > 0:
                training_loss = training_loss.mean()

            training_loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()
                lr_scheduler.step()
                
                # Calculate global training step
                step = (epoch * len(train_dataloader) + i + 1) // accumulation_steps
        
            total_loss += training_loss.item()

        average_loss = total_loss / len(train_dataloader)

        # Test the model
        testing_loss = test_model(test_df, tokenizer, model, device, wandb)
        early_stop = early_stopper.early_stop(testing_loss)
        if early_stop:
            print("Early stopping")
            break

        wandb.log({
            "training_loss": average_loss,
            "learning_rate": lr_scheduler.get_last_lr()[0], 
            "step": step,
            "epoch": epoch + 1
        })

    # Save the model and adapter after training
    save_model_and_adapter(model)
    # Finish wandb logging
    wandb.finish()
    

def test_model(test_df, tokenizer, model, device, wandb):
    """
    Tests the model on a given dataset and logs the evaluation metrics.

    Args:
        test_df (pd.DataFrame): DataFrame with testing data.
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing schemas.
        model (CustomBERTModel): The model to evaluate.
        device (torch.device): The device to run the model on.
        wandb: The wandb object for logging.

    Returns:
        float: The average testing loss.
    """

    # Create datasets with dynamic padding and batching
    test_dataset = CustomDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_fn(batch, tokenizer))

    model.eval()
    total_loss = 0.0

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    total_actual_labels = []
    total_predicted_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), leave=False, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids, attention_mask, labels, extra_features = batch["input_ids"], batch["attention_mask"], batch["label"], batch["extra_features"]

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask, extra_features=extra_features)   
            
            # Calculate the testing loss
            testing_loss = loss_fn(logits, labels)

            # Need to average the loss if we are using DataParallel
            if testing_loss.dim() > 0:
                testing_loss = testing_loss.mean()
            total_loss += testing_loss.item()

            # Get the actual and predicted labels
            actual_labels = labels.cpu().numpy()
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
            total_actual_labels.extend(actual_labels)
            total_predicted_labels.extend(predicted_labels)

    average_loss = total_loss / len(test_loader)

    # Calculate the accuracy, precision, recall, f1 score of the positive class
    accuracy = accuracy_score(total_actual_labels, total_predicted_labels)
    precision = precision_score(total_actual_labels, total_predicted_labels, average="binary")
    recall = recall_score(total_actual_labels, total_predicted_labels, average="binary")
    f1 = f1_score(total_actual_labels, total_predicted_labels, average="binary")

    #print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, F1: {f1}')
    wandb.log({"accuracy": accuracy, "testing_loss": average_loss, "precision": precision, "recall": recall, "F1": f1})

    return average_loss





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


def has_valid_prefix(path, schema_paths):
    """
    Check if the prefix of a tuple path exists in the schema paths.

    Args:
        path (tuple): The tuple path to check.
        schema_paths (set): Set of schema paths.

    Returns:
        bool: True if the prefix exists in the schema paths, False otherwise.
    """
    prefix = path[:-1]
    return prefix in schema_paths


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
            if schema == "package-json.json":
                schema = json.load(schema_file)
            else:
                schema = jsonref.load(schema_file)
                schema = jsonref.JsonRef.replace_refs(schema) # Added this line
    except Exception as e:
        print(f"Error loading and dereferencing schema {schema_path}: {e}")
        return filtered_df
    
    # Get all paths in the schema
    schema_paths = set(get_schema_paths(schema))

    # Only keep rows where the 'path' exists in schema_paths
    filtered_df = filtered_df[
        filtered_df["path"].apply(lambda path: in_schema(path, schema_paths))
    ]
    '''
    # Filter paths based on valid prefixes
    filtered_df = filtered_df[
        filtered_df["path"].apply(lambda path: has_valid_prefix(path, schema_paths))
    ]
    '''

    # Reset the index of the filtered DataFrame
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df






def load_model_and_adapter(save_path="frequent_pattern_model"):
    """
    Load the custom model and its adapter.

    Args:
        save_path (str): Path where the custom model and adapter are saved.

    Returns:
        CustomBERTModel: The loaded custom model.
        AutoTokenizer: The tokenizer used for processing schemas.
    """
    # Initialize a new instance of the custom model and tokenizer
    custom_model = CustomBERTModel()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load the base CodeBERT model and adapter
    custom_model.codebert = AutoAdapterModel.from_pretrained(save_path)
    custom_model.codebert.load_adapter(save_path, load_as=ADAPTER_NAME)
    custom_model.codebert.set_active_adapters(ADAPTER_NAME)
    custom_model.load_state_dict(torch.load(os.path.join(save_path, "frequent_patterns_model_state_dict.pth")))

    return custom_model, tokenizer


def create_schema_pairs_with_common_properties(df, max_depth=6):
    """
    Optimized version to create a DataFrame with pairs of schemas that have at least two common properties,
    while normalizing nesting depth.

    Args:
        df (pd.DataFrame): Input DataFrame containing columns 'path', 'schema', 'filename'.
        max_depth (int): Maximum depth for normalization.

    Returns:
        pd.DataFrame: A new DataFrame with pairs of schemas that share at least two properties.
    """
    # Precompute path-to-row lookup to avoid slow df[df["path"] == path1] operations
    path_to_row = {row["path"]: row for _, row in df.iterrows()}

    # Extract schema properties once
    path_to_properties = {}
    for path, row in path_to_row.items():
        try:
            schema = json.loads(row["schema"])
            properties = frozenset(extract_properties(schema).keys())
        except (json.JSONDecodeError, TypeError):
            properties = frozenset()
        path_to_properties[path] = properties

    # Reverse index: property -> set of paths containing that property
    property_to_paths = defaultdict(set)
    for path, properties in path_to_properties.items():
        for prop in properties:
            property_to_paths[prop].add(path)

    # Find candidate pairs with at least 2 shared properties
    candidate_pairs = set()
    for paths in property_to_paths.values():
        if len(paths) > 1:  
            for path1, path2 in combinations(paths, 2):
                common_properties = path_to_properties[path1] & path_to_properties[path2]
                if len(common_properties) >= 1:
                    candidate_pairs.add((path1, path2))

    # Precompute nesting depth normalization
    path_to_depth = {
        path: min(len(path), max_depth) / max_depth
        for path in path_to_row
    }

    # Collect valid pairs
    rows = []
    for path1, path2 in candidate_pairs:
        row1, row2 = path_to_row[path1], path_to_row[path2]

        rows.append({
            "filename": row1["filename"],
            "path1": path1,
            "path2": path2,
            "path1_freq": row1["path_frequency"],
            "path2_freq": row2["path_frequency"],
            "nested_key_freq1": row1["nested_key_frequency"],
            "nested_key_freq2": row2["nested_key_frequency"],
            "nesting_depth1": path_to_depth[path1],
            "nesting_depth2": path_to_depth[path2],
            "schema1": row1["schema"],
            "schema2": row2["schema"],
        })

    return pd.DataFrame(rows)


def build_definition_graph(eval_loader, custom_model, device, schema_name):
    """
    Build a definition graph from tokenized schema pairs using the given model, with batching.

    Args:
        eval_loader (DataLoader): DataLoader containing the tokenized schema pairs.
        custom_model (PreTrainedModel): The model used for predicting connections.
        device (torch.device): The device (CPU or GPU) to run the model on.
        schema_name (str): The name of the schema.

    Returns:
        nx.Graph: A graph with edges representing predicted connections between pairs.
    """
    graph = nx.Graph()
    edges = []
    for batch in tqdm(eval_loader, total=len(eval_loader), leave=True, desc="Processing pairs for " + schema_name):
        
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        input_ids, attention_mask, extra_features = batch["input_ids"], batch["attention_mask"], batch["extra_features"]

        # Model prediction
        with torch.no_grad():
            logits = custom_model(input_ids=input_ids, attention_mask=attention_mask, extra_features=extra_features)
            preds = torch.argmax(logits, dim=-1).tolist()
            #print(f"Predictions: {preds}")
        # Append edges based on predictions
        for idx, pred in enumerate(preds):
            if pred == 1:
                path1 = batch["path1"][idx]
                path2 = batch["path2"][idx]
                print(f"Adding edge between {path1} and {path2}")
                edges.append((path1, path2))
        
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
    FN: Definition from the schema that does not exist in the file.SS
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


def evaluate_single_schema(schema, test_ground_truth):
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
    
    df, frequent_ref_defn_paths, schema_name, failure_flags = process_data.process_schema(schema, "")
    if df is not None and frequent_ref_defn_paths is not None:
        
        # Remove paths whose prefixes are not explicitly defined in the schema
        #df = remove_additional_properties(df, schema_name)
        #if len(df) < 2:
        #    print(f"Schema {schema_name} has less than 2 paths after filtering.", flush=True)
        #    return None
        
           
        # Load the model and tokenizer
        custom_model, tokenizer = load_model_and_adapter()

        # Enable multiple GPU usage with DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for model parallelism.", flush=True)
            custom_model = nn.DataParallel(custom_model)

        # Create a DataFrame with pairs of schemas that have at least two common properties
        pairs_df = create_schema_pairs_with_common_properties(df)
        
        # Process and normalize nesting depths for evaluation
        #normalized_df = normalize_nesting_depths(pairs_df, True, max_depth=6)
        
        # Tokenize and merge schemas
        eval_df = merge_schema_tokens(pairs_df, tokenizer)
        
        # Initialize the original model for calculating cosine similarity
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original_model = AutoAdapterModel.from_pretrained(MODEL_NAME)
        original_model.to(device)
        
        # Calculate the cosine similarity between the paths
        eval_df = process_data.calculate_cosine_similarity(eval_df, original_model, tokenizer, device)
        
        # Create datasets with dynamic padding and batching
        eval_dataset = CustomEvalDataset(eval_df)
        eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_eval_fn(batch, tokenizer))
        
        custom_model.to(device)
        custom_model.eval()
    
        graph = build_definition_graph(eval_loader, custom_model, device, schema_name)
    
        # Predict clusters and sort each predicted cluster alphabetically
        predicted_clusters = [sorted(cluster) for cluster in find_definitions_from_graph(graph)]

        # Get the ground truth clusters and sort each actual cluster alphabetically
        defn_paths_dict = test_ground_truth.get(schema_name, {})
        actual_clusters = [sorted([tuple(path) for path in cluster]) for cluster in sorted(defn_paths_dict.values())]

        # Calculate the precision, recall, and F1-score
        precision, recall, f1_score, _, _ = calc_scores(actual_clusters, predicted_clusters)

        print(f"Actual clusters: {schema_name}", flush=True)
        for cluster in actual_clusters:
            print(cluster, flush=True)
        
        print(f"Predicted clusters: {schema_name}", flush=True)
        for cluster in predicted_clusters:
            print(cluster, flush=True)

        print(f"Schema: {schema_name}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")
        print(flush=True)
        # Return evaluation metrics and sorted clusters for the schema
        return schema, precision, recall, f1_score, actual_clusters, predicted_clusters
        
    return None


def evaluate_data(test_ground_truth, output_file="evaluation_results.json"):
    """
    Evaluate the model on the entire test data sequentially and store results in a JSON file.

    Args:
        test_ground_truth (dict): Dictionary containing ground truth information for test data.
        output_file (str): Path to the JSON file to store the evaluation results.
    """
    # Get the test schemas
    test_schemas = list(test_ground_truth.keys())
    recalls = []
    f1_scores = []
    results = []

    # Process schemas sequentially
    for schema in tqdm(test_schemas, total=len(test_schemas), position=1):
        result = evaluate_single_schema(schema, test_ground_truth)
        if result is not None:
            schema, precision, recall, f1_score, actual_clusters, predicted_clusters = result
            f1_scores.append(f1_score)
            recalls.append(recall)

            # Collect results for each schema
            results.append({
                "schema": schema,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "actual_clusters": actual_clusters,
                "predicted_clusters": predicted_clusters
            })

    # Calculate average F1-score and average recall, then add them to results
    if f1_scores:
        avg_f1_score = sum(f1_scores) / len(f1_scores)
        avg_recall = sum(recalls) / len(recalls)
        results.append({"average_f1_score": avg_f1_score, "average_recall": avg_recall})

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
    precision_scores = []  # To store precision scores for each schema
    recall_scores = []  # To store recall scores for each schema

    # Iterate through each filename group
    for schema_name, group in tqdm(df.groupby("filename"), position=1, total=len(df["filename"].unique()), desc="Grouping paths"):
        # Convert string representation of paths to tuples
        group["path"] = group["path"].apply(ast.literal_eval)

        # Remove paths not explicitly defined in the schema
        filtered_df = remove_additional_properties(group, schema_name)
        print(f"Number of paths after filtering: {len(filtered_df)} for schema {schema_name}", flush=True)

        # Skip schemas with insufficient paths after filtering
        if filtered_df.empty or len(filtered_df) < 2:
            print(f"Schema {schema_name} has less than 2 paths after filtering. Skipping.", flush=True)
            continue

        paths = filtered_df["path"].tolist()
        distinct_keys = filtered_df["distinct_nested_keys"].tolist()

        # Dictionary to track paths based on shared keys
        group_dict = defaultdict(list)

        # Process each path and its associated keys
        for path, keys in zip(paths, distinct_keys):
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
        defn_paths_dict = test_ground_truth.get(schema_name, {})
        actual_clusters = [sorted([tuple(path) for path in cluster]) for _, cluster in defn_paths_dict.items()]

        print(f"Actual clusters: {actual_clusters}", flush=True)
        print(f"Predicted clusters: {predicted_clusters}", flush=True)

        # Evaluate the predicted clusters against the ground truth
        precision, recall, f1_score, matched_definitions, matched_paths = calc_scores(actual_clusters, predicted_clusters)
        print(f"Schema: {schema_name}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}", flush=True)
        print(flush=True)

        # Store individual F1 scores for average calculation later
        f1_scores.append(f1_score)
        precision_scores.append(precision)
        recall_scores.append(recall)

        # Collect results for the current schema
        results = {
            "schema": schema_name,
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
        avg_precision = sum(precision_scores) / len(precision_scores)
        avg_recall = sum(recall_scores) / len(recall_scores)
        all_results.append({
            "average_f1_score": avg_f1_score,
            "average_precision": avg_precision,
            "average_recall": avg_recall
        })

    # Write all results to the specified JSON file
    with open(output_file, "w") as json_file:
        json.dump(all_results, json_file, indent=4)


def get_json_schema_size(original):
    """
    Calculate the size of JSON schemas in kilobytes, excluding whitespace.

    Returns:
        None: Prints the schema names and their sizes in kilobytes.
    """
    # Load the ground truth file
    with open("test_ground_truth.json", 'r') as f:
        for line in f:
            json_data = json.loads(line)
            for schema_name in json_data.keys():
                schema_path = os.path.join(SCHEMA_FOLDER, schema_name)
                try:

                    with open(schema_path, 'r') as schema_file:
                        if original == "yes":
                            schema = json.load(schema_file)
                        else:
                            schema = jsonref.load(schema_file)
                            schema = jsonref.JsonRef.replace_refs(schema)
                            
                        compact_schema = json.dumps(schema, separators=(',', ':'))
                        size_in_kb = round(len(compact_schema.encode('utf-8')) / 1024, 2)
                        print(f"Schema: {schema_name}, Size: {size_in_kb} KB", flush=True)
                except TypeError as e:
                    print(f"Error serializing schema {schema_path}: {e}", flush=True)
                    continue
                except Exception as e:
                    print(f"Error loading schema {schema_path}: {e}", flush=True)
                    continue



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
        