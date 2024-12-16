import ast
import concurrent.futures
import json
import jsonref
import math
import networkx as nx
import os
import subprocess
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from adapters import AdapterTrainer, AutoAdapterModel
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
        path1 = self.data.iloc[idx]["path1"]
        path2 = self.data.iloc[idx]["path2"]
        nesting_depth1 = len(path1)
        nesting_depth2 = len(path2)

        return {
            "input_ids": tokenized_schema,
            "attention_mask": [1] * len(tokenized_schema),
            "label": label,
            "nesting_depth1": nesting_depth1,
            "nesting_depth2": nesting_depth2,
        }


def collate_fn(batch, tokenizer):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    nesting_depth1 = torch.tensor([item["nesting_depth1"] for item in batch])
    nesting_depth2 = torch.tensor([item["nesting_depth2"] for item in batch])

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "label": labels,
        "nesting_depth1": nesting_depth1,
        "nesting_depth2": nesting_depth2
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
        self.nesting_depth1 = nn.Linear(1, HIDDEN_SIZE) 
        self.nesting_depth2 = nn.Linear(1, HIDDEN_SIZE)

        # Final classifier to process concatenated logits and custom features
        self.modified_classifier = nn.Linear(HIDDEN_SIZE * 2 + 2, 2)


    def forward(self, input_ids, attention_mask, nesting_depth1, nesting_depth2):
        # Pass input through the pre-trained CodeBERT with the adapter
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the logits from the classification head
        logits = outputs.logits

        # Process nesting depths with custom layers
        depth1_emb = F.relu(self.nesting_depth1(nesting_depth1.unsqueeze(-1).float()))
        depth2_emb = F.relu(self.nesting_depth2(nesting_depth2.unsqueeze(-1).float()))
        
        # Concatenate the [CLS] output with the embeddings of depth1 and depth2
        combined_output = torch.cat([logits, depth1_emb, depth2_emb], dim=-1)
        
        # Pass the combined output through the final classification layer
        logits = self.modified_classifier(combined_output)
    
        return logits
            

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
    learning_rate = 1e-6
    num_epochs = 75

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

    # Train the model
    model.train()
     # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    pbar = tqdm(range(num_epochs), position=0, desc="Epoch")
    for epoch in pbar:
        total_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader, position=1, total=len(train_dataloader), desc="Training")):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids, attention_mask, labels, nesting_depth1, nesting_depth2 = batch["input_ids"], batch["attention_mask"], batch["label"], batch["nesting_depth1"], batch["nesting_depth2"]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask, nesting_depth1=nesting_depth1, nesting_depth2=nesting_depth2)
  
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
    test_dataset = CustomDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: collate_fn(batch, tokenizer))

    model.eval()
    total_loss = 0.0

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    total_actual_labels = []
    total_predicted_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids, attention_mask, labels, nesting_depth1, nesting_depth2 = batch["input_ids"], batch["attention_mask"], batch["label"], batch["nesting_depth1"], batch["nesting_depth2"]

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask, nesting_depth1=nesting_depth1, nesting_depth2=nesting_depth2)
            
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
    model = CustomBERTModel()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load the base CodeBERT model and adapter
    model.codebert = AutoAdapterModel.from_pretrained(save_path)
    model.codebert.load_adapter(save_path, load_as=ADAPTER_NAME)
    model.codebert.set_active_adapters(ADAPTER_NAME)

    # Load the state_dict for the custom layers
    #model.load_state_dict(torch.load(os.path.join(save_path, "frequent_patterns_model_state_dict.pth")))

    return model, tokenizer



def find_pairs_with_common_properties(df):
    """
    Generate pairs of paths from a DataFrame where the schemas have at least two properties in common.

    Args:
        df (pd.DataFrame): A DataFrame containing path information.

    Yields:
        tuple: A tuple containing the indices of the two paths (i, j) and a set of common properties between their schemas.
    """

    # Extract properties for each schema into a list
    properties_list = [
        set(json.loads(row["schema"]).get("properties", {}).keys())
        if isinstance(row["schema"], str) else set()
        for _, row in df.iterrows()
    ]

    # Compare pairs of schemas
    for i, properties_i in enumerate(properties_list):
        for j in range(i + 1, len(properties_list)):
            common_properties = properties_i & properties_list[j]
            if len(common_properties) >= 2:
                yield i, j, sorted(common_properties)


def merge_eval_schema_tokens(batch, df, tokenizer):
    """
    Merge the tokens of two schemas for evaluation, with truncation if necessary to fit the maximum token length.

    Args:
        batch (list): A list of tuples containing the indices of the two schemas and a set of common properties.
        df (pd.DataFrame): DataFrame containing the paths and schemas.
        tokenizer (PreTrainedTokenizer): Tokenizer used for processing schemas.

    Returns:
        list: A list of merged tokenized schemas, adhering to token length constraints.
    """
    
    # Special tokens
    bos_token_id = tokenizer.bos_token_id  # [BOS]
    sep_token_id = tokenizer.sep_token_id  # [SEP]
    eos_token_id = tokenizer.eos_token_id  # [EOS]

    tokenized_schemas = []

    # Loop through the entire batch passed in as 'pairs'
    for i1, i2, common_properties in batch:
        schema1 = json.loads(df["schema"].iloc[i1])
        schema2 = json.loads(df["schema"].iloc[i2])

        # Order properties by commonality
        ordered_schema1, ordered_schema2 = order_properties_by_commonality(schema1, schema2)
        
        # Tokenize schemas
        tokenized_schema1 = tokenize_schema(ordered_schema1, tokenizer)
        tokenized_schema2 = tokenize_schema(ordered_schema2, tokenizer)
        
        # Calculate the total length of the merged tokenized schemas
        total_len = len(tokenized_schema1) + len(tokenized_schema2)
        max_tokenized_len = MAX_TOK_LEN - 3  # Account for BOS, EOS, and newline token lengths

        # Truncate the schemas proportionally if they exceed the max token length
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
        
    return tokenized_schemas


def process_pairs(batch, df, model, device, tokenizer):
    """
    Process a batch of schema pairs and return edges for connected pairs.

    Args:
        batch (list): A list of tuples containing the indices of the two schemas and a set of common properties.
        df (pd.DataFrame): DataFrame containing the paths and schemas.
        model (PreTrainedModel): The model used for predicting connections.
        device (torch.device): The device to run the model on.
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing schemas.

    Returns:
        list[tuple]: A list of tuples where each tuple contains the paths of two schemas that are predicted to be connected.
    """
    batch_edges = []
    batch_input_ids = []
    batch_attention_masks = []

    # Tokenize and merge schemas
    tokenized_schemas = merge_eval_schema_tokens(batch, df, tokenizer)
    
    # Loop through each tokenized schema in the batch
    for tokenized_schema in tokenized_schemas:
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
            i1, i2, _ = batch[idx]
            batch_edges.append((df["path"].iloc[i1], df["path"].iloc[i2]))
    
    return batch_edges


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

    # Process schemas in parallel using 
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







def group_paths(df, test_ground_truth, min_common_keys=1, output_file="evaluation_results_baseline_model.json"):
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
        

