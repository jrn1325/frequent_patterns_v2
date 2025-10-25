import ast
import itertools
import json
import jsonref
import math
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from accelerate import Accelerator
from adapters import AutoAdapterModel
from collections import OrderedDict
from copy import deepcopy
from itertools import combinations
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, accuracy_score
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler, AutoModelForSequenceClassification



import process_data
MODEL_NAME = "microsoft/codebert-base"
ADAPTER_PATH = "./adapter-model/adapter"
FULL_PATH = "./adapter-model/full"
ADAPTER_NAME = "frequent-patterns"
BATCH_SIZE = 64
MAX_TOK_LEN = 512
ACCUMULATION_STEPS = 2
HIDDEN_SIZE = 768
LEARNING_RATE = 2e-5
NUM_EPOCHS = 25
SEED = 42

SCHEMA_FOLDER = "converted_processed_schemas"
JSON_FOLDER = "processed_jsons"
JSON_SUBSCHEMA_KEYWORDS = {"allOf", "oneOf", "anyOf", "not"}
JSON_SCHEMA_KEYWORDS = {"properties", "patternProperties", "additionalProperties", "items", "prefixItems", "allOf", "oneOf", "anyOf", "not", "if", "then", "else", "$ref"}



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
    #df = df.drop(["schema1", "schema2", "filename"], axis=1)

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
    Dataset for JSON schemas (no chunking).
    Each example already has tokenized input_ids and attention_mask.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            "input_ids": torch.tensor(row["tokenized_schema"], dtype=torch.long),
            "attention_mask": torch.ones(len(row["tokenized_schema"]), dtype=torch.long),
            "label": torch.tensor(row["label"], dtype=torch.float)
        }

def collate_fn(batch, tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": labels
    }

class CustomEvalDataset(Dataset):
    """
    Custom PyTorch Dataset class for Evaluating the model.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the tokenized schemas.

    Returns:
        dict: A dictionary containing the input IDs, attention mask, paths, and filenames for each example.
    """

    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        tokenized_schema = row["tokenized_schema"]
        path1 = row["path1"]
        path2 = row["path2"]
        filename = row["filename"]
        return {"input_ids": tokenized_schema, "attention_mask": [1] * len(tokenized_schema), "path1": path1, "path2": path2, "filename": filename}

def collate_eval_fn(batch, tokenizer):
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    path1 = [item["path1"] for item in batch]
    path2 = [item["path2"] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "path1": path1,
        "path2": path2
    }

def initialize_model(training_mode="adapter"):
    """
    Initialize the model and tokenizer in either 'adapter' (LoRA) or 'full' mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

    if training_mode == "adapter":
        # Wrap with LoRA for adapter-style training
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )
        model = get_peft_model(model, peft_config)
        print(f"Initialized {MODEL_NAME} in ADAPTER (LoRA) mode.")
    else:
        print(f"Initialized {MODEL_NAME} in FULL fine-tuning mode.")

    return model, tokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensures reproducibility, but may slow down
    torch.backends.cudnn.benchmark = False     # Disable benchmark mode for reproducibility

def train_model(train_df, test_df, training_mode="adapter"):
    wandb.init(
        project="custom-codebert_frequent_patterns",
        config={
            "accumulation_steps": ACCUMULATION_STEPS,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "model_name": MODEL_NAME,
            "training_mode": training_mode,
            "adapter_name": ADAPTER_NAME,
        }
    )

    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()
    set_seed(SEED)

    # Initialize model & tokenizer
    model, tokenizer = initialize_model(training_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Merge and tokenize schemas
    train_df = merge_schema_tokens(train_df, tokenizer)
    test_df = merge_schema_tokens(test_df, tokenizer)
    
    # Build datasets and dataloaders
    train_dataset = CustomDataset(train_df)
    test_dataset = CustomDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    # Optimizer + Scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_loader) // ACCUMULATION_STEPS
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )

    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # ------------------- TRAINING LOOP -------------------
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Training {epoch+1}/{NUM_EPOCHS}")

        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).float()

            with accelerator.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.view(-1)
                loss = loss_fn(logits, labels)

            accelerator.backward(loss)
            total_train_loss += loss.item()

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)

        # ------------------- EVALUATION -------------------
        model.eval()
        total_eval_loss = 0.0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {epoch+1}/{NUM_EPOCHS}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device).float()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.view(-1)
                loss = loss_fn(logits, labels)
                total_eval_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).long()
                all_preds.extend(accelerator.gather(preds).cpu().numpy())
                all_labels.extend(accelerator.gather(labels).cpu().numpy())

        avg_test_loss = total_eval_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        wandb.log({
            "training_loss": avg_train_loss,
            "testing_loss": avg_test_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch + 1
        })

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | "
              f"Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    save_model_and_adapter(unwrapped_model, training_mode)
    wandb.finish()

def save_model_and_adapter(model, training_mode="adapter"):
    if training_mode == "adapter":
        os.makedirs(ADAPTER_PATH, exist_ok=True)
        model.save_pretrained(ADAPTER_PATH)
        print(f"Saved adapter model to {ADAPTER_PATH}")
    else:
        os.makedirs(FULL_PATH, exist_ok=True)
        model.save_pretrained(FULL_PATH)
        print(f"Saved full fine-tuned model to {FULL_PATH}")

def load_model_and_adapter(eval_mode="adapter"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if eval_mode == "adapter":
        config = PeftConfig.from_pretrained(ADAPTER_PATH)
        base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print(f"Loaded LoRA adapter from {ADAPTER_PATH}")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(FULL_PATH, num_labels=1)
        print(f"Loaded full fine-tuned model from {FULL_PATH}")

    return model, tokenizer



def jaccard(set1, set2):
    set1, set2 = set(set1), set(set2)
    union = set1 | set2
    inter = set1 & set2
    return len(inter) / len(union) if union else 0

def create_schema_pairs_with_common_properties(df, jaccard_threshold=0.25, min_common_props=2):
    """
    Create candidate schema pairs with overlapping properties and normalized path depth.

    Args:
        df (pd.DataFrame): DataFrame with 'path', 'schema', 'filename', 'nested_keys'.
        jaccard_threshold (float): Minimum Jaccard similarity between properties.
        min_common_props (int): Minimum shared properties.
        min_total_props (int): Skip schemas with fewer than this number of properties.

    Returns:
        pd.DataFrame: Schema pair candidates.
    """
    path_to_row = {row["path"]: row for _, row in df.iterrows()}

    path_to_properties = {}
    for path, row in path_to_row.items():
        try:
            schema = json.loads(row["schema"])
            props = extract_properties(schema)
            properties = frozenset(props.keys())
        except Exception:
            properties = frozenset()
        path_to_properties[path] = properties

    candidate_pairs = set()
    all_paths = list(path_to_properties.keys())

    for path1, path2 in combinations(all_paths, 2):
        props1 = path_to_properties[path1]
        props2 = path_to_properties[path2]

        common = props1 & props2
        if len(common) < min_common_props:
            continue

        if jaccard(props1, props2) >= jaccard_threshold:
            candidate_pairs.add((path1, path2))

    output_rows = []
    for path1, path2 in candidate_pairs:
        row1, row2 = path_to_row[path1], path_to_row[path2]
        output_rows.append({
            "filename": row1["filename"],
            "path1": path1, "path2": path2,
            "schema1": row1["schema"], "schema2": row2["schema"],
        })

    return pd.DataFrame(output_rows)

class DisjointSet:
    def __init__(self):
        self.parent = {}

    def find(self, u):
        if u not in self.parent:
            self.parent[u] = u
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        self.parent[self.find(u)] = self.find(v)

    def groups(self):
        clusters = {}
        for node in self.parent:
            root = self.find(node)
            clusters.setdefault(root, []).append(node)
        return [group for group in clusters.values() if len(group) > 1]

def generate_clusters(eval_loader, model, schema_name, device):
    """
    Generate clusters of schema connections based on model predictions.

    Args:
        eval_loader (DataLoader): DataLoader containing the tokenized schema pairs.
        model (PreTrainedModel): The model used for predicting connections.
        schema_name (str): The name of the schema.
        device (str): Device to run inference on ("cuda" or "cpu").

    Returns:
        list: A list of clusters, where each cluster is a list of paths that are connected.
    """
    model.eval()
    ds = DisjointSet()

    for batch in tqdm(eval_loader, total=len(eval_loader), leave=True, desc=f"Processing pairs for {schema_name}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().cpu().tolist()

        # Append edges based on predictions
        for idx, pred in enumerate(preds):
            if pred == 1:
                ds.union(batch["path1"][idx], batch["path2"][idx])

    return ds.groups()

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

def get_pairs(clusters):
    """
    Given a list of clusters, return a set of all pairs in the same cluster.

    Args:
        clusters (list): List of clusters, where each cluster is a list of paths.
    Returns:
        set: A set of tuples representing pairs of paths in the same cluster.
    """
    pairs = set()
    for cluster in clusters:
        for a, b in combinations(cluster, 2):
            pairs.add((a, b))
            pairs.add((b, a))
    return pairs

def calculate_metrics(actual_clusters, predicted_clusters):
    """
    Calculate precision, recall, and F1-score for the given actual and predicted clusters.
    Args:
        actual_clusters (list): List of actual clusters.
        predicted_clusters (list): List of predicted clusters.
    Returns:    
        tuple: Precision, recall, and F1-score.
    """
    actual_pairs = get_pairs(actual_clusters)
    predicted_pairs = get_pairs(predicted_clusters)

    tp = len(actual_pairs & predicted_pairs)# Correctly predicted paths
    fp = len(predicted_pairs - actual_pairs)# Extra paths in prediction
    fn = len(actual_pairs - predicted_pairs)# Missed paths in actual

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return precision, recall, f1

def evaluate_single_schema(schema, test_ground_truth, eval_mode):
    """
    Helper function to evaluate a single schema.

    Args:
        schema (str): The schema filename to be evaluated.
        test_ground_truth (dict): Ground truth clusters for test schemas.
        eval_mode (str): Evaluation mode, either "adapter" or "full".

    Returns:
        tuple: schema name, precision, recall, F1, actual clusters, predicted clusters
        or None if schema could not be processed.
    """

    df, frequent_ref_defn_paths, schema_name, _ = process_data.process_schema(schema, schema)
    if df is not None and frequent_ref_defn_paths:
        print(f"Schema {schema_name} has frequent reference definitions.", flush=True)

        # Load the model
        model, tokenizer = load_model_and_adapter(eval_mode)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Preprocess data
        pairs_df = create_schema_pairs_with_common_properties(df)
        if pairs_df.empty:
            print(f"No pairs found for schema {schema_name}.", flush=True)
            return None

        eval_df = merge_schema_tokens(pairs_df, tokenizer)
        eval_dataset = CustomEvalDataset(eval_df)
        eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_eval_fn(b, tokenizer))

        # Prepare model and dataloader for Accelerate
        accelerator = Accelerator(mixed_precision="fp16")
        accelerator.wait_for_everyone()
        model, eval_loader = accelerator.prepare(model, eval_loader)
        model.eval()

        # Predict clusters
        predicted_clusters = generate_clusters(eval_loader, model, schema_name, device)

        # Ground truth
        defn_paths_dict = test_ground_truth.get(schema_name, {})
        actual_clusters = [sorted([tuple(path) for path in cluster]) for cluster in sorted(defn_paths_dict.values())]

        # Metrics
        precision, recall, f1_score = calculate_metrics(actual_clusters, predicted_clusters)

        print(f"Actual clusters: {schema_name}", flush=True)
        for cluster in actual_clusters:
            print(cluster, flush=True)

        print(f"Predicted clusters: {schema_name}", flush=True)
        for cluster in predicted_clusters:
            print(cluster, flush=True)

        print(f"Schema: {schema_name}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}", flush=True)
        return schema, precision, recall, f1_score, actual_clusters, predicted_clusters
    else:
        print(f"Schema {schema_name} could not be processed or has no frequent reference definitions.", flush=True)

    return None

def evaluate_model(test_ground_truth, eval_mode, output_file="evaluation_results.json"):
    """
    Evaluate the model on the entire test data sequentially and store results in a JSON file.

    Args:
        test_ground_truth (dict): Dictionary containing ground truth information for test data.
        eval_mode (str): Evaluation mode, either "adapter" or "full".
        output_file (str): Path to the JSON file to store the evaluation results.
    """
    # Get the test schemas
    test_schemas = list(test_ground_truth.keys())
    recalls = []
    precision_scores = []
    f1_scores = []
    results = []

    #test_schemas = ["gitpod-configuration.json"]
    # Process schemas sequentially
    for schema in tqdm(test_schemas, total=len(test_schemas), position=1):
        result = evaluate_single_schema(schema, test_ground_truth, eval_mode)

        if result is not None:
            schema, precision, recall, f1_score, actual_clusters, predicted_clusters = result
            f1_scores.append(f1_score)
            recalls.append(recall)
            precision_scores.append(precision)

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
        avg_precision = sum(precision_scores) / len(precision_scores)
        print(f'Average Precision: {avg_precision}', flush=True)
        print(f'Average Recall: {avg_recall}', flush=True)
        print(f'Average F1-score: {avg_f1_score}', flush=True)
       
        results.append({"average_precision": avg_precision,  "average_recall": avg_recall, "average_f1_score": avg_f1_score})

    if eval_mode == "adapter":
        output_file = "adapter_" + output_file
    else:   
        output_file = "full_" + output_file

    # Write results to JSON file
    with open(output_file, "w") as json_file:
        json.dump(results, json_file, indent=4)
    

def parse_path(path_str):
    """Convert tuple string like "('$', 'a', 'b')" → ['a', 'b']
    Args:        
        path_str (str): String representation of a path tuple.
    
    Returns:
        list: List of path components, excluding the root '$'.
    """
    try:
        path_tuple = ast.literal_eval(path_str)
        return list(path_tuple[1:])  # remove '$'
    except Exception:
        return []

def insert_path(schema_dict, path, subschema):
    """
    Insert a subschema at a nested path into a global schema.
    
    Args:
        schema_dict (dict): The global schema dictionary to modify.
        path (list): The path where the subschema should be inserted.
        subschema (dict): The subschema to insert at the specified path.
    """
    current = schema_dict.setdefault("properties", {})
    for key in path[:-1]:
        current = current.setdefault(key, {"type": "object", "properties": {}})["properties"]
    current[path[-1]] = subschema

def merge_dicts(a, b):
    """
    Deep merge b into a.
    
    Args:
        a (dict): The target dictionary to merge into.
        b (dict): The source dictionary to merge from.
    """
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            merge_dicts(a[key], b[key])
        else:
            a[key] = deepcopy(b[key])

def compare_schema_sizes(deref_schemas_dir, original_schemas_dir, abstracted_schemas_dir):
    """
    Compare the sizes of original and abstracted JSON schemas in KB.
    
    Args:
        deref_schemas_dir (str): Directory containing dereferenced JSON schemas.
        original_schemas_dir (str): Directory containing original JSON schemas.
        abstracted_schemas_dir (str): Directory containing abstracted JSON schemas.
    """
    total_deref_kb = 0
    total_original_kb = 0
    total_abstracted_kb = 0
    reductions = []

    for filename in os.listdir(original_schemas_dir):
        with open(os.path.join(deref_schemas_dir, filename)) as f1, \
                open(os.path.join(original_schemas_dir, filename)) as f2, \
                open(os.path.join(abstracted_schemas_dir, filename)) as f3:
            try:
                deref_json = json.load(f1)
                original_json = json.load(f2)
                abstracted_json = json.load(f3)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {filename}: {e}", flush=True)
                continue
            
            try:
                d = len(json.dumps(deref_json, separators=(',', ':'), sort_keys=True).encode("utf-8")) / 1024
                o = len(json.dumps(original_json, separators=(',', ':'), sort_keys=True).encode("utf-8")) / 1024
                a = len(json.dumps(abstracted_json, separators=(',', ':'), sort_keys=True).encode("utf-8")) / 1024
            except TypeError as e:
                print(f"Error calculating size for {filename}: {e}", flush=True)
                continue

            total_deref_kb += d
            total_original_kb += o
            total_abstracted_kb += a
            reduction = (d - a) / (d - o) * 100 if (d - o) != 0 else 0
            reductions.append(reduction)
            print(f"File: {filename}, Dereferenced Size: {d:.2f} KB, Abstracted Size: {a:.2f} KB, Original Size: {o:.2f} KB,  Size Reduction: {reduction:.2f}%", flush=True)

    print(f"\nTotal Dereferenced Size: {total_deref_kb:.2f} KB")
    print(f"Total Abstracted Size: {total_abstracted_kb:.2f} KB")
    print(f"Total Original Size: {total_original_kb:.2f} KB")
    
    # Compute average reduction across all files
    if reductions:
        average_reduction = sum(reductions) / len(reductions)
    else:
        average_reduction = 0

    print(f"\nAverage Size Reduction Across Files: {average_reduction:.2f}%")







def group_paths(df, test_ground_truth, min_common_keys=2, output_file="evaluation_results_baseline_model.json"):
    """
    Greedily group paths that share a core set of nested keys. New paths can only join a group
    if they overlap with the original keys that caused the group to be formed.

    Args:
        df (pd.DataFrame): DataFrame with columns: "path", "distinct_nested_keys", "filename"
        test_ground_truth (dict): Ground truth clusters
        min_common_keys (int): Minimum shared keys for grouping
        output_file (str): Where to write evaluation results

    Returns:
        None
    """
    all_results = []
    f1_scores, precision_scores, recall_scores = [], [], []

    for schema_name, group in tqdm(df.groupby("filename"), desc="Grouping paths", position=1):
        group["path"] = group["path"].apply(ast.literal_eval)
        paths = group["path"].tolist()
        distinct_keys = group["distinct_nested_keys"].tolist()

        path_key_pairs = [
            (path, frozenset(ast.literal_eval(keys)))
            for path, keys in zip(paths, distinct_keys)
        ]
        path_key_pairs.sort(key=lambda x: -len(x[1]))  # largest keysets first

        # Each group: (required_shared_keys, all_merged_keys, list_of_paths)
        groups = []

        for path, keys in path_key_pairs:
            added = False
            for i, (required_keys, full_keys, path_list) in enumerate(groups):
                if len(keys & required_keys) >= min_common_keys:
                    updated_keys = full_keys | keys
                    groups[i] = (required_keys, updated_keys, path_list + [path])
                    added = True
                    break

            if not added:
                groups.append((keys, keys, [path]))

        # Filter groups with more than one path, sort paths within clusters
        predicted_clusters = [sorted([tuple(p) for p in path_list]) for _, _, path_list in groups if len(path_list) > 1]
        predicted_clusters = sorted(predicted_clusters)

        # Get the actual clusters for the current filename
        defn_paths_dict = test_ground_truth.get(schema_name, {})
        actual_clusters = [sorted([tuple(path) for path in cluster]) for _, cluster in defn_paths_dict.items()]

        print(f"Actual clusters: {actual_clusters}", flush=True)
        print(f"Predicted clusters: {predicted_clusters}", flush=True)

        # Evaluate the predicted clusters against the ground truth
        precision, recall, f1_score = calculate_metrics(actual_clusters, predicted_clusters)
        print(f"Schema: {schema_name}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}", flush=True)
        print(flush=True)

        # Store individual scores
        f1_scores.append(f1_score)
        precision_scores.append(precision)
        recall_scores.append(recall)

        results = {
            "schema": schema_name,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "actual_clusters": actual_clusters,
            "predicted_clusters": predicted_clusters
        }
        all_results.append(results)

    # Add macro average across schemas
    if f1_scores:
        avg_f1_score = sum(f1_scores) / len(f1_scores)
        avg_precision = sum(precision_scores) / len(precision_scores)
        avg_recall = sum(recall_scores) / len(recall_scores)
        all_results.append({
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1_score": avg_f1_score
        })

    # Save all results
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
                schema_path = os.path.join("deref", schema_name)
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



