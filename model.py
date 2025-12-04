import argparse
import ast
import json
import jsonref
import math
import numpy as np
import os
import pandas as pd
import shutil
import sys
import time
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb

from accelerate import Accelerator
from accelerate.utils import set_seed
from adapters import AutoAdapterModel, SeqBnConfig
from collections import OrderedDict
from copy import deepcopy
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

import warnings
warnings.filterwarnings("ignore")

import process_data

MODEL_NAME = "microsoft/codebert-base"
ADAPTER_PATH = "./adapter-model/adapter"
FULL_PATH = "./adapter-model/full"
ADAPTER_NAME = "frequent_patterns"
BATCH_SIZE = 64
MAX_TOK_LEN = 512
ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS = 25
SEED = 101
HIDDEN_SIZE = 768
NUM_NUMERIC_FEATURES = 5
NUM_LABELS = 2
SCHEMA_FOLDER = "converted_processed_schemas"
JSON_FOLDER = "processed_jsons"
JSON_SUBSCHEMA_KEYWORDS = {"allOf", "oneOf", "anyOf", "not"}
JSON_SCHEMA_KEYWORDS = {"properties", "patternProperties", "additionalProperties", "items", "prefixItems", "allOf", "oneOf", "anyOf", "not", "if", "then", "else", "$ref"}



# -------------------- Early Stopper --------------------
# https://stackoverflow.com/a/73704579
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.01):
        """
        patience: number of validations without improvement before stopping
        min_delta: minimum change in val loss to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def early_stop(self, val_loss):
        """
        Returns True if training should stop early.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        # No improvement → increase counter
        self.counter += 1

        # Patience exceeded
        return self.counter >= self.patience


# -------------------- Schema Property Ordering --------------------
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

def tokenize_schema(schema, tokenizer, device):
    """
    Tokenizes the input schema text with a maximum length constraint.

    Args:
        schema (str): The schema text to be tokenized.
        tokenizer (PreTrainedTokenizer): The tokenizer for processing the schema text.
        device (torch.device): The device to move the token tensors to.

    Returns:
        list: A list of token IDs representing the tokenized schema.
    """
    tokens = tokenizer(json.dumps(schema["properties"]), return_tensors="pt")["input_ids"].to(device)
    tokens = tokens.squeeze(0).tolist()
    tokens = tokens[1:-1]
    return tokens

def merge_schema_tokens(df, tokenizer, device):
    """
    Merge tokenized schemas for training pairs.

    Args:
        df (pandas.DataFrame): DataFrame containing schema pairs.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing schemas.
        device (torch.device): The device to move the token tensors to.

    Returns:
        pandas.DataFrame: DataFrame with merged tokenized schemas and average numeric features.
    """

    bos_token_id = tokenizer.bos_token_id
    sep_token_id = tokenizer.sep_token_id
    eos_token_id = tokenizer.eos_token_id

    merged_tokens = []
    avg_datatype_entropy_list = []
    avg_key_entropy_list = []
    avg_parent_frequency_list = []
    avg_num_nested_keys_list = []
    avg_semantic_similarity_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Merging schema tokens", leave=False):
        schema1 = json.loads(row["schema1"])
        schema2 = json.loads(row["schema2"])

        # Order properties
        ordered_schema1, ordered_schema2 = order_properties_by_commonality(schema1, schema2)

        # Tokenize (tokenize_schema MUST accept dict or JSON string)
        tokenized_schema1 = tokenize_schema(ordered_schema1, tokenizer, device)
        tokenized_schema2 = tokenize_schema(ordered_schema2, tokenizer, device)

        total_len = len(tokenized_schema1) + len(tokenized_schema2)
        max_len = MAX_TOK_LEN - 3  # [BOS] [SEP] [EOS]

        if total_len > max_len:
            truncate = total_len - max_len
            t1 = math.ceil(len(tokenized_schema1) / total_len * truncate)
            t2 = math.ceil(len(tokenized_schema2) / total_len * truncate)
            tokenized_schema1 = tokenized_schema1[:-t1]
            tokenized_schema2 = tokenized_schema2[:-t2]

        merged = (
            [bos_token_id] +
            tokenized_schema1 +
            [sep_token_id] +
            tokenized_schema2 +
            [eos_token_id]
        )
        merged_tokens.append(merged)

        # ---------- average numeric features ----------
        avg_datatype_entropy_list.append(
            (schema1.get("datatype_entropy", 0) + schema2.get("datatype_entropy", 0)) / 2
        )
        avg_key_entropy_list.append(
            (schema1.get("key_entropy", 0) + schema2.get("key_entropy", 0)) / 2
        )
        avg_parent_frequency_list.append(
            (schema1.get("parent_frequency", 0) + schema2.get("parent_frequency", 0)) / 2
        )
        avg_num_nested_keys_list.append(
            (schema1.get("num_nested_keys", 0) + schema2.get("num_nested_keys", 0)) / 2
        )
        avg_semantic_similarity_list.append(
            (schema1.get("semantic_similarity", 0) + schema2.get("semantic_similarity", 0)) / 2
        )

    df["tokenized_schema"] = merged_tokens
    df["avg_datatype_entropy"] = avg_datatype_entropy_list
    df["avg_key_entropy"] = avg_key_entropy_list
    df["avg_parent_frequency"] = avg_parent_frequency_list
    df["avg_num_nested_keys"] = avg_num_nested_keys_list
    df["avg_semantic_similarity"] = avg_semantic_similarity_list

    return df


# -------------------- Dataset --------------------
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_TOK_LEN):
        self.labels = torch.tensor(dataframe["label"].values, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.tokenized_schemas = dataframe["tokenized_schema"].tolist()

        # Numeric metadata
        self.avg_datatype_entropy = dataframe["avg_datatype_entropy"].values
        self.avg_key_entropy = dataframe["avg_key_entropy"].values
        self.avg_parent_frequency = dataframe["avg_parent_frequency"].values
        self.avg_num_nested_keys = dataframe["avg_num_nested_keys"].values
        self.avg_semantic_similarity = dataframe["avg_semantic_similarity"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        tokens = self.tokenized_schemas[idx]

        # Pad or truncate manually (since tokens are already ids)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

        attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in tokens]

        numeric_feats = torch.tensor([
            float(self.avg_datatype_entropy[idx]),
            float(self.avg_key_entropy[idx]),
            float(self.avg_parent_frequency[idx]),
            float(self.avg_num_nested_keys[idx]),
            float(self.avg_semantic_similarity[idx])
        ], dtype=torch.float)

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "numeric_feats": numeric_feats,
            "labels": self.labels[idx]
        }

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    numeric_feats = torch.stack([b["numeric_feats"].float() for b in batch], dim=0)
    labels = torch.tensor([b["labels"].item() for b in batch], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "numeric_feats": numeric_feats,
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

    def __init__(self, dataframe, tokenizer, max_length=MAX_TOK_LEN):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataframe = dataframe
        self.tokenized_schemas = dataframe["tokenized_schema"].tolist()

        # Numeric metadata
        self.avg_datatype_entropy = self.dataframe["avg_datatype_entropy"].values
        self.avg_key_entropy = self.dataframe["avg_key_entropy"].values
        self.avg_parent_frequency = self.dataframe["avg_parent_frequency"].values
        self.avg_num_nested_keys = self.dataframe["avg_num_nested_keys"].values
        self.avg_semantic_similarity = self.dataframe["avg_semantic_similarity"].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        tokens = self.tokenized_schemas[idx]

        # Pad or truncate manually (since tokens are already ids)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))

        attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in tokens]

        numeric_feats = torch.tensor([
            float(self.avg_datatype_entropy[idx]),
            float(self.avg_key_entropy[idx]),
            float(self.avg_parent_frequency[idx]),
            float(self.avg_num_nested_keys[idx]),
            float(self.avg_semantic_similarity[idx])
        ], dtype=torch.float)

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "numeric_feats": numeric_feats,
            "path1": self.dataframe["path1"].values[idx],
            "path2": self.dataframe["path2"].values[idx],
            "filename": self.dataframe["filename"].values[idx],
        }
    
def collate_eval_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    numeric_feats = torch.stack([b["numeric_feats"].float() for b in batch], dim=0)
    path1 = [b["path1"] for b in batch]
    path2 = [b["path2"] for b in batch]
    filename = [b["filename"] for b in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "numeric_feats": numeric_feats,
        "path1": path1,
        "path2": path2,
        "filename": filename
    }


# -------------------- Model Initialization & Training & Testing --------------------
class CustomCodeBERT(nn.Module):
    def __init__(self, 
                 model_name, 
                 num_numeric_features, 
                 num_labels, 
                 dropout=0.3, 
                 training_mode="adapter"):

        super().__init__()
        self.training_mode = training_mode
        self.num_numeric_features = num_numeric_features

        # Load base model
        if training_mode == "adapter":
            self.base_model = AutoAdapterModel.from_pretrained(model_name)
            self.base_model.config.output_hidden_states = True

            config = SeqBnConfig(
                mh_adapter=False,
                output_adapter=True,
                reduction_factor=16,
                non_linearity="relu",
                dropout=dropout
            )
            self.base_model.add_adapter(ADAPTER_NAME, config=config)
            self.base_model.add_classification_head(ADAPTER_NAME, num_labels=num_labels)
            self.base_model.set_active_adapters(ADAPTER_NAME)
            self.base_model.train_adapter(ADAPTER_NAME)

        else:
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                output_hidden_states=True 
            )
            self.base_model.config.output_hidden_states = True

            # Replace classifier
            if hasattr(self.base_model, "classifier") and isinstance(self.base_model.classifier, nn.Linear):
                in_features = self.base_model.classifier.in_features
                out_features = self.base_model.classifier.out_features
                self.base_model.classifier = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features, out_features)
                )

        # Hidden dimension
        hidden_size = self.base_model.config.hidden_size

        # Numeric projection
        self.numeric_proj = nn.Linear(num_numeric_features, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Final classifier after concatenation
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, numeric_feats, labels=None):
        # Encode token sequence (fp16 allowed)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.hidden_states[-1][:, 0, :]
        numeric_feats = numeric_feats.float()

        with torch.amp.autocast("cuda", enabled=False):
            numeric_emb = self.numeric_proj(numeric_feats)  

        numeric_emb = numeric_emb.to(cls_emb.dtype)

        # Concatenate safely
        combined = torch.cat([cls_emb, numeric_emb], dim=-1)
        combined = self.dropout(combined)

        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"logits": logits, "loss": loss, "cls_emb": cls_emb}

def initialize_model(num_labels, num_numeric_features, training_mode="adapter"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = CustomCodeBERT(
        model_name=MODEL_NAME,
        num_numeric_features=num_numeric_features,
        num_labels=num_labels,
        training_mode=training_mode,
    )

    print(f"Initialized {MODEL_NAME} with {num_labels} labels in {training_mode} mode")
    return model, tokenizer

def train_model(train_df, test_df, training_mode="adapter"):
    """
    Train the model on the training data and evaluate on the test data.

    Args:
        train_df (pd.DataFrame)
        test_df (pd.DataFrame)
        training_mode (str): "adapter" or "full"
    """

    # --- Initialize W&B ---
    wandb.init(
        project="custom-codebert_frequent_patterns",
        config={
            "accumulation_steps": ACCUMULATION_STEPS,
            "batch_size": BATCH_SIZE,
            "dataset": "json-schemas",
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "model_name": MODEL_NAME,
            "training_mode": training_mode,
            "adapter_name": ADAPTER_NAME,
        }
    )

    # --- Accelerator ---
    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()
    set_seed(SEED)

    # --- Model + Tokenizer ---
    model, tokenizer = initialize_model(num_labels=2, num_numeric_features=NUM_NUMERIC_FEATURES, training_mode=training_mode)
    
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Merge tokenized schemas ---
    train_df = merge_schema_tokens(train_df, tokenizer, device)
    test_df = merge_schema_tokens(test_df, tokenizer, device)
    
    # --- Dataset / DataLoader ---
    train_dataset = CustomDataset(train_df, tokenizer)
    test_dataset = CustomDataset(test_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=wandb.config.learning_rate)
    num_training_steps = wandb.config.epochs * len(train_loader) // wandb.config.accumulation_steps
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # --- Prepare with accelerator ---
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler
    )

    # --- Early stopper ---
    early_stopper = EarlyStopper(patience=2, min_delta=0.001)

    # --- Training loop ---
    for epoch in range(wandb.config.epochs):
        model.train()
        total_loss = 0.0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            outputs = model(**batch)
            loss = outputs["loss"] / wandb.config.accumulation_steps
            accelerator.backward(loss)
            total_loss += loss.item() * wandb.config.accumulation_steps

            if (i + 1) % wandb.config.accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)

        # --- Evaluation ---
        model.eval()
        all_labels, all_preds = [], []
        total_eval_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                outputs = model(**batch)
                logits = outputs["logits"]
                loss = outputs["loss"]

                total_eval_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(accelerator.gather(preds).cpu().numpy())
                all_labels.extend(accelerator.gather(batch["labels"]).cpu().numpy())

        avg_eval_loss = total_eval_loss / len(test_loader)

        # --- Overall metrics ---
        accuracy = accuracy_score(all_labels, all_preds)

        # Macro metrics:
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # --- W&B logging ---
        metrics = {
            "epoch": epoch + 1,
            "training_loss": avg_train_loss,
            "testing_loss": avg_eval_loss,
            "accuracy": accuracy,
            "learning_rate": scheduler.get_last_lr()[0],
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        wandb.log(metrics)
        print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}, "
              f"Testing Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")

        # --- Early stopping ---
        #if early_stopper.early_stop(avg_eval_loss):
        #    print("Early stopping triggered!")
        #    break

    # --- Save model/adapter ---
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    save_model_and_adapter(unwrapped_model, training_mode)
    wandb.finish()


# -------------------- Save & Load Model/Adapter --------------------
def save_model_and_adapter(model, training_mode="adapter"):
    if isinstance(model, nn.DataParallel):
        model = model.module

    if training_mode == "adapter":
        os.makedirs(ADAPTER_PATH, exist_ok=True)

        # 1. Save adapter
        model.base_model.save_adapter(ADAPTER_PATH, ADAPTER_NAME)

        # 2. Save pretrained backbone
        model.base_model.save_pretrained(ADAPTER_PATH)

        # 3. Save custom layers (numeric_proj + classifier + wrapper architecture)
        torch.save(model.state_dict(), os.path.join(ADAPTER_PATH, "custom_model_weights.pt"))

        print(f"Saved adapter + custom model weights to {ADAPTER_PATH}")

    else:
        os.makedirs(FULL_PATH, exist_ok=True)
        model.base_model.save_pretrained(FULL_PATH)
        torch.save(model.state_dict(), os.path.join(FULL_PATH, "custom_model_weights.pt"))
        print(f"Saved full fine-tuned model to {FULL_PATH}")

def load_model_and_adapter(training_mode="adapter"):
    # 1. Recreate architecture
    model = CustomCodeBERT(model_name=MODEL_NAME, num_numeric_features=NUM_NUMERIC_FEATURES, num_labels=NUM_LABELS, training_mode=training_mode)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if training_mode == "adapter":
        # 2. Load base model (CodeBERT backbone)
        model.base_model = AutoAdapterModel.from_pretrained(MODEL_NAME)
        model.base_model.config.output_hidden_states = True

        # 3. Load the saved adapter into the base model
        adapter_name = model.base_model.load_adapter(ADAPTER_PATH)
        model.base_model.set_active_adapters(adapter_name)

        # 4. Load your custom layers (numeric_proj + classifier)
        state = torch.load(os.path.join(ADAPTER_PATH, "custom_model_weights.pt"), map_location="cpu")
        model.load_state_dict(state, strict=False)

        print("Loaded CustomCodeBERT with adapters from", ADAPTER_PATH)

    else:
        # Full fine-tuned model
        model.base_model = AutoModelForSequenceClassification.from_pretrained(FULL_PATH)
        state = torch.load(os.path.join(FULL_PATH, "custom_model_weights.pt"), map_location="cpu")
        model.load_state_dict(state)
        print("Loaded full CustomCodeBERT model.")

    model.eval()
    return model, tokenizer


# -------------------- Schema Pair Creation --------------------
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

def generate_clusters(eval_loader, model, schema_name):
    """
    Generate clusters of schema connections based on model predictions.

    Args:
        eval_loader (DataLoader): DataLoader containing the tokenized schema pairs.
        model (PreTrainedModel): The model used for predicting connections.
        schema_name (str): The name of the schema.

    Returns:
        list: A list of clusters, where each cluster is a list of paths that are connected.
    """

    model.eval()
    ds = DisjointSet()

    for batch in tqdm(eval_loader, total=len(eval_loader), leave=True, desc=f"Processing pairs for {schema_name}"):

        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], numeric_feats=batch["numeric_feats"])
            logits = outputs["logits"] 
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs > 0.5).long().cpu().tolist()

        # Add positive edges
        for idx, pred in enumerate(preds):
            if pred == 1:
                ds.union(batch["path1"][idx], batch["path2"][idx])

    return ds.groups()

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


# -------------------- Evaluation --------------------
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

    df, frequent_ref_defn_paths, _, _ = process_data.process_schema(schema, SCHEMA_FOLDER)
    if df is None or len(frequent_ref_defn_paths) == 0:
        print(f"Skipping schema {schema} due to processing issues.", flush=True)
        return None

    # Load the model
    model, tokenizer = load_model_and_adapter(eval_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preprocess data
    pairs_df = create_schema_pairs_with_common_properties(df)
    if pairs_df.empty:
        print(f"No pairs found for schema {schema}.", flush=True)
        return None

    eval_df = merge_schema_tokens(pairs_df, tokenizer, device)
    eval_dataset = CustomEvalDataset(eval_df, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_eval_fn)

    # Prepare model and dataloader for Accelerate
    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()
    model, eval_loader = accelerator.prepare(model, eval_loader)
    model.eval()

    # Predict clusters
    predicted_clusters = generate_clusters(eval_loader, model, schema)
    print(f"Predicted clusters for {schema}", flush=True)

    # Ground truth
    defn_paths_dict = test_ground_truth.get(schema, {})
    actual_clusters = [sorted([tuple(path) for path in cluster]) for cluster in sorted(defn_paths_dict.values())]

    # Metrics
    precision, recall, f1_score = calculate_metrics(actual_clusters, predicted_clusters)

    print(f"Actual clusters: {schema}", flush=True)
    for cluster in actual_clusters:
        print(cluster, flush=True)

    print(f"Predicted clusters: {schema}", flush=True)
    for cluster in predicted_clusters:
        print(cluster, flush=True)

    print(f"Schema: {schema}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}", flush=True)
    return schema, precision, recall, f1_score, actual_clusters, predicted_clusters

def evaluate_model(test_ground_truth, eval_mode, output_file="evaluation_results.json"):
    """
    Evaluate the model on the entire test data and store results in a JSON file.

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
    


def main():
    """
    Main function to train or evaluate the model based on the provided mode from the command-line arguments. 
    If the mode is 'train', it calls the load_data function to train the model. If the mode is 'test', 
    it calls the evaluate_model function to evaluate the model. If the mode is unknown, it prints 
    an error message and exits.
    """

    start_time = time.time()

    try:
        argparse_parser = argparse.ArgumentParser(description="Train or evaluate the CustomCodeBERT model.")
        argparse_parser.add_argument("train_data", type=str, help="Path to the training data CSV file.")
        argparse_parser.add_argument("test_data", type=str, help="Path to the testing data CSV file.")
        argparse_parser.add_argument("mode", type=str, help="Mode: 'train' or 'eval'.")
        argparse_parser.add_argument("version", type=str, help="Extra argument for training mode: 'adapter' or 'full'.")
        args = argparse_parser.parse_args()

        train_data, test_data, mode, version = args.train_data, args.test_data, args.mode, args.version

        if mode == "train":
            train_df = pd.read_csv(train_data, delimiter=';')
            test_df = pd.read_csv(test_data, delimiter=';')
            train_model(train_df, test_df, training_mode=version)
            print(f'Training time: {time.time() - start_time}')

        elif mode == "eval":
            test_ground_truth = {}
            with open("test_ground_truth.json", 'r') as json_file:
                for line in json_file:
                    test_ground_truth.update(json.loads(line))
            evaluate_model(test_ground_truth, eval_mode=version)
            print(f'Evaluation time: {time.time() - start_time}')
        
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 
  

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()