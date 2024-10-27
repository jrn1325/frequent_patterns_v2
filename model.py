import concurrent.futures
import json
import jsonref
import math
import networkx as nx
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import tqdm
import wandb

from adapters import AdapterTrainer, AutoAdapterModel
from torch.optim import AdamW
from transformers import AutoTokenizer, get_scheduler, TrainingArguments, EvalPrediction

import process_data


MAX_TOK_LEN = 512
MODEL_NAME = "microsoft/codebert-base" 
PATH = "./adapter-model"
ADAPTER_NAME = "frequent_patterns"
SCHEMA_FOLDER = "processed_schemas"
JSON_FOLDER = "processed_jsons"
BATCH_SIZE = 8
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


def extract_properties(schema):
    """Extract properties from a JSON schema."""
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
    common_properties = set(properties1.keys()).intersection(set(properties2.keys()))

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
    learning_rate = 1e-6
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
        except (json.JSONDecodeError) as e:
            print(f"Skipping row {i} due to error: {e}")
            properties_list.append(set()) 
        except Exception as e:
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


def load_and_dereference_schema(schema_path):
    """
    Load the JSON schema from the specified path and recursively resolve $refs within it.

    Args:
        schema_path (str): The path to the JSON schema file.

    Returns:
        dict: The dereferenced JSON schema or None.
    """
    try:
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

            if "$ref" in schema:
                resolved_root = jsonref.JsonRef.replace_refs(schema)
                # Merge the resolved ref into the schema without discarding other properties
                partially_resolved_schema = {**resolved_root, **schema}
                # Remove $ref since it's now replaced with its resolution
                partially_resolved_schema.pop("$ref", None)
                resolved_schema = partially_resolved_schema
            else:
                resolved_schema = schema

            # Resolve any additional $refs that might exist after the first replacement
            resolved_schema = jsonref.JsonRef.replace_refs(resolved_schema)
            return resolved_schema

    except jsonref.JsonRefError as e:
        print(f"Error dereferencing schema {schema_path}: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing schema {schema_path}: {e}")
        return None
    except Exception as e:
        print(f"Unknown error dereferencing schema {schema_path}: {e}")
        return None
    

def extract_additional_properties_paths(schema, path=('$',)):
    """
    Recursively extract paths from a JSON schema where additionalProperties is an object or true.

    Args:
        schema (dict): The JSON schema.
        path (tuple): Current path being traversed (used for recursion).

    Yields:
        tuple: Each path found in the schema where additionalProperties is an object or true.
    """
    if not isinstance(schema, dict):
        return

    # Check for additionalProperties condition
    if "additionalProperties" in schema:
        if schema["additionalProperties"] is True or isinstance(schema["additionalProperties"], dict):
            yield path

    # Handle properties
    if "properties" in schema:
        for prop, subschema in schema["properties"].items():
            current_path = path + (prop,)
            yield from extract_additional_properties_paths(subschema, current_path)

    # Handle items for arrays
    if "items" in schema:
        items_schema = schema["items"]
        array_path = path + ('*',)
        yield from extract_additional_properties_paths(items_schema, array_path)
        
    # Handle subschemas
    for subschema_key in JSON_SUBSCHEMA_KEYWORDS:
        if subschema_key in schema:
            for subschema in schema[subschema_key]:
                yield from extract_additional_properties_paths(subschema, path)

    # Handle conditional schemas
    for condition_key in ["if", "then", "else"]:
        if condition_key in schema:
            yield from extract_additional_properties_paths(schema[condition_key], path)


def has_additional_properties_prefix_or_exact(path, additional_properties_paths):
    """
    Check if the given path matches exactly or starts with any prefix from additionalProperties_paths.

    Args:
        path (tuple): The path to check.
        additional_properties_paths (set): A set of paths representing additionalProperties.

    Returns:
        bool: True if the path matches exactly or starts with any additionalProperties prefix, False otherwise.
    """
    for additional_path in additional_properties_paths:
        # Check for exact match
        if path == additional_path:
            return True
        # Check for prefix match
        if path[:len(additional_path)] == additional_path:
            return True
    return False


def remove_additional_properties(filtered_df, schema):
    """
    Remove paths from the DataFrame that are not explicitly defined in the schema or have additionalProperties paths as prefixes or exact matches.

    Args:
        filtered_df (pd.DataFrame): DataFrame containing paths and associated data.
        schema (dict): The JSON schema to extract valid paths from.

    Returns:
        pd.DataFrame: Filtered DataFrame with paths that are explicitly defined in the schema.
    """
    # Load the schema
    schema_path = os.path.join(SCHEMA_FOLDER, schema)
    schema = load_and_dereference_schema(schema_path)
    #schema = process_data.load_schema(schema_path)

    # Extract paths where additionalProperties is an object or true
    additional_properties_paths = set(extract_additional_properties_paths(schema))
    print(f"Additional properties paths: {additional_properties_paths}")
    if not additional_properties_paths:
        return filtered_df

    # Filter out rows where the 'path' matches exactly or has an additionalProperties prefix
    filtered_df = filtered_df[
        ~filtered_df["path"].apply(lambda path: has_additional_properties_prefix_or_exact(path, additional_properties_paths))
    ]

    return filtered_df



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


def build_definition_graph(df, model, device, tokenizer):
    """
    Build a definition graph from tokenized schema pairs using the given model, without batching.

    Args:
        df (pd.DataFrame): A DataFrame containing paths and their tokenized schemas.
        model (PreTrainedModel): The model used for predicting connections.
        device (torch.device): The device (CPU or GPU) to run the model on.
        tokenizer (PreTrainedTokenizer): The tokenizer used for processing schemas.

    Returns:
        nx.Graph: A graph with edges representing predicted connections between pairs.
    """
    graph = nx.Graph()
    pairs = list(find_pairs_with_common_properties(df))
    
    with tqdm.tqdm(total=len(pairs), desc="Processing pairs", position=0, leave=True) as pbar:
        edges = process_pairs(pairs, df, model, device, tokenizer)

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
        tokenizer (PreTrainedTokenizer): The tokenizer for schema tokenization.
        test_ground_truth (dict): Ground truth clusters for test schemas.

    Returns:
        tuple: Contains schema name, precision, recall, F1 score, actual clusters, and predicted clusters.
        None: If the schema could not be processed.
    """
    
    # Load the model and tokenizer
    m, tokenizer = load_model_and_adapter()
    m.eval()
    m.to(device)

    filtered_df, frequent_ref_defn_paths, schema_name, failure_flags = process_data.process_schema(schema)
        
    if filtered_df is not None and frequent_ref_defn_paths is not None:
        filtered_df[["path", "schema"]].to_csv(f"./{schema_name}_filtered_df.csv", index=False)
        # Filter paths with schema
        #df = remove_additional_properties(filtered_df, schema)
        #df["path"].to_csv(f"./{schema_name}_filtered_df.csv", index=False)
        #print(f"Schema: {schema}, Number of paths: {len(df)}")
        
        # Build a definition graph
        graph = build_definition_graph(filtered_df, m, device, tokenizer)
        #print(f"Number of edges: {len(graph.edges)}")
        
        # Predict clusters
        predicted_clusters = find_definitions_from_graph(graph)
        #print(f"Number of predicted clusters: {len(predicted_clusters)}")

        # Get the ground truth clusters
        ground_truth_dict = test_ground_truth.get(schema_name, {})
        actual_clusters = [[tuple(inner_list) for inner_list in outer_list] for outer_list in ground_truth_dict.values()]

        # Calculate the precision, recall, and F1-score
        precision, recall, f1_score = calc_scores(actual_clusters, predicted_clusters)

        # Return evaluation metrics and clusters for the schema
        return schema, precision, recall, f1_score, actual_clusters, predicted_clusters
        
    return None


def evaluate_data(test_ground_truth):
    """
    Evaluate the model on the entire test data using multiple CPUs.

    Args:
        test_ground_truth (dict): Dictionary containing ground truth information for test data.
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the test schemas
    test_schemas = list(test_ground_truth.keys())
    f1_scores = []

    # Process schemas in parallel using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_single_schema, schema, device, test_ground_truth): schema for schema in test_schemas}

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=1):
            result = future.result()
            if result is not None:
                
                schema, precision, recall, f1_score, actual_clusters, predicted_clusters = result
                f1_scores.append(f1_score)

                # Print the evaluation metrics
                print(f"Schema: {schema}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")

                # Print the actual and predicted clusters
                print("Actual clusters")
                for actual_cluster in actual_clusters:
                    print(actual_cluster)

                print("\nPredicted clusters:")
                for predicted_cluster in predicted_clusters:
                    print(predicted_cluster)
                
                print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")
                
    # Print the average F1-score
    if f1_scores:
        print("Average F1-score:", sum(f1_scores) / len(f1_scores))
        
  
