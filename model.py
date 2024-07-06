import ast
import json
import jsonref
import math
import networkx as nx
import numpy as np
import os
import sys
import torch
import tqdm
import wandb

from adapters import AdapterTrainer, AutoAdapterModel
from torch.optim import AdamW
from transformers import AutoTokenizer, get_scheduler, TrainingArguments, EvalPrediction

import process_data


MAX_TOK_LEN = 512
ADAPTER_PATH = "./adapter_50"
ADAPTER_NAME = "mrpc"
SCHEMA_FOLDER = "schemas"
JSON_FOLDER = "jsons"

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


def initialize_model(model_name, adapter_name=ADAPTER_NAME):
    """
    Initializes a pre-trained model with an adapter for classification tasks.
    
    Args:
        model_name (str): The name of the pre-trained model to load.
        adapter_name (str): The name of the adapter to add to the model. Defaults to ADAPTER_NAME.
    
    Returns:
        tuple: A tuple containing the initialized model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)
    model.add_adapter(adapter_name, config="seq_bn")
    model.set_active_adapters(adapter_name)
    model.add_classification_head(adapter_name, num_labels=2)
    model.train_adapter(adapter_name)
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
    for idx, (tokenized_schema1, tokenized_schema2) in tqdm.tqdm(df[["Tokenized_schema1", "Tokenized_schema2"]].iterrows(), position=4, leave=False, total=len(df), desc="merge tokens"):
        if not isinstance(tokenized_schema1, list) or not isinstance(tokenized_schema2, list):
            # Convert string representations of lists to actual lists
            tokenized_schema1 = ast.literal_eval(tokenized_schema1)
            tokenized_schema2 = ast.literal_eval(tokenized_schema2)

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
    df["Tokenized_schema"] = tokenized_schemas
    df = df.drop(["Tokenized_schema1", "Tokenized_schema2", "Filename"], axis=1)

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

    max_length = max(len(schema) for schema in df["Tokenized_schema"])
    pad_token_id = tokenizer.pad_token_id

    dataset = []
    for idx in range(len(df)):
        schema = df["Tokenized_schema"].iloc[idx]
        label = df["Label"].iloc[idx]

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
     
    model_name = "microsoft/codebert-base" 
    accumulation_steps = 4
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 50

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

    # Merge tokenized schemas
    train_df_with_tokens = merge_schema_tokens(train_df, tokenizer) 
    test_df_with_tokens = merge_schema_tokens(test_df, tokenizer)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Transform data into dict
    train_data = transform_data(train_df_with_tokens, tokenizer, device)
    test_data = transform_data(test_df_with_tokens, tokenizer, device)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

     # Calculate total steps
    total_steps = len(train_data) // batch_size * num_epochs

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
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_accuracy,
        optimizers=(optimizer, lr_scheduler),
    )

    # Train the model
    trainer.train()
    trainer.evaluate()
        
    # Save the adapter
    save_adapter(model)
    

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
    model.save_adapter(ADAPTER_PATH, ADAPTER_NAME, with_head=True)

    # Log the adapter artifact
    artifact = wandb.Artifact("customized_codebert_frequent_patterns", type="model")
    artifact.add_dir(ADAPTER_PATH)
    wandb.log_artifact(artifact)
    wandb.save("./adapter_50/*")


def load_model_and_adapter():
    """
    Load the model and adapter from the specified path.

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
    m = AutoAdapterModel.from_pretrained(model_name)

    # Load the adapter
    m.load_adapter(ADAPTER_PATH, config=adapter_config["config"])

    # Activate the adapter
    m.set_active_adapters(adapter_name)

    return m


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


def build_definition_graph(df, model, device):
    """
    Build a definition graph from tokenized schema pairs using the given model.

    Args:
        df (pd.DataFrame): A DataFrame containing pairs and tokenized schemas.
        model (PreTrainedModel): The model used for predicting connections.
        device (torch.device): The device (CPU or GPU) to run the model on.

    Returns:
        nx.Graph: A graph with edges representing predicted connections between pairs.
    """
    # Create a graph
    graph = nx.Graph()
    
    # Loop over tokenized schemas
    for _, row in tqdm.tqdm(df.iterrows(), position=5, leave=False, total=len(df), desc="making graph"):
        pair = row["Pairs"]
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
    print("Cliques")
    for clique in cliques:
        print(clique)
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

    return processed_cliques


def evaluate_data(test_ground_truth, model, tokenizer):
    """
    Evaluate the model on the entire test data.

    Args:
        test_ground_truth (dict): Dictionary containing ground truth information for test data.
        model (PreTrainedModel): The trained model to be evaluated.
        tokenizer: The tokenizer used for processing schemas.
    """

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get the test schemas
    test_schemas = test_ground_truth.keys()

    for schema in tqdm.tqdm(test_schemas, position=1, leave=False, total=len(test_schemas)):
        # Create a dataframe for schema that meets all the conditions
        filtered_df, frequent_ref_defn_paths = process_data.process_schema(schema, JSON_FOLDER, SCHEMA_FOLDER)
        if filtered_df is not None and frequent_ref_defn_paths is not None:
            # Generate good and bad pairs
            labeled_df = process_data.generate_pairs(filtered_df, frequent_ref_defn_paths)

            # Merge the tokenized schemas
            df = merge_schema_tokens(labeled_df, tokenizer) 
        
            # Build a definition graph
            graph = build_definition_graph(df, model, device)
    
            # Predict clusters
            predicted_clusters = find_definitions_from_graph(graph)

            # Get the ground truth clusters
            ground_truth_dict = test_ground_truth.get(schema, {})
            actual_clusters = [[tuple(inner_list) for inner_list in outer_list] for outer_list in ground_truth_dict.values()]

            # Calculate the precision, recall, and F1-score
            precision, recall, f1_score = calc_scores(actual_clusters, predicted_clusters)

            # Print the evaluation metrics
            print(f"Schema: {schema}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")

            # Print the actual and predicted clusters
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


def group_paths(df, test_ground_truth, min_common_keys=2):
    """
    Group paths that have at least two distinct nested keys in common per filename.

    Args:
        df (pd.DataFrame): DataFrame containing columns "Path", "Distinct_keys", and "Filename".
        min_common_keys (int, optional): Minimum number of distinct nested keys required for paths to be grouped together. The default is 2.

    Returns:
        dict: Dictionary where keys are filenames and values are lists of groups of paths
              with at least the specified number of distinct keys in common.
    """

    # Group data by filename
    for filename, group in tqdm.tqdm(df.groupby("Filename"), position=1, leave=False, total=len(df.groupby("Filename")), desc="group"):
        paths = group["Path"].tolist()
        distinct_keys = group["Distinct_keys"].tolist()

        # Use a list to keep track of groups
        groups = []
        
        # Iterate through each path and try to group them
        for i, (path, keys) in enumerate(zip(paths, distinct_keys)):
            path = ast.literal_eval(path)
            keys = ast.literal_eval(keys)

            added_to_group = False
            for group in groups:
                # Check if this path shares at least min_common_keys with any path in the current group
                if any(len(set(keys) & set(other_keys)) >= min_common_keys for other_path, other_keys in group):
                    group.append((path, keys))
                    added_to_group = True
                    break
            if not added_to_group:
                # Create a new group if it doesn't fit in any existing group
                groups.append([(path, keys)])
        
        # Convert groups of tuples to groups of paths
        valid_groups = [group for group in groups if len(group) > 1]
        grouped_paths = [[path for path, _ in group] for group in valid_groups]
    

        # Get the ground truth clusters
        ground_truth_dict = test_ground_truth.get(filename, {})

        # Evaluate
        evaluate_grouped_paths(grouped_paths, ground_truth_dict, filename)


def evaluate_grouped_paths(predicted_clusters, ground_truth_dict, filename):

    actual_clusters = [[tuple(inner_list) for inner_list in outer_list] for outer_list in ground_truth_dict.values()]

    # Calculate the precision, recall, and F1-score
    precision, recall, f1_score = calc_scores(actual_clusters, predicted_clusters)

    # Print the evaluation metrics
    print(f"Schema: {filename}, Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")

    # Print the actual and predicted clusters
    print("Actual clusters")
    for actual_cluster in actual_clusters:
        print(actual_cluster)

    print()

    print("Predicted clusters:")
    for predicted_cluster in predicted_clusters:
        print(predicted_cluster)
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print()


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Convert non-serializable objects to a string representation
        return str(obj)

def dereference(obj, schema, definitions_to_keep):
    if isinstance(obj, dict):
        if "$ref" in obj:
            ref = obj["$ref"]
            if ref not in definitions_to_keep and isinstance(ref, str) and len(ref) > 0:
                referenced_obj = dereference(get_definition(ref, schema), schema, definitions_to_keep)
                obj = referenced_obj
        else:
            for key, value in obj.items():
                obj[key] = dereference(value, schema, definitions_to_keep)
    elif isinstance(obj, list):
        obj = [dereference(item, schema, definitions_to_keep) for item in obj]
    return obj

def get_definition(ref, schema):
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
    else:
        # Handle other reference formats if needed
        pass

    
def dereference_and_calculate_schema_size(schema, definitions_to_keep=None):
    """
    Load, dereference, and calculate the size of the JSON schema in bytes.

    Args:
        schema (str): Name of the JSON schema file.
        definitions_to_keep (list, optional): List of definitions to exclude from dereferencing.

    Returns:
        tuple: The dereferenced JSON schema and its size in bytes.
    """

    schema_path = os.path.join(SCHEMA_FOLDER, schema)

    # Load the JSON schema from the file
    with open(schema_path, 'r') as file:
        schema = json.load(file)

    # If definitions_to_keep is not provided, initialize it as an empty list
    if definitions_to_keep is None:
        definitions_to_keep = []

    # Dereference the schema
    dereferenced_schema = dereference(schema, schema, definitions_to_keep)

    # Calculate the size of the dereferenced schema in bytes using a custom JSON encoder
    schema_str = json.dumps(dereferenced_schema, separators=(',', ':'))
    schema_size = len(schema_str.encode("utf-8"))
    
    return schema_size