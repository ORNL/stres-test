import os
import torch
import random
import numpy as np

from transformers import AutoModelForSequenceClassification

from OpenTSLM.src.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo



def set_all_seeds(seed=42):
    """
    Created with help of Google AI.
    Sets the seed for Python's random module, NumPy, and PyTorch. Also configures PyTorch for deterministic behavior with CuDNN.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch on CPU
    torch.cuda.manual_seed(seed)  # PyTorch on GPU (if available)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    # Configure CuDNN for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed (optional, but good for reproducibility)
    os.environ["PYTHONHASHSEED"] = str(seed)
    pass


def reformat_glue_example_as_prompt(task_name, example):
    """
    Format an example from the GLUE dataset into a single prompt string.
    Args:
        task_name (str): The name of the GLUE task.
        example (dict): A single example from the dataset.
    Returns:
        str: A formatted prompt string based on the task and example.
        any: Ground truth label for the example.
    """
    if task_name in ["mnli", "mnli_matched", "mnli_mismatched", "ax"]:
        # MNLI-like tasks (including AX)
        prompt = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nEntailment label (Entailment, Neutral, or Contradiction): "
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        label = label_map[example["label"]]
    elif task_name == "sst2":
        prompt = f"Text: {example['sentence']}\nSentiment (Positive or Negative): "
        label = "Positive" if example["label"] == 1 else "Negative"
    elif task_name == "cola":
        prompt = f"Sentence: {example['sentence']}\nIs the sentence linguistically acceptable? Answer Yes or No: "
        label = "Yes" if example["label"] == 1 else "No"
    elif task_name == "rte":
        prompt = f"Premise: {example['sentence1']}\nHypothesis: {example['sentence2']}\nIs the hypothesis entailed by the premise? Answer Yes or No: "
        label = "Yes" if example["label"] == 1 else "No"
    elif task_name == "qqp":
        prompt = f"Question 1: {example['question1']}\nQuestion 2: {example['question2']}\nAre these two questions essentially asking the same thing? Answer Yes or No: "
        label = "Yes" if example["label"] == 1 else "No"
    elif task_name == "mrpc":
        prompt = f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nAre these two sentences semantically equivalent? Answer Yes or No: "
        label = "Yes" if example["label"] == 1 else "No"
    elif task_name == "qnli":
        prompt = f"Question: {example['question']}\nSentence: {example['sentence']}\nIs the sentence relevant to answer the question? Answer Yes or No: "
        label = "Yes" if example["label"] == 1 else "No"
    elif task_name == "stsb":
        prompt = f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nWhat is the semantic similarity score between these sentences (0 to 5)? "
        label = str(example["label"])
    elif task_name == "wnli":# Winograd NLP Inference (WNLI)
        prompt = f"Sentence: {example['sentence1']}\nDoes the sentence resolve the coreference correctly? Answer Yes or No: "
        label = "Yes" if example["label"] == 1 else "No"
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    return prompt, label

def reformat_glue_example_as_prompt_with_int_label(task_name, example):
    """
    Format an example from the GLUE dataset into a single prompt string.
    Args:
        task_name (str): The name of the GLUE task.
        example (dict): A single example from the dataset.
    Returns:
        str: A formatted prompt string based on the task and example.
        any: Ground truth label for the example.
    """
    if task_name in ["mnli", "mnli_matched", "mnli_mismatched", "ax"]:
        # MNLI-like tasks (including AX)
        prompt = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nEntailment label (Entailment, Neutral, or Contradiction): "
        #label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        # label = label_map[example["label"]]
    elif task_name == "sst2":
        prompt = f"Text: {example['sentence']}\nSentiment (Positive or Negative): "
        # label = "Positive" if example["label"] == 1 else "Negative"
    elif task_name == "cola":
        prompt = f"Sentence: {example['sentence']}\nIs the sentence linguistically acceptable? Answer Yes or No: "
        # label = "Yes" if example["label"] == 1 else "No"
    elif task_name == "rte":
        prompt = f"Premise: {example['sentence1']}\nHypothesis: {example['sentence2']}\nIs the hypothesis entailed by the premise? Answer Yes or No: "
        # label = "Yes" if example["label"] == 1 else "No"
    elif task_name == "qqp":
        prompt = f"Question 1: {example['question1']}\nQuestion 2: {example['question2']}\nAre these two questions essentially asking the same thing? Answer Yes or No: "
        # label = "Yes" if example["label"] == 1 else "No"
    elif task_name == "mrpc":
        prompt = f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nAre these two sentences semantically equivalent? Answer Yes or No: "
        # label = "Yes" if example["label"] == 1 else "No"
    elif task_name == "qnli":
        prompt = f"Question: {example['question']}\nSentence: {example['sentence']}\nIs the sentence relevant to answer the question? Answer Yes or No: "
        # label = "Yes" if example["label"] == 1 else "No"
    elif task_name == "stsb":
        prompt = f"Sentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nWhat is the semantic similarity score between these sentences (0 to 5)? "
        # label = str(example["label"])
    elif task_name == "wnli":# Winograd NLP Inference (WNLI)
        prompt = f"Sentence: {example['sentence1']}\nDoes the sentence resolve the coreference correctly? Answer Yes or No: "
        # label = "Yes" if example["label"] == 1 else "No"
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    return prompt, example["label"]#label


# def load_llm(log_file_path, llm_id, device):
#     open(log_file_path, "a").write("load_llm: START \n")
#     if os.path.exists(chkpt_path):
#         checkpoint = torch.load(chkpt_path, map_location="cpu", weights_only=False) # Always load checkpoint to CPU first to avoid GPU OOM spikes
#         if model_type == "OpenTSLMFlamingo":
#             model = OpenTSLMFlamingo(
#                 cross_attn_every_n_layers=1,
#                 llm_id=llm_id,
#                 device=device, 
#             )
#             model_state = checkpoint["model_state"]
#             if hasattr(model, "module"): # Add 'module.' prefix for DDP
#                 model_state = {f"module.{k}": v for k, v in model_state.items()}
#             try: # Load state dict with strict=False to handle missing keys
#                 try:
#                     missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
#                 except:
#                     open(log_file_path, "a").write("load_model_from_checkpoint: loading state dict failed on initial attempt - retrying after concatenating more positional encoding parameters to vision/time-series tokenizer \n")
#                     og_pos_enc_param_dims = model.model.vision_encoder.pos_embed.size()
#                     new_pos_enc_params = torch.nn.Parameter(torch.randn(og_pos_enc_param_dims[0], max_patches-og_pos_enc_param_dims[1], og_pos_enc_param_dims[2])).to(model.device)
#                     model.model.vision_encoder.pos_embed = torch.nn.Parameter(torch.cat([model.model.vision_encoder.pos_embed,new_pos_enc_params], dim=1))
#                     model.llm.vision_encoder.pos_embed = model.model.vision_encoder.pos_embed
#                     missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
#                 if missing_keys:
#                     open(log_file_path, "a").write("⚠️  Warning: Missing keys when loading checkpoint from "+str(chkpt_path)+": \n")
#                     for key in missing_keys[:10]:  # Show first 10 missing keys
#                         open(log_file_path, "a").write("   - "+str(key)+" \n")
#                     if len(missing_keys) > 10:
#                         open(log_file_path, "a").write("   ... and "+str(len(missing_keys) - 10)+" more keys \n")
#                 if unexpected_keys:
#                     open(log_file_path, "a").write("⚠️  Warning: Unexpected keys when loading checkpoint from "+str(chkpt_path)+": \n")
#                     for key in unexpected_keys[:10]:  # Show first 10 unexpected keys
#                         open(log_file_path, "a").write("   - "+str(key)+" \n")
#                     if len(unexpected_keys) > 10:
#                         open(log_file_path, "a").write("   ... and "+str(len(unexpected_keys) - 10)+" more keys \n")
#             except Exception as e:
#                 open(log_file_path, "a").write("load_model_from_checkpoint: RuntimeError!! Failed to load model state from chkpt_path == "+str(chkpt_path)+" \n")
#                 raise RuntimeError(f"Failed to load model state from checkpoint from {chkpt_path}: {e}")
#         else:
#             open(log_file_path, "a").write("load_model_from_checkpoint: ValueError!! Unrecognized model_type == "+str(model_type)+" \n")
#             raise ValueError(model_type)
#     else: 
#         open(log_file_path, "a").write("load_model_from_checkpoint: ValueError!! Path doesn't exist; chkpt_path == "+str(chkpt_path)+" \n")
#         raise ValueError(chkpt_path)
#     open(log_file_path, "a").write("load_model_from_checkpoint: STOP \n")
#     return model


def load_model_from_checkpoint(chkpt_path, log_file_path, llm_id, device, max_patches=None, model_type="OpenTSLMFlamingo"):
    open(log_file_path, "a").write("load_model_from_checkpoint: START \n")
    if os.path.exists(chkpt_path):
        checkpoint = torch.load(chkpt_path, map_location="cpu", weights_only=False) # Always load checkpoint to CPU first to avoid GPU OOM spikes
        if model_type == "OpenTSLMFlamingo":
            model = OpenTSLMFlamingo(
                cross_attn_every_n_layers=1,
                llm_id=llm_id,
                device=device, 
            )
            model_state = checkpoint["model_state"]
            if hasattr(model, "module"): # Add 'module.' prefix for DDP
                model_state = {f"module.{k}": v for k, v in model_state.items()}
            try: # Load state dict with strict=False to handle missing keys
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
                except:
                    open(log_file_path, "a").write("load_model_from_checkpoint: loading state dict failed on initial attempt - retrying after concatenating more positional encoding parameters to vision/time-series tokenizer \n")
                    og_pos_enc_param_dims = model.model.vision_encoder.pos_embed.size()
                    assert max_patches is not None
                    new_pos_enc_params = torch.nn.Parameter(torch.randn(og_pos_enc_param_dims[0], max_patches-og_pos_enc_param_dims[1], og_pos_enc_param_dims[2])).to(model.device)
                    model.model.vision_encoder.pos_embed = torch.nn.Parameter(torch.cat([model.model.vision_encoder.pos_embed,new_pos_enc_params], dim=1))
                    model.llm.vision_encoder.pos_embed = model.model.vision_encoder.pos_embed
                    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
                if missing_keys:
                    open(log_file_path, "a").write("⚠️  Warning: Missing keys when loading checkpoint from "+str(chkpt_path)+": \n")
                    for key in missing_keys[:10]:  # Show first 10 missing keys
                        open(log_file_path, "a").write("   - "+str(key)+" \n")
                    if len(missing_keys) > 10:
                        open(log_file_path, "a").write("   ... and "+str(len(missing_keys) - 10)+" more keys \n")
                if unexpected_keys:
                    open(log_file_path, "a").write("⚠️  Warning: Unexpected keys when loading checkpoint from "+str(chkpt_path)+": \n")
                    for key in unexpected_keys[:10]:  # Show first 10 unexpected keys
                        open(log_file_path, "a").write("   - "+str(key)+" \n")
                    if len(unexpected_keys) > 10:
                        open(log_file_path, "a").write("   ... and "+str(len(unexpected_keys) - 10)+" more keys \n")
            except Exception as e:
                open(log_file_path, "a").write("load_model_from_checkpoint: RuntimeError!! Failed to load model state from chkpt_path == "+str(chkpt_path)+" \n")
                raise RuntimeError(f"Failed to load model state from checkpoint from {chkpt_path}: {e}")
        elif "foundation_model" in model_type:
            model = AutoModelForSequenceClassification.from_pretrained(chkpt_path)
            # model.to(device)
        else:
            open(log_file_path, "a").write("load_model_from_checkpoint: ValueError!! Unrecognized model_type == "+str(model_type)+" \n")
            raise ValueError(model_type)
    else: 
        open(log_file_path, "a").write("load_model_from_checkpoint: ValueError!! Path doesn't exist; chkpt_path == "+str(chkpt_path)+" \n")
        raise ValueError(chkpt_path)
    open(log_file_path, "a").write("load_model_from_checkpoint: STOP \n")
    return model


def get_token_embedding_summary_stats(embeddings):
    assert len(embeddings.shape) == 2
    TOK_DIM = 0
    FEATURE_DIM = 1
    num_toks = embeddings.shape[TOK_DIM]
    num_tok_features = embeddings.shape[FEATURE_DIM]
    summary = {
        "token feature means": np.mean(embeddings, axis=0), 
        "token feature std devs": np.std(embeddings, axis=0), 
        "outlier token ids by feature": {'feature'+str(i): [] for i in range(num_tok_features)},
        "outlier array": np.zeros(embeddings.shape)
    }
    for i, (avg, stddev) in enumerate(zip(summary["token feature means"], summary["token feature std devs"])):
        for row_ind in range(num_toks):
            if np.abs(embeddings[row_ind, i] - avg) > 4.*stddev:
                summary["outlier token ids by feature"]['feature'+str(i)].append(row_ind)
                summary['outlier array'][row_ind, i] = 1
    return summary