import os
import re
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

import torch
import torch.nn.functional as F

from usgs_data.data_utils import prepare_text_timeseries_sample_batch_for_generative_eval
from utils.time_series_utils import extend_time_series_to_match_patch_size_and_aggregate


RECOGNIZED_GENERAL_EDGE_ROOTS = [
    # event locational information
    'originated from GPS coordinates', 
    "struck the region referred to as the", 
    'occured nearest to major population of', 
    'had a point of origin from the nearest major population that was approximately', 
    'occured in an area with estimated population density (per square km) of', 
    'was located ', #{} of', #.format(event_orig_dist_from_place), 
    "has an origin characterized by a ", #{} with a value of".format(tensor_element_value_name)
    'occured in the region referred to as the', 
    'occured in the country named', 
    # general location information
    'experienced an earthquake on the date of', 
    'encompasses a city named', 
    'is located in the region referred to as the', 
    'is located in the', 
    'is located ', #{} of'.format(event_origin_rel_to_city_country_str_reverse_dir), 
    'was the closest major population to', 
    'has structures characterized by the statement that', 
    'has experienced earthquakes characterized by the statement that', 
    'has a population of', 
    # fault info
    "has the fault type of", 
    "moves ", #+split_kg_relation[1], 
    "converges at a rate of ", #{} with".format(conv_rt), 
    # event identifiers
    'has Event ID:', 
    'is referred to as the', 
    'occured at UTC time', 
    'occured on the date of', 
    "occured in the year", 
    # event characteristic stats
    "has a magnitude of", 
    'had approximate magnitude of', 
    'originated from a depth of', 
    "has a W-phase Moment Tensor with a ", #{} element value of".format(tensor_element_value_name), text)
    # event effects
    'triggered a shaking related fatalities alert level of', 
    'triggered an economical loss alert level of', 
    'exposed the population to shaking with a rated level of', 
    'exposed an estimated population to earthquake shaking of around', 
    'caused shaking level of ', #{} in'.format(area_exposure_levels_by_city[city_name]), 
    # other
    "is an", 
    "is a", 
    'is oriented', 
    'experienced shaking level of ', #{} caused by'.format(area_exposure_levels_by_city[city_name]), 
    'is contextualized by the statement:', 
    'has ISO value of', 
]


def _normalize_decoded_text(text):
    if text is None:
        return None
    # Remove explicit special-token markers if present and strip all whitespace.
    text = re.sub(r"<\|[^>]*\|>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u2581", "").replace("\u0120", "").replace("\u200b", "")
    return re.sub(r"\s+", "", text)

def extract_numbers_as_floats(text):
    """
    Extract numeric substrings from mixed text and return them as floats.
    Notes:
    - Preserves decimals like "3.14" as a single number.
    - Supports optional leading '+' / '-'.
    - Does not treat '.' as a decimal unless digits are on both sides.
    """
    if text is None:
        return []
    return [float(match) for match in re.findall(r"[+-]?(?:\d+\.\d+|\d+)", str(text))]
    
def _euclidean_dist_avgd_across_str(pred_text, targ_text): 
    """ Note: assumes ordering is important (e.g. coordinates) and thus does not include post-hoc matching logic """
    dist = np.inf
    if pred_text is None or targ_text is None:
        return dist
    pred_norm = _normalize_decoded_text(pred_text)
    targ_norm = _normalize_decoded_text(targ_text)
    if pred_norm is None or targ_norm is None:
        return dist
    pred_vals = extract_numbers_as_floats(pred_norm)
    targ_vals = extract_numbers_as_floats(targ_norm)
    if len(targ_vals) == 0:
        if len(pred_vals) == 0:
            return 0.
        else: 
            return dist
    if len(pred_vals) == 0: 
        assert len(targ_vals) > 0
        return dist
    return np.sqrt(sum([(a-b)**(2.) for a,b in zip(pred_vals, targ_vals)]))

def _char_level_normalized_hamming_dist(pred_text, targ_text): 
    dist = 1.#np.inf
    if pred_text is None or targ_text is None:
        return dist
    pred_norm = _normalize_decoded_text(pred_text)
    targ_norm = _normalize_decoded_text(targ_text)
    if pred_norm is None or targ_norm is None:
        return dist
    aligned_len = min(len(pred_norm), len(targ_norm))
    if aligned_len == 0:
        return dist
    mismatches = sum(1 for a,b in zip(pred_norm, targ_norm) if a != b)
    dist = mismatches / (1.*aligned_len)
    return dist 

def _char_level_normalized_Damerau_Levenshtein_dist(pred_text, targ_text): 
    dist = 1.#np.inf
    if pred_text is None or targ_text is None:
        return dist
    pred_norm = _normalize_decoded_text(pred_text)
    targ_norm = _normalize_decoded_text(targ_text)
    if pred_norm is None or targ_norm is None:
        return dist
    pred_norm_len = len(pred_norm) # m
    targ_norm_len = len(targ_norm) # n
    max_len = max(pred_norm_len, targ_norm_len)
    if max_len == 0:
        return dist
    # Optimal string alignment Damerau-Levenshtein distance: allows insertions, deletions, substitutions, and adjacent transpositions.
    # initialize the object for recording edit costs
    min_edit_dists_dynam_program_arr = [[0] * (targ_norm_len + 1) for _ in range(pred_norm_len + 1)]
    for i in range(pred_norm_len + 1):
        min_edit_dists_dynam_program_arr[i][0] = i
    for j in range(targ_norm_len + 1):
        min_edit_dists_dynam_program_arr[0][j] = j
    # fill in the costs
    for i in range(1, pred_norm_len + 1):
        for j in range(1, targ_norm_len + 1):
            cost = 0 if pred_norm[i - 1] == targ_norm[j - 1] else 1
            # determine which edit action results in the minimal cost and record
            min_edit_dists_dynam_program_arr[i][j] = min(
                min_edit_dists_dynam_program_arr[i - 1][j] + 1,         # deletion
                min_edit_dists_dynam_program_arr[i][j - 1] + 1,         # insertion
                min_edit_dists_dynam_program_arr[i - 1][j - 1] + cost,  # substitution
            )
            if (
                i > 1 and j > 1
                and pred_norm[i - 1] == targ_norm[j - 2]
                and pred_norm[i - 2] == targ_norm[j - 1]
            ):
                min_edit_dists_dynam_program_arr[i][j] = min(  # transposition
                    min_edit_dists_dynam_program_arr[i][j], 
                    min_edit_dists_dynam_program_arr[i - 2][j - 2] + 1
                )  # transposition
    # the (raw) distance is the final element in the cost array
    raw_dist = min_edit_dists_dynam_program_arr[pred_norm_len][targ_norm_len]
    # normalize the distance before returning
    dist = raw_dist / (1.0 * max_len)
    return dist 


def evaluate_model_kg_predictions_on_kg_text_timeseries_data(
    model, tokenizer, dataset, data_POST_PROMPT_END, log_file_path, max_patches, max_new_tokens_per_samp=30, top_k=10, 
):  
    eval_results = dict()
    with open(log_file_path, "a") as logfile: 
        logfile.write("evaluate_model_kg_predictions_on_kg_text_timeseries_data: START \n")
        if dataset.file is None and getattr(dataset, "hdf5_path", None):
            dataset.file = h5py.File(dataset.hdf5_path, 'r')
        assert len(list(set(RECOGNIZED_GENERAL_EDGE_ROOTS))) == len(RECOGNIZED_GENERAL_EDGE_ROOTS)
        
        # LOOP OVER DATA AND RUN EVALUATION ------------------------------------------------------
        total_num_eval_samps = 0
        model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                if i % 100 == 0:
                    logfile.write("evaluate_model_kg_predictions_on_kg_text_timeseries_data: i == "+str(i)+" \n")
                    
                # get relational edge for current sample query
                sample_idx = i
                if hasattr(dataset, "get_hdf5_index"):
                    sample_idx = dataset.get_hdf5_index(i)
                curr_samp_grp = dataset.file[f'sample_{sample_idx}']
                curr_samp_query = ''.join([elem.decode('utf-8') for elem in curr_samp_grp['text_based_knowedge_graph_sample']['query'][()]])
                
                query_relation_edge = None
                for rger in RECOGNIZED_GENERAL_EDGE_ROOTS:
                    if rger in curr_samp_query:
                        query_relation_edge = rger+curr_samp_query.split(rger)[1]
                        break
                if query_relation_edge is None:
                    logfile.write("evaluate_model_kg_predictions_on_kg_text_timeseries_data: DEBUGGING -- NO RECOGNIZED EDGE IN curr_samp_query == "+str(curr_samp_query)+" \n")
                    logfile.write("evaluate_model_kg_predictions_on_kg_text_timeseries_data: DEBUGGING -- RECOGNIZED_GENERAL_EDGE_ROOTS == "+str(RECOGNIZED_GENERAL_EDGE_ROOTS)+" \n")
                    raise ValueError("evaluate_model_kg_predictions_on_kg_text_timeseries_data: NO EDGE FOUND FOR SAMPLE i == "+str(i))
                if query_relation_edge not in eval_results.keys():
                    eval_results[query_relation_edge] = []
                
                curr_samp = dataset[i]
                curr_samp["time_series"] = [torch.as_tensor(ts, dtype=torch.float32) for ts in curr_samp["time_series"]]
                curr_samp["time_series"] = torch.cat([t.unsqueeze(0) for t in curr_samp["time_series"]], dim=0).to(model.device)
                patch_size = model.model.vision_encoder.patch_size
                curr_samp = extend_time_series_to_match_patch_size_and_aggregate([curr_samp], patch_size=patch_size)[0]
                
                input_ids, images, attention_mask, _ = model.pad_and_apply_batch([curr_samp,], include_labels=True)
                gen_outs = model.llm.generate(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens_per_samp,
                    eos_token_id=model.text_tokenizer.eos_token_id,
                    pad_token_id=model.text_tokenizer.pad_token_id, 
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                target_ids = model.text_tokenizer(
                    curr_samp["answer"], return_tensors="pt", add_special_tokens=False,
                ).input_ids.to(model.device)
                pred_ids = gen_outs.sequences[:, input_ids.shape[1]:] # grab only the predicted tokens that represent the first answer_length tokens predicted AFTER the tokens representing input reconstruction (note that .generate() appears to prepend the start token)
                pred_probs = [F.softmax(s, dim=-1) for s in gen_outs.scores]

                results_by_seq_pos = []
                for index in range(min(target_ids.size()[-1], pred_ids.size()[1])):
                    pred_token_probs = torch.squeeze(pred_probs[index])
                    sorted_pred_token_probs, sorted_pred_token_ids = pred_token_probs.sort(descending=True)
                    targ_tok_rank = (sorted_pred_token_ids == target_ids[0,index]).nonzero(as_tuple=True)[0].item() # see https://stackoverflow.com/questions/47863001/how-pytorch-tensor-get-the-index-of-specific-value
                    perf_on_ans_tokens = {
                        "answer token id": sorted_pred_token_ids[0].detach().cpu().item(), 
                        "answer token rank": 0, 
                        "answer token prob": sorted_pred_token_probs[0].detach().cpu().item(), 
                        "target token id": target_ids[0,index].detach().cpu().item(), 
                        "target token rank": targ_tok_rank, # see https://stackoverflow.com/questions/47863001/how-pytorch-tensor-get-the-index-of-specific-value
                        "target token prob": sorted_pred_token_probs[targ_tok_rank].detach().cpu().item(), 
                    }
                    top_k_token_details = {
                        "top_"+str(i)+"_token": {
                            "Prob": sorted_pred_token_probs[i].detach().cpu().item(), "Token": sorted_pred_token_ids[i].detach().cpu().item(), 
                        } for i in range(top_k)
                    }
                    results_by_seq_pos.append({"perf_on_ans_tokens": perf_on_ans_tokens, "top_k_token_details": top_k_token_details, })

                curr_results = {
                    "input_ids": input_ids.cpu(), 
                    "target_ids": target_ids.cpu(), 
                    "pred_probs": [p.cpu() for p in pred_probs], 
                    "pred_ids": pred_ids.cpu(), 
                    "results_by_seq_pos": results_by_seq_pos, 
                }
                eval_results[query_relation_edge].append(curr_results)

                # if i == 10:  # FOR DEBUGGING PURPOSES
                #     logfile.write("evaluate_model_kg_predictions_on_kg_text_timeseries_data: BREAKING FOR DEBUGGING PURPOSES \n") # FOR DEBUGGING PURPOSES
                #     break # FOR DEBUGGING PURPOSES

        for qre in eval_results.keys():
            logfile.write("evaluate_model_kg_predictions_on_kg_text_timeseries_data: qre == "+str(qre)+"; len(eval_results[qre]) == "+str(len(eval_results[qre]))+" \n")
        logfile.write("evaluate_model_kg_predictions_on_kg_text_timeseries_data: STOP \n")
    return eval_results




def evaluate_model_text_predictions_on_text_timeseries_data(
    model, tokenizer, dataloader, log_file_path, max_patches, challenge_categories, prompt_filter_str_to_challenge_category_map, 
    max_samps_per_category=50, max_new_tokens_per_samp=30, top_k=10, universal_prompt_pretext=None, 
):
    open(log_file_path, "a").write("evaluate_model_text_predictions_on_text_timeseries_data: START \n")
    eval_results = {chal_key: [] for chal_key in challenge_categories}
    eval_device = None
    og_device = model.device

    # SET UP MODEL ON EVAL DEVICE (WITH ACCELERATION CAPABILITIES IF POSSIBLE) ---------------
    open(log_file_path, "a").write("evaluate_model_text_predictions_on_text_timeseries_data: initial model.device == "+str(model.device)+" \n")
    if torch.cuda.is_available():
        eval_device = 'cuda'
    elif torch.backends.mps.is_available():
        eval_device = 'mps'
    else:
        eval_device = 'cpu'

    if eval_device != model.device:
        open(log_file_path, "a").write("evaluate_model_text_predictions_on_text_timeseries_data: moving model to eval_device == "+str(eval_device)+" \n")
        model.device = eval_device
        model.to(model.device)
    open(log_file_path, "a").write("evaluate_model_text_predictions_on_text_timeseries_data: final model.device == "+str(model.device)+" \n")
    
    # LOOP OVER DATA AND RUN EVALUATION ------------------------------------------------------
    total_num_eval_samps = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 100 == 0:
                open(log_file_path, "a").write("evaluate_model_text_predictions_on_text_timeseries_data: i == "+str(i)+" \n")
            assert len(batch) == 1
            samp_category = None#get_samp_category(batch[0]['pre_prompt'])
            for cat_prompt_flag in prompt_filter_str_to_challenge_category_map.keys():
                if cat_prompt_flag in batch[0]['pre_prompt']: #pre_prompt_text:
                    samp_category = prompt_filter_str_to_challenge_category_map[cat_prompt_flag]
            if samp_category in challenge_categories and len(eval_results[samp_category]) < max_samps_per_category:
                open(log_file_path, "a").write("evaluate_model_text_predictions_on_text_timeseries_data: \t samp_category == "+str(samp_category)+" \n")
                input_ids, images, attention_mask, target_ids = prepare_text_timeseries_sample_batch_for_generative_eval(
                    batch, tokenizer, eval_device, log_file_path, universal_prompt_pretext=universal_prompt_pretext, 
                )
                prompt_length = input_ids.size()[-1]
                answer_length = target_ids.size()[-1]
                gen_outs = None
                if max_patches is not None:
                    images = images[:,:,:,:max_patches]
                    gen_outs = model.model.generate(
                        images, input_ids, attention_mask, max_new_tokens=max_new_tokens_per_samp, return_dict_in_generate=True, output_scores=True
                    )# see https://github.com/StanfordBDHG/OpenTSLM/blob/main/src/model/llm/OpenTSLMFlamingo.py and https://github.com/mlfoundations/open_flamingo/blob/914076ee07d0235943f1556e9aaf2fb974337f0e/open_flamingo/src/flamingo.py and https://discuss.huggingface.co/t/how-can-i-obtain-the-logits-via-model-generate/110636/2
                else: # case where model is an llm
                    raise NotImplementedError()
                    gen_outs = model(input_ids)
                pred_ids = gen_outs.sequences[:,prompt_length:prompt_length+answer_length] # grab only the predicted tokens that represent the first answer_length tokens predicted AFTER the tokens representing input reconstruction (note that .generate() appears to prepend the start token)
                pred_probs = [F.softmax(s, dim=-1) for s in gen_outs.scores]

                results_by_seq_pos = []
                for index in range(min(answer_length, pred_ids.size()[1])):
                    pred_token_probs = torch.squeeze(pred_probs[index])
                    sorted_pred_token_probs, sorted_pred_token_ids = pred_token_probs.sort(descending=True)
                    targ_tok_rank = (sorted_pred_token_ids == target_ids[0,index]).nonzero(as_tuple=True)[0].item() # see https://stackoverflow.com/questions/47863001/how-pytorch-tensor-get-the-index-of-specific-value
                    perf_on_ans_tokens = {
                        "answer token id": sorted_pred_token_ids[0].item(), 
                        "answer token rank": 0, 
                        "answer token prob": sorted_pred_token_probs[0].item(), 
                        "target token id": target_ids[0,index].item(), 
                        "target token rank": targ_tok_rank, # see https://stackoverflow.com/questions/47863001/how-pytorch-tensor-get-the-index-of-specific-value
                        "target token prob": sorted_pred_token_probs[targ_tok_rank].item(), 
                    }
                    top_k_token_details = {
                        "top_"+str(i)+"_token": {
                            "Prob": sorted_pred_token_probs[i].item(), "Token": sorted_pred_token_ids[i].item(), #"Prob": sorted_pred_token_probs[0, i].item(), "Token": sorted_pred_token_ids[0, i], #"Logit": pred_logits[0, index, sorted_token_positions[0, i]].item(), "Prob": sorted_token_probs[0, i].item(), "Token": sorted_token_positions[0, i], 
                        } for i in range(top_k)
                    }
                    results_by_seq_pos.append({"perf_on_ans_tokens": perf_on_ans_tokens, "top_k_token_details": top_k_token_details, })
                curr_results = {
                    "input_ids": input_ids.cpu(), 
                    "target_ids": target_ids.cpu(), 
                    "pred_probs": [p.cpu() for p in pred_probs], 
                    "pred_ids": pred_ids.cpu(), 
                    "results_by_seq_pos": results_by_seq_pos, 
                }
                eval_results[samp_category].append(curr_results)
            stop_iterating = True
            for sc in eval_results.keys():
                if len(eval_results[sc]) < max_samps_per_category:
                    stop_iterating = False
                    break
            if stop_iterating:
                break

    model.device = og_device
    model.to(og_device) # return the model to its original device
    open(log_file_path, "a").write("evaluate_model_text_predictions_on_text_timeseries_data: STOP \n")
    return eval_results


def visualize_llm_classifier_performance(log_file_path, save_dir, results_dict):
    """
    Code written with assistance from GPT-4o
    Visualizes LLM performances
    Parameters:
        results_dict: Dictionary containing performance results for various models. Structured as 
            results_dict[model_key] = { 
                perturbation_key: { 
                    input_sample_category_key: [ 
                        { 
                            ...
                            "results_by_seq_pos": [
                                { 
                                    "answer class rank": 0, 
                                    "answer class prob": sorted_pred_class_probs[0], 
                                    "target class rank": (sorted_pred_class_ids == target_ids[0,index]).nonzero(as_tuple=True)[0], 
                                    "target class prob": (sorted_pred_class_probs == target_ids[0,index]).nonzero(as_tuple=True)[0], 
                                }, ... 
                            ] 
                        }, ...
                    ], ... 
                }, ..
            }
    Returns:
        None (Generates plots and prints aggregate statistics).
    """
    open(log_file_path, "a").write("visualize_llm_classifier_performance: START \n")
    # Aggregate the statistics ---------------------------------------------------------------
    open(log_file_path, "a").write("visualize_llm_classifier_performance: aggregating stats \n")
    aggregate_stats = {}
    for model_key in results_dict.keys():
        # open(log_file_path, "a").write("visualize_llm_classifier_performance: DEBUGGING -- model_key == "+str(model_key)+" \n")
        for perturbation_key in results_dict[model_key].keys():
            # open(log_file_path, "a").write("visualize_llm_classifier_performance: DEBUGGING -- \tperturbation_key == "+str(perturbation_key)+" \n")
            agg_model_key = str(model_key)+'~~'+str(perturbation_key)
            aggregate_stats[agg_model_key] = dict()
            # open(log_file_path, "a").write("visualize_llm_classifier_performance: DEBUGGING -- \results_dict[model_key][perturbation_key] == "+str(results_dict[model_key][perturbation_key])+" \n")
            for challenge_key in results_dict[model_key][perturbation_key].keys(): 
                aggregate_stats[agg_model_key][challenge_key] = {
                    "answer_class_matches": [],
                    "answer_class_probs": [],
                    "target_class_ranks": [],
                    "target_class_probs": [],
                }
                for sample_results in results_dict[model_key][perturbation_key][challenge_key]["sample_level_results"]:
                    for seq_pos_pred_results in sample_results["results_by_seq_pos"]:
                        perf_on_ans = seq_pos_pred_results#["perf_on_ans_tokens"]
                        # Collect relevant statistics
                        answer_class_match = 1. if perf_on_ans["target class rank"] == 0. else 0.#.item() == 0. else 0.
                        answer_prob = perf_on_ans["answer class prob"]#.item()  # as float
                        target_rank = perf_on_ans["target class rank"]#.item()  # as integer
                        target_prob = perf_on_ans["target class prob"]#.item()  # as float
                        # Append to model-specific aggregate statistics
                        aggregate_stats[agg_model_key][challenge_key]["answer_class_matches"].append(answer_class_match)
                        aggregate_stats[agg_model_key][challenge_key]["answer_class_probs"].append(answer_prob)
                        aggregate_stats[agg_model_key][challenge_key]["target_class_ranks"].append(target_rank)
                        aggregate_stats[agg_model_key][challenge_key]["target_class_probs"].append(target_prob)
    
    # Compute summary statistics -------------------------------------------------------------
    open(log_file_path, "a").write("visualize_llm_classifier_performance: summarizing stats \n")
    summary_stats = {}
    for model_key in aggregate_stats.keys():
        summary_stats[model_key] = dict()
        for challenge_key in aggregate_stats[model_key].keys():
            stats = aggregate_stats[model_key][challenge_key]
            summary_stats[model_key][challenge_key] = {
                "mean_answer_class_match_rate": np.mean(stats["answer_class_matches"]),
                "std_answer_class_match_rate": np.std(stats["answer_class_matches"]),
                "mean_answer_prob": np.mean(stats["answer_class_probs"]),
                "std_answer_prob": np.std(stats["answer_class_probs"]),
                "mean_target_rank": np.mean(stats["target_class_ranks"]),
                "std_target_rank": np.std(stats["target_class_ranks"]),
                "mean_target_prob": np.mean(stats["target_class_probs"]),
                "std_target_prob": np.std(stats["target_class_probs"]),
            }
    # Print summary stats
    open(log_file_path, "a").write("visualize_llm_classifier_performance: -------- SUMMARY STATISTICS -------- \n")
    for model_key in summary_stats.keys():
        open(log_file_path, "a").write("visualize_llm_classifier_performance: \t model_key == "+str(model_key)+" \n")
        for challenge_key, stats in summary_stats[model_key].items():
            open(log_file_path, "a").write("visualize_llm_classifier_performance: \t\t challenge_key == "+str(challenge_key)+" \n")
            for metric, value in stats.items():
                open(log_file_path, "a").write("visualize_llm_classifier_performance: \t\t\t "+str(metric)+" == "+str(value)+" \n")
    open(log_file_path, "a").write("visualize_llm_classifier_performance: ------------------------------------ \n")

    # Create bar plots to visualize summary stat distros -------------------------------------
    open(log_file_path, "a").write("visualize_llm_classifier_performance: preparing objects for plotting \n")
    model_keys = sorted(list(summary_stats.keys()))
    challenge_keys = sorted(list(summary_stats[model_keys[0]].keys()))
    model_challenge_perf_keys = [mkey+"::"+ckey for mkey in model_keys for ckey in challenge_keys]
    
    avg_answer_match_rate = [summary_stats[mkey][ckey]["mean_answer_class_match_rate"] for mkey in model_keys for ckey in challenge_keys]
    std_answer_match_rate = [summary_stats[mkey][ckey]["std_answer_class_match_rate"] for mkey in model_keys for ckey in challenge_keys]
    avg_answer_probs = [summary_stats[mkey][ckey]["mean_answer_prob"] for mkey in model_keys for ckey in challenge_keys]
    std_answer_probs = [summary_stats[mkey][ckey]["std_answer_prob"] for mkey in model_keys for ckey in challenge_keys]
    avg_target_ranks = [summary_stats[mkey][ckey]["mean_target_rank"] for mkey in model_keys for ckey in challenge_keys]
    std_target_ranks = [summary_stats[mkey][ckey]["std_target_rank"] for mkey in model_keys for ckey in challenge_keys]
    avg_target_probs = [summary_stats[mkey][ckey]["mean_target_prob"] for mkey in model_keys for ckey in challenge_keys]
    std_target_probs = [summary_stats[mkey][ckey]["std_target_prob"] for mkey in model_keys for ckey in challenge_keys]
    
    open(log_file_path, "a").write("visualize_llm_classifier_performance: creating bar plots \n")
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.bar(model_challenge_perf_keys, avg_answer_match_rate, color="blue", alpha=0.7, yerr=std_answer_match_rate, capsize=5, ecolor="grey",)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Answer Class Match Rate (Avg. Correct) ± Std.Dev.")
    plt.title("Comparison of Answer Class Accuracy")
    plt.subplot(1, 2, 2)
    plt.bar(model_challenge_perf_keys, avg_answer_probs, color="slategrey", alpha=0.7, yerr=std_answer_probs, capsize=5, ecolor="grey",)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Mean Answer Class Probability ± Std.Dev.")
    plt.title("Comparison of Answer Class Probabilities")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_barplot_comparing_answer_class_predictions_across_models.png")
    plt.close()

    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.bar(model_challenge_perf_keys, avg_target_ranks, color="orange", alpha=0.7, yerr=std_target_ranks, capsize=5, ecolor="grey",)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Mean Target Class Rank ± Std.Dev.")
    plt.title("Comparison of Target Class Ranks")
    plt.subplot(1, 2, 2)
    plt.bar(model_challenge_perf_keys, avg_target_probs, color="purple", alpha=0.7, yerr=std_target_probs, capsize=5, ecolor="grey",)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Mean Target Class Probability ± Std.Dev.")
    plt.title("Comparison of Target Class Probabilities")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_barplot_comparing_target_class_predictions_across_models.png")
    plt.close()
    
    # Visualization: Box plots for distributions
    open(log_file_path, "a").write("visualize_llm_classifier_performance: creating box plots \n")
    plt.figure(figsize=(15, 12))
    sns.boxplot(data=[aggregate_stats[mkey][ckey]["answer_class_matches"] for mkey in model_keys for ckey in challenge_keys], palette="tab20c")#"rocket")#palette="Set2")
    plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
    plt.title("Distribution of Answer Class Matches Across Models")
    plt.ylabel("Match")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_answer_class_accuracy_across_models.png")
    plt.close()
    plt.figure(figsize=(15, 12))
    sns.boxplot(data=[aggregate_stats[mkey][ckey]["answer_class_probs"] for mkey in model_keys for ckey in challenge_keys], palette="tab20c")#"rocket")#"Set2")
    plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
    plt.title("Distribution of Answer Class Probabilities Across Models")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_answer_class_probabilities_across_models.png")
    plt.close()
    plt.figure(figsize=(15, 12))
    sns.boxplot(data=[aggregate_stats[mkey][ckey]["target_class_ranks"] for mkey in model_keys for ckey in challenge_keys], palette="tab20c")#"rocket")#"Set2")
    plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
    plt.title("Distribution of Target Class Ranks Across Models")
    plt.ylabel("Rank")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_target_class_rankings_across_models.png")
    plt.close()
    plt.figure(figsize=(15, 12))
    sns.boxplot(data=[aggregate_stats[mkey][ckey]["target_class_probs"] for mkey in model_keys for ckey in challenge_keys], palette="tab20c")#"rocket")#"Set2")
    plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
    plt.title("Distribution of Target Class Probabilities Across Models")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_target_class_probabilities_across_models.png")
    plt.close()
    
    open(log_file_path, "a").write("visualize_llm_classifier_performance: STOP \n")
    pass


def visualize_llm_performance(log_file_path, save_dir, results_dict):
    """
    Code written with assistance from GPT-4o
    Visualizes LLM performances
    Parameters:
        results_dict: Dictionary containing performance results for various models. Structured as 
            results_dict[model_key] = { 
                perturbation_key: { 
                    input_sample_category_key: [ 
                        { 
                            ...
                            "results_by_seq_pos": [
                                { 
                                    "perf_on_ans_tokens": { 
                                        "answer token rank": 0, 
                                        "answer token prob": sorted_pred_token_probs[0], 
                                        "target token rank": (sorted_pred_token_ids == target_ids[0,index]).nonzero(as_tuple=True)[0], 
                                        "target token prob": (sorted_pred_token_probs == target_ids[0,index]).nonzero(as_tuple=True)[0], 
                                    }, 
                                    ...
                                }, ... 
                            ] 
                        }, ...
                    ], ... 
                }, ..
            }
    Returns:
        None (Generates plots and prints aggregate statistics).
    """
    open(log_file_path, "a").write("visualize_llm_performance: START \n")
    # Aggregate the statistics ---------------------------------------------------------------
    open(log_file_path, "a").write("visualize_llm_performance: aggregating stats \n")
    aggregate_stats = {}
    for model_key in results_dict.keys():
        # open(log_file_path, "a").write("visualize_llm_performance: DEBUGGING -- model_key == "+str(model_key)+" \n")
        for perturbation_key in results_dict[model_key].keys():
            # open(log_file_path, "a").write("visualize_llm_performance: DEBUGGING -- \tperturbation_key == "+str(perturbation_key)+" \n")
            agg_model_key = str(model_key)+'~~'+str(perturbation_key)
            aggregate_stats[agg_model_key] = dict()
            # open(log_file_path, "a").write("visualize_llm_performance: DEBUGGING -- \results_dict[model_key][perturbation_key] == "+str(results_dict[model_key][perturbation_key])+" \n")
            for challenge_key in results_dict[model_key][perturbation_key].keys(): 
                aggregate_stats[agg_model_key][challenge_key] = {
                    "answer_token_matches": [],
                    "answer_token_probs": [],
                    "target_token_ranks": [],
                    "target_token_probs": [],
                }
                for sample_results in results_dict[model_key][perturbation_key][challenge_key]["sample_level_results"]:
                    for seq_pos_pred_results in sample_results["results_by_seq_pos"]:
                        perf_on_ans_tokens = seq_pos_pred_results["perf_on_ans_tokens"]
                        # Collect relevant statistics
                        answer_token_match = 1. if perf_on_ans_tokens["target token rank"] == 0. else 0.#.item() == 0. else 0.
                        answer_prob = perf_on_ans_tokens["answer token prob"]#.item()  # as float
                        target_rank = perf_on_ans_tokens["target token rank"]#.item()  # as integer
                        target_prob = perf_on_ans_tokens["target token prob"]#.item()  # as float
                        # Append to model-specific aggregate statistics
                        aggregate_stats[agg_model_key][challenge_key]["answer_token_matches"].append(answer_token_match)
                        aggregate_stats[agg_model_key][challenge_key]["answer_token_probs"].append(answer_prob)
                        aggregate_stats[agg_model_key][challenge_key]["target_token_ranks"].append(target_rank)
                        aggregate_stats[agg_model_key][challenge_key]["target_token_probs"].append(target_prob)
    
    # Compute summary statistics -------------------------------------------------------------
    open(log_file_path, "a").write("visualize_llm_performance: summarizing stats \n")
    summary_stats = {}
    for model_key in aggregate_stats.keys():
        summary_stats[model_key] = dict()
        for challenge_key in aggregate_stats[model_key].keys():
            stats = aggregate_stats[model_key][challenge_key]
            summary_stats[model_key][challenge_key] = {
                "mean_answer_token_match_rate": np.mean(stats["answer_token_matches"]),
                "std_answer_token_match_rate": np.std(stats["answer_token_matches"]),
                "mean_answer_prob": np.mean(stats["answer_token_probs"]),
                "std_answer_prob": np.std(stats["answer_token_probs"]),
                "mean_target_rank": np.mean(stats["target_token_ranks"]),
                "std_target_rank": np.std(stats["target_token_ranks"]),
                "mean_target_prob": np.mean(stats["target_token_probs"]),
                "std_target_prob": np.std(stats["target_token_probs"]),
            }
    # Print summary stats
    open(log_file_path, "a").write("visualize_llm_performance: -------- SUMMARY STATISTICS -------- \n")
    for model_key in summary_stats.keys():
        open(log_file_path, "a").write("visualize_llm_performance: \t model_key == "+str(model_key)+" \n")
        for challenge_key, stats in summary_stats[model_key].items():
            open(log_file_path, "a").write("visualize_llm_performance: \t\t challenge_key == "+str(challenge_key)+" \n")
            for metric, value in stats.items():
                open(log_file_path, "a").write("visualize_llm_performance: \t\t\t "+str(metric)+" == "+str(value)+" \n")
    open(log_file_path, "a").write("visualize_llm_performance: ------------------------------------ \n")

    # Create bar plots to visualize summary stat distros -------------------------------------
    open(log_file_path, "a").write("visualize_llm_performance: preparing objects for plotting \n")
    model_keys = sorted(list(summary_stats.keys()))
    challenge_keys = sorted(list(summary_stats[model_keys[0]].keys()))
    model_challenge_perf_keys = [mkey+"::"+ckey for mkey in model_keys for ckey in challenge_keys]
    
    avg_answer_match_rate = [summary_stats[mkey][ckey]["mean_answer_token_match_rate"] for mkey in model_keys for ckey in challenge_keys]
    std_answer_match_rate = [summary_stats[mkey][ckey]["std_answer_token_match_rate"] for mkey in model_keys for ckey in challenge_keys]
    avg_answer_probs = [summary_stats[mkey][ckey]["mean_answer_prob"] for mkey in model_keys for ckey in challenge_keys]
    std_answer_probs = [summary_stats[mkey][ckey]["std_answer_prob"] for mkey in model_keys for ckey in challenge_keys]
    avg_target_ranks = [summary_stats[mkey][ckey]["mean_target_rank"] for mkey in model_keys for ckey in challenge_keys]
    std_target_ranks = [summary_stats[mkey][ckey]["std_target_rank"] for mkey in model_keys for ckey in challenge_keys]
    avg_target_probs = [summary_stats[mkey][ckey]["mean_target_prob"] for mkey in model_keys for ckey in challenge_keys]
    std_target_probs = [summary_stats[mkey][ckey]["std_target_prob"] for mkey in model_keys for ckey in challenge_keys]
    
    open(log_file_path, "a").write("visualize_llm_performance: creating bar plots \n")
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.bar(model_challenge_perf_keys, avg_answer_match_rate, color="blue", alpha=0.7, yerr=std_answer_match_rate, capsize=5, ecolor="grey",)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Answer Token Match Rate (Avg. Correct) ± Std.Dev.")
    plt.title("Comparison of Answer Token Accuracy")
    plt.subplot(1, 2, 2)
    plt.bar(model_challenge_perf_keys, avg_answer_probs, color="slategrey", alpha=0.7, yerr=std_answer_probs, capsize=5, ecolor="grey",)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Mean Answer Token Probability ± Std.Dev.")
    plt.title("Comparison of Answer Token Probabilities")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_barplot_comparing_answer_token_predictions_across_models.png")
    plt.close()

    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.bar(model_challenge_perf_keys, avg_target_ranks, color="orange", alpha=0.7, yerr=std_target_ranks, capsize=5, ecolor="grey",)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Mean Target Token Rank ± Std.Dev.")
    plt.title("Comparison of Target Token Ranks")
    plt.subplot(1, 2, 2)
    plt.bar(model_challenge_perf_keys, avg_target_probs, color="purple", alpha=0.7, yerr=std_target_probs, capsize=5, ecolor="grey",)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Mean Target Token Probability ± Std.Dev.")
    plt.title("Comparison of Target Token Probabilities")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_barplot_comparing_target_token_predictions_across_models.png")
    plt.close()
    
    # Visualization: Box plots for distributions
    open(log_file_path, "a").write("visualize_llm_performance: creating box plots \n")
    plt.figure(figsize=(15, 12))
    sns.boxplot(data=[aggregate_stats[mkey][ckey]["answer_token_matches"] for mkey in model_keys for ckey in challenge_keys], palette="tab20c")#"rocket")#palette="Set2")
    plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
    plt.title("Distribution of Answer Token Matches Across Models")
    plt.ylabel("Match")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_answer_token_accuracy_across_models.png")
    plt.close()
    plt.figure(figsize=(15, 12))
    sns.boxplot(data=[aggregate_stats[mkey][ckey]["answer_token_probs"] for mkey in model_keys for ckey in challenge_keys], palette="tab20c")#"rocket")#"Set2")
    plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
    plt.title("Distribution of Answer Token Probabilities Across Models")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_answer_token_probabilities_across_models.png")
    plt.close()
    plt.figure(figsize=(15, 12))
    sns.boxplot(data=[aggregate_stats[mkey][ckey]["target_token_ranks"] for mkey in model_keys for ckey in challenge_keys], palette="tab20c")#"rocket")#"Set2")
    plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
    plt.title("Distribution of Target Token Ranks Across Models")
    plt.ylabel("Rank")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_target_token_rankings_across_models.png")
    plt.close()
    plt.figure(figsize=(15, 12))
    sns.boxplot(data=[aggregate_stats[mkey][ckey]["target_token_probs"] for mkey in model_keys for ckey in challenge_keys], palette="tab20c")#"rocket")#"Set2")
    plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
    plt.title("Distribution of Target Token Probabilities Across Models")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_target_token_probabilities_across_models.png")
    plt.close()
    
    open(log_file_path, "a").write("compare_llm_performance: STOP \n")
    pass


def compare_llm_performance(log_file_path, save_dir, results_dict):
    """
    Code written with assistance from GPT-4o
    Compares LLM performances across models and perturbations
    Parameters:
        results_dict: Dictionary containing performance results for various models. Structured as 
            results_dict[model_key] = { 
                perturbation_key: { 
                    input_sample_category_key: [ 
                        { 
                            "input_ids": input_ids.cpu(), 
                            "target_ids": target_ids.cpu(), 
                            "pred_probs": [p.cpu() for p in pred_probs], 
                            "pred_ids": pred_ids.cpu(), 
                            "results_by_seq_pos": [
                                { 
                                    "perf_on_ans_tokens": { 
                                        "answer token id": sorted_pred_token_ids[0], 
                                        "answer token rank": 0, 
                                        "answer token prob": sorted_pred_token_probs[0], 
                                        "target token id": target_ids[0,index], 
                                        "target token rank": (sorted_pred_token_ids == target_ids[0,index]).nonzero(as_tuple=True)[0], 
                                        "target token prob": (sorted_pred_token_probs == target_ids[0,index]).nonzero(as_tuple=True)[0], 
                                    }, 
                                    "top_k_token_details": { 
                                        "top_"+str(i)+"_token": { 
                                            "Prob": sorted_pred_token_probs[0, i].item(), 
                                            "Token": sorted_pred_token_ids[0, i], 
                                        } for i in range(top_k) 
                                    } 
                                }, ... 
                            ] 
                        }, ...
                    ], ... 
                }, ..
            }
    Returns:
        None (Generates plots and prints aggregate statistics).
    """
    open(log_file_path, "a").write("compare_llm_performance: START \n")
    # Aggregate the statistics ---------------------------------------------------------------
    open(log_file_path, "a").write("compare_llm_performance: aggregating stats \n")
    aggregate_stats = {}
    for model_key in results_dict.keys():
        for perturbation_key in results_dict[model_key].keys():
            agg_model_key = str(model_key)+'~~'+str(perturbation_key)
            aggregate_stats[agg_model_key] = dict()
            for challenge_key in results_dict[model_key][perturbation_key].keys(): 
                aggregate_stats[agg_model_key][challenge_key] = {
                    "answer_token_matches": [],
                    "answer_token_probs": [],
                    "target_token_ranks": [],
                    "target_token_probs": [],
                }
                for sample_results in results_dict[model_key][perturbation_key][challenge_key]:
                    for seq_pos_pred_results in sample_results["results_by_seq_pos"]:
                        perf_on_ans_tokens = seq_pos_pred_results["perf_on_ans_tokens"]
                        # Collect relevant statistics
                        answer_token_match = 1. if perf_on_ans_tokens["target token rank"] == 0. else 0.#.item() == 0. else 0.
                        answer_prob = perf_on_ans_tokens["answer token prob"]#.item()  # as float
                        target_rank = perf_on_ans_tokens["target token rank"]#.item()  # as integer
                        target_prob = perf_on_ans_tokens["target token prob"]#.item()  # as float
                        # Append to model-specific aggregate statistics
                        aggregate_stats[agg_model_key][challenge_key]["answer_token_matches"].append(answer_token_match)
                        aggregate_stats[agg_model_key][challenge_key]["answer_token_probs"].append(answer_prob)
                        aggregate_stats[agg_model_key][challenge_key]["target_token_ranks"].append(target_rank)
                        aggregate_stats[agg_model_key][challenge_key]["target_token_probs"].append(target_prob)
    
    # Compute summary statistics -------------------------------------------------------------
    open(log_file_path, "a").write("compare_llm_performance: summarizing stats \n")
    summary_stats = {}
    for model_key in aggregate_stats.keys():
        summary_stats[model_key] = dict()
        for challenge_key in aggregate_stats[model_key].keys():
            stats = aggregate_stats[model_key][challenge_key]
            summary_stats[model_key][challenge_key] = {
                "mean_answer_token_match_rate": np.mean(stats["answer_token_matches"]),
                "std_answer_token_match_rate": np.std(stats["answer_token_matches"]),
                "mean_answer_prob": np.mean(stats["answer_token_probs"]),
                "std_answer_prob": np.std(stats["answer_token_probs"]),
                "mean_target_rank": np.mean(stats["target_token_ranks"]),
                "std_target_rank": np.std(stats["target_token_ranks"]),
                "mean_target_prob": np.mean(stats["target_token_probs"]),
                "std_target_prob": np.std(stats["target_token_probs"]),
            }
    # Print summary stats
    open(log_file_path, "a").write("compare_llm_performance: -------- SUMMARY STATISTICS -------- \n")
    for model_key in summary_stats.keys():
        open(log_file_path, "a").write("compare_llm_performance: \t model_key == "+str(model_key)+" \n")
        for challenge_key, stats in summary_stats[model_key].items():
            open(log_file_path, "a").write("compare_llm_performance: \t\t challenge_key == "+str(challenge_key)+" \n")
            for metric, value in stats.items():
                open(log_file_path, "a").write("compare_llm_performance: \t\t\t "+str(metric)+" == "+str(value)+" \n")
    open(log_file_path, "a").write("compare_llm_performance: ------------------------------------ \n")

    # Create bar plots to visualize summary stat distros -------------------------------------
    open(log_file_path, "a").write("compare_llm_performance: preparing objects for plotting \n")
    model_keys = sorted(list(summary_stats.keys()))
    challenge_keys = sorted(list(summary_stats[model_keys[0]].keys()))
    model_challenge_perf_keys = [mkey+"::"+ckey for mkey in model_keys for ckey in challenge_keys]
    
    avg_answer_match_rate = [summary_stats[mkey][ckey]["mean_answer_token_match_rate"] for mkey in model_keys for ckey in challenge_keys]
    std_answer_match_rate = [summary_stats[mkey][ckey]["std_answer_token_match_rate"] for mkey in model_keys for ckey in challenge_keys]
    avg_answer_probs = [summary_stats[mkey][ckey]["mean_answer_prob"] for mkey in model_keys for ckey in challenge_keys]
    std_answer_probs = [summary_stats[mkey][ckey]["std_answer_prob"] for mkey in model_keys for ckey in challenge_keys]
    avg_target_ranks = [summary_stats[mkey][ckey]["mean_target_rank"] for mkey in model_keys for ckey in challenge_keys]
    std_target_ranks = [summary_stats[mkey][ckey]["std_target_rank"] for mkey in model_keys for ckey in challenge_keys]
    avg_target_probs = [summary_stats[mkey][ckey]["mean_target_prob"] for mkey in model_keys for ckey in challenge_keys]
    std_target_probs = [summary_stats[mkey][ckey]["std_target_prob"] for mkey in model_keys for ckey in challenge_keys]
    
    open(log_file_path, "a").write("compare_llm_performance: creating bar plots \n")
    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.bar(model_challenge_perf_keys, avg_answer_match_rate, color="blue", alpha=0.7, yerr=std_answer_match_rate, capsize=5, ecolor="grey",)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Answer Token Match Rate (Avg. Correct) ± Std.Dev.")
    plt.title("Comparison of Answer Token Accuracy")
    plt.subplot(1, 2, 2)
    plt.bar(model_challenge_perf_keys, avg_answer_probs, color="slategrey", alpha=0.7, yerr=std_answer_probs, capsize=5, ecolor="grey",)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Mean Answer Token Probability ± Std.Dev.")
    plt.title("Comparison of Answer Token Probabilities")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_barplot_comparing_answer_token_predictions_across_models.png")
    plt.close()

    plt.figure(figsize=(24, 12))
    plt.subplot(1, 2, 1)
    plt.bar(model_challenge_perf_keys, avg_target_ranks, color="orange", alpha=0.7, yerr=std_target_ranks, capsize=5, ecolor="grey",)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Mean Target Token Rank ± Std.Dev.")
    plt.title("Comparison of Target Token Ranks")
    plt.subplot(1, 2, 2)
    plt.bar(model_challenge_perf_keys, avg_target_probs, color="purple", alpha=0.7, yerr=std_target_probs, capsize=5, ecolor="grey",)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel("Mean Target Token Probability ± Std.Dev.")
    plt.title("Comparison of Target Token Probabilities")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_barplot_comparing_target_token_predictions_across_models.png")
    plt.close()
    
    # Visualization: Box plots for distributions
    open(log_file_path, "a").write("compare_llm_performance: creating box plots \n")
    plt.figure(figsize=(15, 12))
    sns.boxplot(data=[aggregate_stats[mkey][ckey]["answer_token_matches"] for mkey in model_keys for ckey in challenge_keys], palette="tab20c")#"rocket")#palette="Set2")
    plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
    plt.title("Distribution of Answer Token Matches Across Models")
    plt.ylabel("Match")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_answer_token_accuracy_across_models.png")
    plt.close()
    plt.figure(figsize=(15, 12))
    sns.boxplot(data=[aggregate_stats[mkey][ckey]["answer_token_probs"] for mkey in model_keys for ckey in challenge_keys], palette="tab20c")#"rocket")#"Set2")
    plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
    plt.title("Distribution of Answer Token Probabilities Across Models")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_answer_token_probabilities_across_models.png")
    plt.close()
    plt.figure(figsize=(15, 12))
    sns.boxplot(data=[aggregate_stats[mkey][ckey]["target_token_ranks"] for mkey in model_keys for ckey in challenge_keys], palette="tab20c")#"rocket")#"Set2")
    plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
    plt.title("Distribution of Target Token Ranks Across Models")
    plt.ylabel("Rank")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_target_token_rankings_across_models.png")
    plt.close()
    plt.figure(figsize=(15, 12))
    sns.boxplot(data=[aggregate_stats[mkey][ckey]["target_token_probs"] for mkey in model_keys for ckey in challenge_keys], palette="tab20c")#"rocket")#"Set2")
    plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
    plt.title("Distribution of Target Token Probabilities Across Models")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_target_token_probabilities_across_models.png")
    plt.close()
    
    # Statistical significance testing for perturbation comparison
    open(log_file_path, "a").write("compare_llm_performance: ~~~~ <<< Statistical T-Tests >>> ~~~~ \n")
    base_model_key = model_keys[0]  # Assuming first one is base model
    open(log_file_path, "a").write("compare_llm_performance: Base model == "+str(base_model_key)+" \n")
    for model_key in model_keys[1:]:
        for challenge_key in challenge_keys:
            t_amatch, p_amatch = ttest_ind(
                aggregate_stats[base_model_key][challenge_key]["answer_token_matches"], 
                aggregate_stats[model_key][challenge_key]["answer_token_matches"]
            )
            t_aprob, p_aprob = ttest_ind(
                aggregate_stats[base_model_key][challenge_key]["answer_token_probs"], 
                aggregate_stats[model_key][challenge_key]["answer_token_probs"]
            )
            t_trank, p_trank = ttest_ind(
                aggregate_stats[base_model_key][challenge_key]["target_token_ranks"], 
                aggregate_stats[model_key][challenge_key]["target_token_ranks"]
            )
            t_tprob, p_tprob = ttest_ind(
                aggregate_stats[base_model_key][challenge_key]["target_token_probs"], 
                aggregate_stats[model_key][challenge_key]["target_token_probs"]
            )
            open(log_file_path, "a").write("compare_llm_performance: Base model vs "+str(model_key)+" ("+str(challenge_key)+" Answer Token Match): T="+str(t_amatch)+", p="+str(p_amatch)+" \n")#print(f"Base model vs {model_key} ({challenge_key} Answer Token Match): T={t_amatch:.3f}, p={p_amatch:.5f}")
            open(log_file_path, "a").write("compare_llm_performance: Base model vs "+str(model_key)+" ("+str(challenge_key)+" Answer Token Prob): T="+str(t_aprob)+", p="+str(p_aprob)+" \n")#print(f"Base model vs {model_key} ({challenge_key} Answer Token Prob): T={t_aprob:.3f}, p={p_aprob:.5f}")
            open(log_file_path, "a").write("compare_llm_performance: Base model vs "+str(model_key)+" ("+str(challenge_key)+" Target Token Rank): T="+str(t_trank)+", p="+str(p_trank)+" \n")#print(f"Base model vs {model_key} ({challenge_key} Target Token Rank): T={t_trank:.3f}, p={p_trank:.5f}")
            open(log_file_path, "a").write("compare_llm_performance: Base model vs "+str(model_key)+" ("+str(challenge_key)+" Target Token Prob): T="+str(t_tprob)+", p="+str(p_tprob)+" \n")#print(f"Base model vs {model_key} ({challenge_key} Target Token Prob): T={t_tprob:.3f}, p={p_tprob:.5f}")
    open(log_file_path, "a").write("compare_llm_performance: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
    
    open(log_file_path, "a").write("compare_llm_performance: STOP \n")
    pass


def compare_llm_performance_on_kg_tasks(log_file_path, save_dir, results_dict, max_challenges_per_plot=25):
    """
    Code written with assistance from GPT-4o
    Compares LLM performances across models and perturbations
    Parameters:
        results_dict: Dictionary containing performance results for various models. Structured as 
            results_dict[model_key] = { 
                input_sample_category_key: [ 
                    { 
                        "input_ids": input_ids.cpu(), 
                        "target_ids": target_ids.cpu(), 
                        "pred_probs": [p.cpu() for p in pred_probs], 
                        "pred_ids": pred_ids.cpu(), 
                        "results_by_seq_pos": [
                            { 
                                "perf_on_ans_tokens": { 
                                    "answer token id": sorted_pred_token_ids[0], 
                                    "answer token rank": 0, 
                                    "answer token prob": sorted_pred_token_probs[0], 
                                    "target token id": target_ids[0,index], 
                                    "target token rank": (sorted_pred_token_ids == target_ids[0,index]).nonzero(as_tuple=True)[0], 
                                    "target token prob": (sorted_pred_token_probs == target_ids[0,index]).nonzero(as_tuple=True)[0], 
                                }, 
                                "top_k_token_details": { 
                                    "top_"+str(i)+"_token": { 
                                        "Prob": sorted_pred_token_probs[0, i].item(), 
                                        "Token": sorted_pred_token_ids[0, i], 
                                    } for i in range(top_k) 
                                } 
                            }, ... 
                        ] 
                    }, ...
                ], ... 
            }
    Returns:
        None (Generates plots and prints aggregate statistics).
    """
    open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: START \n")
    # Aggregate the statistics ---------------------------------------------------------------
    open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: aggregating stats \n")
    aggregate_stats = {}
    for model_key in results_dict.keys():
        agg_model_key = str(model_key)
        aggregate_stats[agg_model_key] = dict()
        for challenge_key in results_dict[model_key].keys(): 
            aggregate_stats[agg_model_key][challenge_key] = {
                "answer_token_matches": [], "answer_token_probs": [], "target_token_ranks": [], "target_token_probs": [],
            }
            for sample_results in results_dict[model_key][challenge_key]:
                for seq_pos_pred_results in sample_results["results_by_seq_pos"]:
                    perf_on_ans_tokens = seq_pos_pred_results["perf_on_ans_tokens"]
                    # Collect relevant statistics
                    answer_token_match = 1. if perf_on_ans_tokens["target token rank"] == 0. else 0.#.item() == 0. else 0.
                    answer_prob = perf_on_ans_tokens["answer token prob"]#.item()  # as float
                    target_rank = perf_on_ans_tokens["target token rank"]#.item()  # as integer
                    target_prob = perf_on_ans_tokens["target token prob"]#.item()  # as float
                    # Append to model-specific aggregate statistics
                    aggregate_stats[agg_model_key][challenge_key]["answer_token_matches"].append(answer_token_match)
                    aggregate_stats[agg_model_key][challenge_key]["answer_token_probs"].append(answer_prob)
                    aggregate_stats[agg_model_key][challenge_key]["target_token_ranks"].append(target_rank)
                    aggregate_stats[agg_model_key][challenge_key]["target_token_probs"].append(target_prob)
    
    # Compute summary statistics -------------------------------------------------------------
    open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: summarizing stats \n")
    summary_stats = {}
    for model_key in aggregate_stats.keys():
        summary_stats[model_key] = dict()
        for challenge_key in aggregate_stats[model_key].keys():
            stats = aggregate_stats[model_key][challenge_key]
            summary_stats[model_key][challenge_key] = {
                "mean_answer_token_match_rate": np.mean(stats["answer_token_matches"]),
                "std_answer_token_match_rate": np.std(stats["answer_token_matches"]),
                "mean_answer_prob": np.mean(stats["answer_token_probs"]),
                "std_answer_prob": np.std(stats["answer_token_probs"]),
                "mean_target_rank": np.mean(stats["target_token_ranks"]),
                "std_target_rank": np.std(stats["target_token_ranks"]),
                "mean_target_prob": np.mean(stats["target_token_probs"]),
                "std_target_prob": np.std(stats["target_token_probs"]),
            }
    # Print summary stats
    open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: -------- SUMMARY STATISTICS -------- \n")
    for model_key in summary_stats.keys():
        open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: \t model_key == "+str(model_key)+" \n")
        for challenge_key, stats in summary_stats[model_key].items():
            open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: \t\t challenge_key == "+str(challenge_key)+" \n")
            for metric, value in stats.items():
                open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: \t\t\t "+str(metric)+" == "+str(value)+" \n")
    open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: ------------------------------------ \n")

    # Create bar plots to visualize summary stat distros -------------------------------------
    open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: preparing objects for plotting \n")
    model_keys = sorted(list(summary_stats.keys()))
    challenge_keys = sorted(list(summary_stats[model_keys[0]].keys()))
    # model_challenge_perf_keys = [mkey+"::"+ckey for mkey in model_keys for ckey in challenge_keys]
    
    open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: len(challenge_keys) == "+str(len(challenge_keys))+" \n")
    num_challenge_chunks = (len(challenge_keys)//max_challenges_per_plot) + 1*(len(challenge_keys) % max_challenges_per_plot > 0)
    for mkey in model_keys: 
        for chunk in range(num_challenge_chunks):
            chunk_challenge_keys = challenge_keys[chunk*max_challenges_per_plot:(chunk+1)*max_challenges_per_plot]
            model_challenge_perf_keys = [mkey+"::"+ckey for ckey in chunk_challenge_keys]
            avg_answer_match_rate = [summary_stats[mkey][ckey]["mean_answer_token_match_rate"] for ckey in chunk_challenge_keys]
            std_answer_match_rate = [summary_stats[mkey][ckey]["std_answer_token_match_rate"] for ckey in chunk_challenge_keys]
            avg_answer_probs = [summary_stats[mkey][ckey]["mean_answer_prob"] for ckey in chunk_challenge_keys]
            std_answer_probs = [summary_stats[mkey][ckey]["std_answer_prob"] for ckey in chunk_challenge_keys]
            avg_target_ranks = [summary_stats[mkey][ckey]["mean_target_rank"] for ckey in chunk_challenge_keys]
            std_target_ranks = [summary_stats[mkey][ckey]["std_target_rank"] for ckey in chunk_challenge_keys]
            avg_target_probs = [summary_stats[mkey][ckey]["mean_target_prob"] for ckey in chunk_challenge_keys]
            std_target_probs = [summary_stats[mkey][ckey]["std_target_prob"] for ckey in chunk_challenge_keys]
        
            open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: creating bar plots \n")
            plt.figure(figsize=(24, 12))
            plt.subplot(1, 2, 1)
            plt.bar(model_challenge_perf_keys, avg_answer_match_rate, color="blue", alpha=0.7, yerr=std_answer_match_rate, capsize=5, ecolor="grey",)
            plt.xticks(rotation=60, ha='right')
            plt.ylabel("Answer Token Match Rate (Avg. Correct) ± Std.Dev.")
            plt.title("Comparison of Answer Token Accuracy")
            plt.subplot(1, 2, 2)
            plt.bar(model_challenge_perf_keys, avg_answer_probs, color="slategrey", alpha=0.7, yerr=std_answer_probs, capsize=5, ecolor="grey",)
            plt.xticks(rotation=60, ha='right')
            plt.ylabel("Mean Answer Token Probability ± Std.Dev.")
            plt.title("Comparison of Answer Token Probabilities")
            plt.tight_layout()
            plt.savefig(save_dir+os.sep+"vis_barplot_comparing_answer_token_predictions_across_models_model_"+str(mkey)+"_p"+str(chunk)+".png")
            plt.close()

            plt.figure(figsize=(24, 12))
            plt.subplot(1, 2, 1)
            plt.bar(model_challenge_perf_keys, avg_target_ranks, color="orange", alpha=0.7, yerr=std_target_ranks, capsize=5, ecolor="grey",)
            plt.xticks(rotation=60, ha='right')
            plt.ylabel("Mean Target Token Rank ± Std.Dev.")
            plt.title("Comparison of Target Token Ranks")
            plt.subplot(1, 2, 2)
            plt.bar(model_challenge_perf_keys, avg_target_probs, color="purple", alpha=0.7, yerr=std_target_probs, capsize=5, ecolor="grey",)
            plt.xticks(rotation=60, ha='right')
            plt.ylabel("Mean Target Token Probability ± Std.Dev.")
            plt.title("Comparison of Target Token Probabilities")
            plt.tight_layout()
            plt.savefig(save_dir+os.sep+"vis_barplot_comparing_target_token_predictions_across_models_"+str(mkey)+"_p"+str(chunk)+".png")
            plt.close()
        
            # Visualization: Box plots for distributions
            open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: creating box plots \n")
            plt.figure(figsize=(15, 12))
            sns.boxplot(data=[aggregate_stats[mkey][ckey]["answer_token_matches"] for ckey in chunk_challenge_keys], palette="tab20c")#"rocket")#palette="Set2")
            plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
            plt.title("Distribution of Answer Token Matches Across Models")
            plt.ylabel("Match")
            plt.tight_layout()
            plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_answer_token_accuracy_across_models_"+str(mkey)+"_p"+str(chunk)+".png")
            plt.close()
            plt.figure(figsize=(15, 12))
            sns.boxplot(data=[aggregate_stats[mkey][ckey]["answer_token_probs"] for ckey in chunk_challenge_keys], palette="tab20c")#"rocket")#"Set2")
            plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
            plt.title("Distribution of Answer Token Probabilities Across Models")
            plt.ylabel("Probability")
            plt.tight_layout()
            plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_answer_token_probabilities_across_models_"+str(mkey)+"_p"+str(chunk)+".png")
            plt.close()
            plt.figure(figsize=(15, 12))
            sns.boxplot(data=[aggregate_stats[mkey][ckey]["target_token_ranks"] for ckey in chunk_challenge_keys], palette="tab20c")#"rocket")#"Set2")
            plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
            plt.title("Distribution of Target Token Ranks Across Models")
            plt.ylabel("Rank")
            plt.tight_layout()
            plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_target_token_rankings_across_models_"+str(mkey)+"_p"+str(chunk)+".png")
            plt.close()
            plt.figure(figsize=(15, 12))
            sns.boxplot(data=[aggregate_stats[mkey][ckey]["target_token_probs"] for ckey in chunk_challenge_keys], palette="tab20c")#"rocket")#"Set2")
            plt.xticks(range(len(model_challenge_perf_keys)), model_challenge_perf_keys, rotation=60, ha='right')
            plt.title("Distribution of Target Token Probabilities Across Models")
            plt.ylabel("Probability")
            plt.tight_layout()
            plt.savefig(save_dir+os.sep+"vis_boxplot_comparing_target_token_probabilities_across_models_"+str(mkey)+"_p"+str(chunk)+".png")
            plt.close()
    
    if len(model_keys) > 1: 
        # Statistical significance testing for perturbation comparison
        open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: ~~~~ <<< Statistical T-Tests >>> ~~~~ \n")
        base_model_key = model_keys[0]  # Assuming first one is base model
        open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: Base model == "+str(base_model_key)+" \n")
        for model_key in model_keys[1:]:
            for challenge_key in challenge_keys:
                t_amatch, p_amatch = ttest_ind(
                    aggregate_stats[base_model_key][challenge_key]["answer_token_matches"], 
                    aggregate_stats[model_key][challenge_key]["answer_token_matches"]
                )
                t_aprob, p_aprob = ttest_ind(
                    aggregate_stats[base_model_key][challenge_key]["answer_token_probs"], 
                    aggregate_stats[model_key][challenge_key]["answer_token_probs"]
                )
                t_trank, p_trank = ttest_ind(
                    aggregate_stats[base_model_key][challenge_key]["target_token_ranks"], 
                    aggregate_stats[model_key][challenge_key]["target_token_ranks"]
                )
                t_tprob, p_tprob = ttest_ind(
                    aggregate_stats[base_model_key][challenge_key]["target_token_probs"], 
                    aggregate_stats[model_key][challenge_key]["target_token_probs"]
                )
                open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: Base model vs "+str(model_key)+" ("+str(challenge_key)+" Answer Token Match): T="+str(t_amatch)+", p="+str(p_amatch)+" \n")#print(f"Base model vs {model_key} ({challenge_key} Answer Token Match): T={t_amatch:.3f}, p={p_amatch:.5f}")
                open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: Base model vs "+str(model_key)+" ("+str(challenge_key)+" Answer Token Prob): T="+str(t_aprob)+", p="+str(p_aprob)+" \n")#print(f"Base model vs {model_key} ({challenge_key} Answer Token Prob): T={t_aprob:.3f}, p={p_aprob:.5f}")
                open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: Base model vs "+str(model_key)+" ("+str(challenge_key)+" Target Token Rank): T="+str(t_trank)+", p="+str(p_trank)+" \n")#print(f"Base model vs {model_key} ({challenge_key} Target Token Rank): T={t_trank:.3f}, p={p_trank:.5f}")
                open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: Base model vs "+str(model_key)+" ("+str(challenge_key)+" Target Token Prob): T="+str(t_tprob)+", p="+str(p_tprob)+" \n")#print(f"Base model vs {model_key} ({challenge_key} Target Token Prob): T={t_tprob:.3f}, p={p_tprob:.5f}")
        open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
    
    open(log_file_path, "a").write("compare_llm_performance_on_kg_tasks: STOP \n")
    pass
