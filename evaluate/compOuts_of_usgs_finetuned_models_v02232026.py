import os
import gc
import sys
import time
import argparse
import pickle as pkl
import csv

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset
import h5py
import numpy as np

from pathlib import Path
from huggingface_hub import login as hfh_login
from transformers import AutoTokenizer

main_script_file_path = Path(__file__)
absolute_main_script_file_path = main_script_file_path.resolve()
repo_directory = absolute_main_script_file_path.parent.parent
sys.path.append(str(repo_directory))
opentslm_src_directory = repo_directory / "OpenTSLM" / "src"
sys.path.append(str(opentslm_src_directory))
from usgs_data.text_time_series_dataset import get_text_time_series_hdf5_data_loader
from utils.general_utils import set_all_seeds, load_model_from_checkpoint
from evaluate.eval_utils import evaluate_model_kg_predictions_on_kg_text_timeseries_data#, compare_llm_performance_on_kg_tasks
from OpenTSLM.src.model.llm.OpenTSLM import OpenTSLM as OpenTSLM_ModelClass

set_all_seeds(seed=42)
TIME_SERIES_SAMPLE_RT = 50#hz, 
INCLUDE_GEN_KNOWLEDGE = True
CONDENSE_TS_DESCRIPTIONS = True
PATCH_SIZE = 4
VAL_TRAIN_OVERLAP_CSV_PATH = os.path.expanduser("~/results/check_val_train_query_overlaps_v02172026/val_train_query_overlaps.csv")

EVAL_DATA_ROOT_FOLDERS = [ 
    "usgs_LP50hzSW4096StSub4Kg4_seismic_data_LatLonRegSplit_v02162026", 
    "usgs_LP50hzSW4096StSub4Kg10_seismic_data_LatLonRegSplit_v02162026", 
    "usgs_LP50hzSW4096StSub16Kg4_seismic_data_LatLonRegSplit_v02162026", 
    "usgs_LP50hzSW4096StSub16Kg10_seismic_data_LatLonRegSplit_v02162026", 
]
EVAL_DATA_CACHE_FOLDER_NAME = "all_sensor_recorded_events"
LOG_FILE_NAME = 'logs.txt'
BATCH_SIZE_AND_DATE_STR_PAIRS_BY_DATA_ROOT_FOLDER = {
    "usgs_LP50hzSW4096StSub4Kg4_seismic_data_LatLonRegSplit_v02162026": [{'batch_size':4, 'date_str': "02182026"},], 
    "usgs_LP50hzSW4096StSub4Kg10_seismic_data_LatLonRegSplit_v02162026": [{'batch_size':4, 'date_str': "02182026"},], 
    "usgs_LP50hzSW4096StSub16Kg4_seismic_data_LatLonRegSplit_v02162026": [{'batch_size':1, 'date_str': "02182026"},], 
    "usgs_LP50hzSW4096StSub16Kg10_seismic_data_LatLonRegSplit_v02162026": [{'batch_size':1, 'date_str': "02182026"},], 
}

def get_finetned_model_dir(dir_template, data_root_folder_name, batch_size, date_str): 
    REFORMATED_DATA_ROOT_FOLDER_NAMES = {
        "usgs_LP50hzSW4096StSub4Kg4_seismic_data_LatLonRegSplit_v02162026": "Sta4Kg4", 
        "usgs_LP50hzSW4096StSub4Kg10_seismic_data_LatLonRegSplit_v02162026": "Sta4Kg10", 
        "usgs_LP50hzSW4096StSub16Kg4_seismic_data_LatLonRegSplit_v02162026": "Sta16Kg4", 
        "usgs_LP50hzSW4096StSub16Kg10_seismic_data_LatLonRegSplit_v02162026": "Sta16Kg10", 
    }
    model_dir = dir_template.replace("_FOO_DATA", "_"+REFORMATED_DATA_ROOT_FOLDER_NAMES[data_root_folder_name])
    model_dir = model_dir.replace("BatchSizeFOO_", "BatchSize"+str(batch_size)+"_")
    model_dir = model_dir.replace("_vFOO", "_v"+str(date_str))
    return model_dir

# MAX_CHALLENGES_PER_PLOT = 25
CURRICULUM_SEISMIC_DATA_VAL_FILE_NAME = "val_data.h5"
HUGGING_FACE_MODEL_TO_EOS_STR_MAP = {
    'OpenTSLM/llama-3.2-1b-': '<|end_of_text|>', # 'meta-llama/Llama-3.2-1B::<|end_of_text|>': 128001, # 'meta-llama/Llama-3.2-1B::<|begin_of_text|>': 128000, 
    'OpenTSLM/llama-3.2-3b-': '<|end_of_text|>', # 'meta-llama/Llama-3.2-3B::<|end_of_text|>': 128001, # 'meta-llama/Llama-3.2-3B::<|begin_of_text|>': 128000, 
    'OpenTSLM/gemma-3-270m-': '<eos>', # 'google/gemma-3-270m::<eos>': 1, # 'google/gemma-3-270m::<bos>': 2, # 'google/gemma-3-270m::<unk>': 3, # 'google/gemma-3-270m::<start_of_image>': 255999, # 'google/gemma-3-270m::<end_of_image>': 256000, # 'google/gemma-3-270m::<image_soft_token>': 262144, 
    'OpenTSLM/gemma-3-1b-pt-': '<eos>', # 'google/gemma-3-1b-pt::<eos>': 1, # 'google/gemma-3-1b-pt::<bos>': 2, # 'google/gemma-3-1b-pt::<unk>': 3, # 'google/gemma-3-1b-pt::<start_of_image>': 255999, # 'google/gemma-3-1b-pt::<end_of_image>': 256000, # 'google/gemma-3-1b-pt::<image_soft_token>': 262144, 
}
HUGGINGFACE_OPENTSLM_FLAMINGO_CHKPTS = [
    # gemma-3-270m
    "OpenTSLM/gemma-3-270m-ecg-flamingo", 
    "OpenTSLM/gemma-3-270m-sleep-flamingo", 
    "OpenTSLM/gemma-3-270m-har-flamingo", 
    "OpenTSLM/gemma-3-270m-m4-flamingo", 
    "OpenTSLM/gemma-3-270m-tsqa-flamingo", 
    # gemma-3-1b-pt
    "OpenTSLM/gemma-3-1b-pt-ecg-flamingo", 
    "OpenTSLM/gemma-3-1b-pt-sleep-flamingo", 
    "OpenTSLM/gemma-3-1b-pt-har-flamingo", 
    "OpenTSLM/gemma-3-1b-pt-m4-flamingo", 
    "OpenTSLM/gemma-3-1b-pt-tsqa-flamingo", 
    # llama-3.2-1b
    "OpenTSLM/llama-3.2-1b-ecg-flamingo", 
    "OpenTSLM/llama-3.2-1b-sleep-flamingo", 
    "OpenTSLM/llama-3.2-1b-har-flamingo", 
    "OpenTSLM/llama-3.2-1b-m4-flamingo", 
    "OpenTSLM/llama-3.2-1b-tsqa-flamingo", 
    # llama-3.2-3b
    "OpenTSLM/llama-3.2-3b-ecg-flamingo", 
    "OpenTSLM/llama-3.2-3b-sleep-flamingo", 
    "OpenTSLM/llama-3.2-3b-har-flamingo", 
    "OpenTSLM/llama-3.2-3b-m4-flamingo", 
    "OpenTSLM/llama-3.2-3b-tsqa-flamingo", 
]
ALL_POSSIBLE_PRETRAINING_STAGE_IDS = [pts_id for pts_id in range(len(sorted(['ecg','sleep','har','m4','tsqa',])))]
FINETUNE_CHKPT_FILE_PATH_END = "checkpoints/best_model.pt"

def decode_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_text(value.item())
        return "".join(decode_text(elem) for elem in value.ravel().tolist())
    if isinstance(value, (list, tuple)):
        return "".join(decode_text(elem) for elem in value)
    return str(value)

def normalize_text(text):
    return " ".join(str(text).split())


def load_overlap_filters(dataset_name, fold_id, log_file_path, csv_path=VAL_TRAIN_OVERLAP_CSV_PATH):
    overlap_sets = {
        "sample_ids": set(), #"event_ids": set(),
        "queries": set(),
        "answers": set(),
    }
    with open(log_file_path, "a") as log_file: 
        if not os.path.exists(csv_path):
            log_file.write("load_overlap_filters: WARNING - overlap CSV not found at "+str(csv_path)+" \n")
            return overlap_sets
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("dataset") != dataset_name or row.get("fold") != fold_id:
                    continue
                overlap_sets["sample_ids"].add(row.get("val_sample_id", ""))
                #overlap_sets["event_ids"].add(row.get("val_event_id", ""))
                overlap_sets["queries"].add(normalize_text(row.get("val_query", "")))
                overlap_sets["answers"].add(normalize_text(row.get("val_answer", "")))
        log_file.write(
            "load_overlap_filters: overlap filter sizes - sample_ids="+str(len(overlap_sets["sample_ids"]))+#", event_ids="+str(len(overlap_sets["event_ids"]))+
            ", queries="+str(len(overlap_sets["queries"]))+
            ", answers="+str(len(overlap_sets["answers"]))+" \n"
        )
    return overlap_sets


def get_overlap_filtered_indices(hdf5_path, overlap_sets, log_file_path):
    keep_indices = []
    drop_count = 0
    with h5py.File(hdf5_path, 'r') as h5f:
        num_samples = h5f.attrs.get('num_samples', 0)
        for idx in range(int(num_samples)):
            sample_key = "sample_"+str(idx)
            grp = h5f.get(sample_key)
            if grp is None:
                continue
            event_id = str(grp.attrs.get("event_id", ""))
            kg_grp = grp.get("text_based_knowedge_graph_sample")
            query = ""
            answer = ""
            if kg_grp is not None:
                query = decode_text(kg_grp["query"][()])
                answer = decode_text(kg_grp["answer"][()])
            norm_query = normalize_text(query)
            norm_answer = normalize_text(answer)
            if (
                sample_key in overlap_sets["sample_ids"]#or event_id in overlap_sets["event_ids"]
                or (norm_query in overlap_sets["queries"] and norm_answer in overlap_sets["answers"])
            ):
                drop_count += 1
                continue
            keep_indices.append(idx)
    open(log_file_path, "a").write(
        "get_overlap_filtered_indices: overlap filtering dropped "+str(drop_count)+
        " samples; remaining "+str(len(keep_indices))+" \n"
    )
    return keep_indices


class FilteredDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.POST_PROMPT_END = base_dataset.POST_PROMPT_END
        self.hdf5_path = base_dataset.hdf5_path
        self.file = base_dataset.file
        pass

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

    def get_hdf5_index(self, idx):
        return self.indices[idx]


class AblatedTimeSeriesDataset(Dataset):
    def __init__(self, base_dataset, noise_mean=0.0, noise_std=1.0):
        self.base_dataset = base_dataset
        self.POST_PROMPT_END = base_dataset.POST_PROMPT_END
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.rng = np.random.default_rng()
        self.hdf5_path = getattr(base_dataset, "hdf5_path", None)
        self.file = getattr(base_dataset, "file", None)
        pass

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        ablated = dict(sample)
        ablated_time_series = []
        for ts in sample["time_series"]:
            noise = self.rng.normal(self.noise_mean, self.noise_std, size=ts.shape)
            if hasattr(ts, "dtype"):
                noise = noise.astype(ts.dtype, copy=False)
            ablated_time_series.append(noise)
        ablated["time_series"] = ablated_time_series
        if "time_series_text" in sample:
            ablated["time_series_text"] = ["UNKNOWN" for _ in sample["time_series_text"]]
        return ablated

    def get_hdf5_index(self, idx):
        if hasattr(self.base_dataset, "get_hdf5_index"):
            return self.base_dataset.get_hdf5_index(idx)
        return idx


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_compare_outputs_eval(
    save_dir, model_finetune_folder, fold_id, model_chkpt_base_dir, eval_data_base_dir, base_llm_id, max_patches, 
    openTSLM_model_type="OpenTSLMFlamingo", gen_eval_result_key_start='', hf_cache_dir=None
):
    # configure environment ------------------------------------------------------------------
    device = get_device()
    eval_data_path = eval_data_base_dir + os.sep + fold_id + os.sep + 'val_data.h5'
    finetuned_model_name = model_finetune_folder.replace('/', '_') + "_" + fold_id
    curr_eval_save_dir = save_dir + os.sep + finetuned_model_name
    if not os.path.exists(curr_eval_save_dir):
        os.mkdir(curr_eval_save_dir)
    curr_log_file_path = curr_eval_save_dir + os.sep + 'logs_for_eval.txt'

    # perform evaluation ---------------------------------------------------------------------
    with open(curr_log_file_path, "a") as log_file: 
        log_file.write("run_compare_outputs_eval: START \n")
        log_file.write("run_compare_outputs_eval: eval_data_path == "+str(eval_data_path)+" \n")
        log_file.write("run_compare_outputs_eval: curr_eval_save_dir == "+str(curr_eval_save_dir)+" \n")
        # define stat-aggregating object(s) ... 
        gen_eval_res_key = gen_eval_result_key_start+finetuned_model_name
        generative_eval_results = {gen_eval_res_key: None, }

        # define tokenizer -------------------------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(base_llm_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        # load the current data --------------------------------------------------------------
        log_file.write("run_compare_outputs_eval: Loading Data \n")
        _, eval_dataset = get_text_time_series_hdf5_data_loader(
            eval_data_path, curr_log_file_path, tokenizer.eos_token, batch_size=1, patch_size=PATCH_SIZE, 
            time_series_sample_rt=TIME_SERIES_SAMPLE_RT, include_gen_knowledge=INCLUDE_GEN_KNOWLEDGE, 
            condense_ts_descriptions=CONDENSE_TS_DESCRIPTIONS, 
        )
        log_file.write("run_compare_outputs_eval: total num loaded samples == "+str(len(eval_dataset))+" \n")
        dataset_name = os.path.basename(os.path.dirname(eval_data_base_dir))
        overlap_sets = load_overlap_filters(dataset_name, fold_id, curr_log_file_path)
        keep_indices = get_overlap_filtered_indices(eval_data_path, overlap_sets, curr_log_file_path)
        filtered_eval_dataset = eval_dataset
        if len(keep_indices) != len(eval_dataset):
            filtered_eval_dataset = FilteredDataset(eval_dataset, keep_indices)
        log_file.write("run_compare_outputs_eval: total num filtered samples == "+str(len(filtered_eval_dataset))+" \n")

        # load model -------------------------------------------------------------------------
        model = None
        log_file.write("run_compare_outputs_eval: Loading model from model_finetune_folder == "+str(model_finetune_folder)+" \n")
        if model_finetune_folder in HUGGINGFACE_OPENTSLM_FLAMINGO_CHKPTS: # case where original OpenTSLM model is to be evaluated
            assert hf_cache_dir is not None
            os.environ["HF_HOME"] = hf_cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
            os.environ["HF_HUB_CACHE"] = hf_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
            model = OpenTSLM_ModelClass.load_pretrained(chkpt_path, cache_dir=hf_cache_dir).to(device)
        else: # case where seismic-finetuned model is to be loaded from chkpt
            model = load_model_from_checkpoint(
                model_chkpt_base_dir+os.sep+model_finetune_folder+os.sep+fold_id+os.sep+FINETUNE_CHKPT_FILE_PATH_END, 
                curr_log_file_path, 
                base_llm_id, 
                device, 
                max_patches=max_patches, 
                model_type=openTSLM_model_type
            )
        log_file.write("run_compare_outputs_eval: model loaded onto device == "+str(model.device)+" \n")

        # evaluate model ---------------------------------------------------------------------
        log_file.write("run_compare_outputs_eval: Calling evaluate_model_kg_predictions_on_kg_text_timeseries_data for model \n")
        generative_eval_results[gen_eval_res_key] = evaluate_model_kg_predictions_on_kg_text_timeseries_data(
            model, tokenizer, filtered_eval_dataset, filtered_eval_dataset.POST_PROMPT_END, curr_log_file_path, max_patches, #max_samps_per_category=50, 
            max_new_tokens_per_samp=30, top_k=10, #universal_prompt_pretext=None, 
        )
        # Note: we no longer call compare_llm_performance_on_kg_tasks (as of Feb 2026) as this method primarily tracks token-level metrics, rather than character-level
        # open(curr_log_file_path, "a").write("run_compare_outputs_eval: Calling compare_llm_performance_on_kg_tasks \n")
        # compare_llm_performance_on_kg_tasks(curr_log_file_path, curr_eval_save_dir, generative_eval_results, max_challenges_per_plot=MAX_CHALLENGES_PER_PLOT)
        log_file.write("run_compare_outputs_eval: Saving generative_eval_results \n")
        with open(curr_eval_save_dir+os.sep+"generative_eval_results.pkl", 'wb') as outfile:
            pkl.dump(generative_eval_results, outfile)

    # evaluate model on time-series ablated dataset ------------------------------------------
    ablated_eval_save_dir = curr_eval_save_dir + os.sep + "ts_ablated"
    os.makedirs(ablated_eval_save_dir, exist_ok=True)
    ablated_log_file_path = ablated_eval_save_dir + os.sep + 'logs_for_eval_ts_ablated.txt'
    with open(ablated_log_file_path, "a") as ablated_log_file: 
        ablated_log_file.write("run_compare_outputs_eval: START (ts_ablated) \n")
        ablated_gen_eval_res_key = gen_eval_res_key + "_ts_ablated"
        ablated_eval_results = {ablated_gen_eval_res_key: None, }
        ablated_eval_dataset = AblatedTimeSeriesDataset(filtered_eval_dataset)
        ablated_log_file.write("run_compare_outputs_eval: Calling evaluate_model_kg_predictions_on_kg_text_timeseries_data (ts_ablated) \n")
        ablated_eval_results[ablated_gen_eval_res_key] = evaluate_model_kg_predictions_on_kg_text_timeseries_data(
            model, tokenizer, ablated_eval_dataset, ablated_eval_dataset.POST_PROMPT_END, ablated_log_file_path, max_patches,
            max_new_tokens_per_samp=30, top_k=10,
        )
        # Note: we no longer call compare_llm_performance_on_kg_tasks (as of Feb 2026) as this method primarily tracks token-level metrics, rather than character-level
        # open(ablated_log_file_path, "a").write("run_compare_outputs_eval: Calling compare_llm_performance_on_kg_tasks (ts_ablated) \n")
        # compare_llm_performance_on_kg_tasks(ablated_log_file_path, ablated_eval_save_dir, ablated_eval_results, max_challenges_per_plot=MAX_CHALLENGES_PER_PLOT)
        ablated_log_file.write("run_compare_outputs_eval: Saving generative_eval_results (ts_ablated) \n")
        with open(ablated_eval_save_dir+os.sep+"generative_eval_results_ts_ablated.pkl", 'wb') as outfile:
            pkl.dump(ablated_eval_results, outfile)
        ablated_log_file.write("run_compare_outputs_eval: STOP (ts_ablated) \n")

    open(curr_log_file_path, "a").write("run_compare_outputs_eval: STOP \n")
    pass



if __name__ == "__main__": 
    # read in arguments ##########################################################################################
    parser = argparse.ArgumentParser(description="USGS Seismic Data Curriculum Learning for OpenTSLM Models.")
    parser.add_argument("--eval_data_root_dir", type=str, required=True, help="Base directory of cached pre-processed data")
    parser.add_argument("--model", type=str, choices=["OpenTSLMSP", "OpenTSLMFlamingo"], required=True, help="Model type to train",)
    parser.add_argument("--openTSLM_base_model_str", type=str, choices=["OpenTSLM/gemma-3-270m-", "OpenTSLM/gemma-3-1b-pt-", "OpenTSLM/llama-3.2-1b-", "OpenTSLM/llama-3.2-3b-", ], required=True, help="OpenTSLM pretrained model path start",)
    #parser.add_argument("--batch_size", type=int, required=True, help="Batch size used to finetune model to be evaluated")
    #parser.add_argument("--date_str", type=str, required=True, help="Date string (MMDDYYY) of when model to be evaluated was last finetuned")
    parser.add_argument("--save_root_dir", type=str, default="None", help="Directory where training outputs should be saved")
    parser.add_argument("--seismic_finetune_base_dir", type=str, default="None", help="Path to where model checkpoints are saved")
    parser.add_argument("--max_patches", type=str, default=None, help="Maximum number of patches to use for time series positional encoding")
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B", help="LLM model ID for OpenTSLMFlamingo (e.g., 'google/medgemma-2b', 'meta-llama/Llama-3.2-1B')",)
    parser.add_argument("--log_file_path", type=str, default="None", help="Path to where logs should be saved")
    parser.add_argument("--max_folds", type=int, default=None, help="Optional limit on number of folds to evaluate per dataset")
    parser.add_argument("--hf_cache_dir", type=str, default="~/huggingface_cache", help="Local Hugging Face cache root to load models from")

    args = parser.parse_args()

    task_ID = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    print("(task"+str(task_ID)+") __MAIN__: START", flush=True)
    eval_data_root_dir = args.eval_data_root_dir
    save_root_dir = args.save_root_dir
    max_patches = int(args.max_patches)
    #batch_size = args.batch_size#bs_ds_p['batch_size']
    #date_str = args.date_str#bs_ds_p['date_str']
    time.sleep(task_ID%10 + 1) # stagger tasks to avoid race conditions

    # set up environment #########################################################################################
    print("(task"+str(task_ID)+") __MAIN__: setting up environment", flush=True)
    print("(task"+str(task_ID)+") __MAIN__: repo_directory == ", repo_directory, flush=True)
    print("(task"+str(task_ID)+") __MAIN__: opentslm_src_directory == ", opentslm_src_directory, flush=True)
    # huggingface_hub_login_credential_filepath = str(repo_directory)+os.sep+"huggingface_hub_login_credential.txt"
    # login_cred = open(huggingface_hub_login_credential_filepath, 'r').read().strip()
    # hfh_login(login_cred)
    # assert torch.cuda.is_available()

    # begin evaluations ##########################################################################################

    # determine which dataset (and fold/repeat of that dataset) will be used for evaluation in the current task
    dataset_folders = sorted(EVAL_DATA_ROOT_FOLDERS)
    dataset_fold_pretrainStage_combos = []
    for dset_f in dataset_folders: 
        path_to_folds = eval_data_root_dir+os.sep+dset_f+os.sep+EVAL_DATA_CACHE_FOLDER_NAME
        fold_folder_names = sorted([x for x in os.listdir(path_to_folds) if "fold_" in x])
        for ffn in fold_folder_names:
            for pts_id in ALL_POSSIBLE_PRETRAINING_STAGE_IDS: 
                dataset_fold_pretrainStage_combos.append((dset_f, ffn, pts_id))
    assert len(dataset_fold_pretrainStage_combos) == 80 # sanity check
    curr_task_dset_fold_and_pretrainStage_combo = dataset_fold_pretrainStage_combos[task_ID-1]
    edrf = curr_task_dset_fold_and_pretrainStage_combo[0]
    fold_id = curr_task_dset_fold_and_pretrainStage_combo[1]
    pretrainStage_id = curr_task_dset_fold_and_pretrainStage_combo[2]

    # for edrf in EVAL_DATA_ROOT_FOLDERS: 
    eval_data_dir = eval_data_root_dir+os.sep+edrf+os.sep+EVAL_DATA_CACHE_FOLDER_NAME
    print("(task"+str(task_ID)+") __MAIN__: now conducting evaluations using data in eval_data_dir == ", eval_data_dir, flush=True)

    # for bs_ds_p in BATCH_SIZE_AND_DATE_STR_PAIRS_BY_DATA_ROOT_FOLDER[edrf]: 
    assert len(BATCH_SIZE_AND_DATE_STR_PAIRS_BY_DATA_ROOT_FOLDER[edrf]) == 1
    bs_ds_p = BATCH_SIZE_AND_DATE_STR_PAIRS_BY_DATA_ROOT_FOLDER[edrf][0]
    batch_size = bs_ds_p['batch_size']
    date_str = bs_ds_p['date_str']
    save_dir = save_root_dir+os.sep+edrf+"_batchSize"+str(batch_size)+"_v"+str(date_str)
    os.makedirs(save_dir, exist_ok=True)
    print("(task"+str(task_ID)+") __MAIN__: \t save_dir ", save_dir, flush=True)
    args.log_file_path = save_dir+os.sep+"logs.txt"
    model_chkpt_base_dir = get_finetned_model_dir(args.seismic_finetune_base_dir, edrf, batch_size, date_str)
    model_finetune_folder = sorted([x for x in os.listdir(model_chkpt_base_dir) if "logs" not in x])[pretrainStage_id]#task_ID-1]
    gen_eval_result_key_start = "seismic_finetuned_"
    print("(task"+str(task_ID)+") __MAIN__: \t model_finetune_folder == ", model_finetune_folder, flush=True)
    print("(task"+str(task_ID)+") __MAIN__: \t gen_eval_result_key_start == ", gen_eval_result_key_start, flush=True)
    
    # eval_data_fold_names = sorted(os.listdir(eval_data_dir)) # ['fold_1',] FOR DEBUGGING PURPOSES
    # if args.max_folds is not None:
    #     eval_data_fold_names = eval_data_fold_names[:args.max_folds]
    # print("(task"+str(task_ID)+") __MAIN__: \t eval_data_fold_names == ", eval_data_fold_names, flush=True)
    # for fold_id in eval_data_fold_names:
    print("(task"+str(task_ID)+") __MAIN__: \t now calling run_compare_outputs_eval with fold_id == "+str(fold_id)+" ~~~~~~~~~~~~~~~~~~~~~~~~~~ ", flush=True)
    # kick off process on available device
    run_compare_outputs_eval(
        save_dir, 
        model_finetune_folder, 
        fold_id, 
        model_chkpt_base_dir, 
        eval_data_dir, 
        args.llm_id, #base_llm_id, 
        max_patches, 
        openTSLM_model_type=args.model, #"OpenTSLMFlamingo"???, 
        gen_eval_result_key_start=gen_eval_result_key_start, 
        hf_cache_dir=args.hf_cache_dir
    )

    print("(task"+str(task_ID)+") __MAIN__: DONE!!! ", flush=True)
    pass
