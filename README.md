# STRES Repository

## Overview

This repository contains the code used to perform STRES Test analyses on text-time-series foundation models. The structure of the repository can be summarized as follows: 

Repository Root Folder: 
 - .gitignore: designates specific naming conventions for exclusively local files/folders
 - .gitmodules: identifies external code repository dependencies
 - Edge Lists (query_relation_edges_metadata/*.txt): these files identify edge types that can be used in detailed evaluations of foundation models (see Step 3 of Usage Instructions)
 - dedupe_lines_to_txt.py: used for making some of the Edge List files
 - evaluate submodule: contains python scripts for evaluating foundation model performance along with a text file (run_commands_for_evaluate_module.txt) detailing the commands used to run the scripts and a dedicated metadata file (eval_edge_groupings_rawEdgeFocussedV02252026.csv) to support some of the scripts. Additionally, this submodule includes a folder (testcases) for testing the functionality of the evaluation scripts.
 - train submodule: contains a python script for finetuning foundation models (train_OpenTSLM_usgs50hzT4096.py) on seismic data and a document detailing commands for running the script (run_commands_for_train_module.txt).
 - usgs_data submodule: contains python scripts for curating and validating STRES datasets along with a text file (run_commands_for_usgs_data_module.txt) detailing the commands used to run the scripts.
 - utils submodule: contains python scripts that support the functionality of the other submodules.

While some of the python files contain elements of metadata - e.g. 'usgs_data/usgs_LowPassShortWindowStationSubset_LatLonRegionSplit_seismic_data.py' designates a specific event ID for exclusion due to invalid data associated with the event and 'utils/general_usgs_info_pretext.py' contains several possible model prompts informed by publicly available information - most of the metadata used in the repository can be found in the files of the query_relation_edges_metadata folder and 'evaluate/eval_edge_groupings_rawEdgeFocussedV02252026.csv'.


## Installation Instructions

Step 1: 

``` git clone --recurse-submodules <url_for_this_repo> ```

Step 2: 

``` pip install -r requirements.txt --target=~/usgs-multimodal-models/usgsMM_env ```

Step 3: 

Replace the definition of the helper function OpenTSLMFlamingo.pad_and_apply_batch.pad_time_series (located at OpenTSLM/src/opentslm/model/llm/OpenTSLMFlamingo.py in the original repo as of the time of writing) with the definition given in ~/usgs-multimodal-models/STRES_OpenTSLMFlamingo_pad_time_series_helper.py

## Usage Instructions

Below we outline the steps for using this repository to test foundation models on the USGS information presented in the publicly available MultiSeismo dataset. Each step refers to the general python scripts needed for specific functionality; bash scripts with names that share common elements with the names of the python scripts are included in the repository that illustrate how to run specific analyses, though they may need to be adapted for a particular compute environment.

Step 1: Curate and vet STRES data set(s)
<ol type="i">
  <li>Generate a partition of seismic events, following the process outlined in usgs_data/partition_and_plot_usgs_data_by_lat_an_lon_v02162026.py</li>
  <li>Curate STRES data set(s) using the usgs_data/usgs_LowPassShortWindowStationSubset_LatLonRegionSplit_seismic_data.py script</li>
  <li>Check/inspect curated STRES data set(s) using usgs_data/print_dataset_prompts.py, usgs_data/check_event_kg_overlaps.py, and in partcular usgs_data/check_val_train_query_overlaps.py (which is used to ensure information does not leak between events located on partition boundaries).</li>
</ol>

Step 2: Finetune pretrained models on USGS STRES data set(s) via train/train_OpenTSLM_usgs50hzT4096.py

Step 3: Evaluate finetuned models
<ol type="i">
  <li>Generate finetuned model predictions on the STRES validation set(s), following the process outlined in evaluate/compOuts_of_usgs_finetuned_models_v02232026.py</li>
  <li>Evaluate the quality of the finetuned models' predictions using the process outlined in evaluate/aggregate_compOutsOfUSGSFinetunedModelsV0223_FocusedOverviewPlots_v02282026.py</li>
  <li>Compare the performance of models when time series information is ablated vs when it's available using the process outlined in evaluate/compareExtensively_ablated_vs_non_ablated_aggregated_compOutsOfUSGSFinetunedModels0223FocOverPlots0228_v03022026.py</li>
</ol>