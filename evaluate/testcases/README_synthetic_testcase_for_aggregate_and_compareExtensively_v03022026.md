# Synthetic testcase for aggregate+compareExtensively (03/02/2026)

This testcase validates:

- `aggregate_compOutsOfUSGSFinetunedModelsV0223_FocusedOverviewPlots_v02282026.py`
- `compareExtensively_ablated_vs_non_ablated_aggregated_compOutsOfUSGSFinetunedModels0223FocOverPlots0228_v03022026.py`

It generates synthetic `generative_eval_results.pkl` and `generative_eval_results_ts_ablated.pkl` files in the same folder/filename pattern used by `compOuts_of_usgs_finetuned_models_v02232026.py` outputs.

## 1) Generate synthetic PKLs + expected values

```bash
python3 ~/usgs-multimodal-models/evaluate/testcases/generate_synthetic_testcase_for_aggregate_and_compareExtensively_v03022026.py
```

Defaults:

- non-ablated PKLs: `~/results/test_compOuts_non_ablated_v03022026`
- ablated PKLs: `~/results/test_compOuts_ablated_v03022026`
- expected stats: `~/results/test_compOuts_expected_v03022026`

## 2) Run aggregate (non-ablated)

```bash
python3 ~/usgs-multimodal-models/evaluate/aggregate_compOutsOfUSGSFinetunedModelsV0223_FocusedOverviewPlots_v02282026.py \
  --results-dir ~/results/test_compOuts_non_ablated_v03022026 \
  --csv-root-save-folder ~/results/test03022026_aggregate_compOutsOfUSGSFinetunedModelsV0223_FocusedOverviewPlots_v02282026 \
  --plot-subdir overview_plots \
  --edge-group-configs-root-dir ~/usgs-multimodal-models/evaluate \
  --glob "compOuts_of_usgs_finetuned_models_v02232026_TSLM*/usgs_LP50hzSW4096StSub*Kg*_seismic_data_LatLonRegSplit_v02162026_batchSize*_v*/OpenTSLM_*_fold_*/generative_eval_results.pkl" \
  --decode-keep-special-tokens \
  --plot-group-by base_model_name,pretraining_stage,num_stations,kg_size,batch_size \
  --plot-bar-by-edge-group \
  --edge-group-col edge_group
```

## 3) Run aggregate (ablated)

```bash
python3 ~/usgs-multimodal-models/evaluate/aggregate_compOutsOfUSGSFinetunedModelsV0223_FocusedOverviewPlots_v02282026.py \
  --results-dir ~/results/test_compOuts_ablated_v03022026 \
  --csv-root-save-folder ~/results/test03022026_aggregate_compOutsOfUSGSFinetunedModelsV0223_FocusedOverviewPlots_v02282026_ABLATED \
  --plot-subdir overview_plots \
  --edge-group-configs-root-dir ~/usgs-multimodal-models/evaluate \
  --glob "compOuts_of_usgs_finetuned_models_v02232026_TSLM*/usgs_LP50hzSW4096StSub*Kg*_seismic_data_LatLonRegSplit_v02162026_batchSize*_v*/OpenTSLM_*_fold_*/ts_ablated/generative_eval_results_ts_ablated.pkl" \
  --decode-keep-special-tokens \
  --plot-group-by base_model_name,pretraining_stage,num_stations,kg_size,batch_size \
  --plot-bar-by-edge-group \
  --edge-group-col edge_group
```

## 4) Run compareExtensively (no code changes needed)

```bash 
sbatch run_compareExtensively_ablated_vs_non_ablated_aggregated_compOutsOfUSGSFinetunedModels0223FocOverPlots0228_v03022026_TEST.sh
```

<!-- ```bash
python ~/usgs-multimodal-models/evaluate/compareExtensively_ablated_vs_non_ablated_aggregated_compOutsOfUSGSFinetunedModels0223FocOverPlots0228_v03022026.py
``` -->

## 5) Verify stats and expected plot artifacts

```bash
python ~/usgs-multimodal-models/evaluate/testcases/verify_outputs_for_aggregate_and_compareExtensively_v03022026.py
```

If verification passes, all of the following are confirmed:

- aggregate sample-level metrics are correct for all three distance metrics,
- aggregate grouped summary stats are correct,
- compareExtensively summary statistics match expected merged/delta math,
- expected key plot files are present.

## Note

This testcase uses the `google/gemma-3-270m` tokenizer. Ensure it is available in your Hugging Face cache or accessible in your runtime environment.
