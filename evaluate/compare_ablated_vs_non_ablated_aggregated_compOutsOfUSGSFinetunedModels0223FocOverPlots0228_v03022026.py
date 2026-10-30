import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRETRAINING_STAGES = ["tsqa-flamingo", "m4-flamingo", "har-flamingo", "sleep-flamingo", "ecg-flamingo", ]
DIFF_COLUMNS = ["best_normalized_Damerau_Levenshtein_dist", "best_normalized_hamming_dist", "best_euclidean_dist_avgd_across_str", ]
DIFF_COLUMNS_TO_TITLE_STR_MAP = {
    "best_normalized_Damerau_Levenshtein_dist": "Best Normalized DLD", 
    "best_normalized_hamming_dist": "Best Normalized HD", 
    "best_euclidean_dist_avgd_across_str": "Best Euclidean Distance (Avg. Over String)", 
}
FILTER_COLUMN = "query_relation_edge"#'C'
CSV_FILE_NAME_TAIL = "_eval_edge_groupings_rawEdgeFocussedV02252026_csv/aggregated_samples.csv"
MATCH_COLS = [#"pkl_path", #"A", 
    "query_relation_edge", #"C", 
    "sample_id", #"D", 
    "base_model_name", #"I", 
    "num_stations", #"I2", 
    "kg_size", #"J", 
    "batch_size", #"K", 
    "base_model_reformatted", #"L", 
    "pretraining_stage", #"M", 
    "fold_id", #"N", # "input_text", #"O", 
]
FILTER_COL_TARGET_VALUES = [
    "caused shaking level of 1 in ", 
    "caused shaking level of 2 in ", 
    "caused shaking level of 3 in ", 
    "caused shaking level of 4 in ", 
    "caused shaking level of 5 in ", 
    "experienced shaking level of 1 caused by ", 
    "experienced shaking level of 2 caused by ", 
    "experienced shaking level of 3 caused by ", 
    "experienced shaking level of 4 caused by ", 
    "experienced shaking level of 5 caused by ", 
    "exposed an estimated population to earthquake shaking of around ", 
    "exposed the population to shaking with a rated level of ", 
    "had a point of origin from the nearest major population that was approximately ", 
    "had approximate magnitude of ", 
    "has Event ID: ", 
    "has ISO value of ", 
    "has a W-phase Moment Tensor with a Catalog element value of ", 
    "has a W-phase Moment Tensor with a Contributor element value of ", 
    "has a W-phase Moment Tensor with a Data Source element value of ", 
    "has a W-phase Moment Tensor with a Depth element value of ", 
    "has a W-phase Moment Tensor with a Half Duration element value of ", 
    "has a W-phase Moment Tensor with a Magnitude element value of ", 
    "has a W-phase Moment Tensor with a Moment element value of ", 
    "has a W-phase Moment Tensor with a Percent DC element value of ", 
    "has an origin characterized by a Azimuthal Gap with a value of ", 
    "has an origin characterized by a Catalog with a value of ", 
    "has an origin characterized by a Contributor with a value of ", 
    "has an origin characterized by a Depth with a value of ", 
    "has an origin characterized by a FERegion with a value of ", 
    "has an origin characterized by a Location Source with a value of ", 
    "has an origin characterized by a Location with a value of ", 
    "has an origin characterized by a Magnitude Source with a value of ", 
    "has an origin characterized by a Magnitude with a value of ", 
    "has an origin characterized by a Minimum Distance with a value of ", 
    "has an origin characterized by a Number of Phases with a value of ", 
    "has an origin characterized by a Number of Stations with a value of ", 
    "has an origin characterized by a Origin Time with a value of ", 
    "has an origin characterized by a Review Status with a value of ", 
    "has an origin characterized by a Travel Time Residual with a value of ", 
    "is referred to as the ", 
    "occured in an area with estimated population density (per square km) of ", 
    "occured in the country named ", 
    "occured in the region referred to as the ", 
    "occured nearest to major population of ", 
    "originated from GPS coordinates ", 
    "originated from a depth of ", 
    "triggered a shaking related fatalities alert level of ", 
    "triggered an economical loss alert level of ", 
    "was the closest major population to ", 
]



if __name__ == "__main__":
    # set up primary arguments
    print("__MAIN__: START ", flush=True)
    save_dir = os.path.expanduser("~/results/compare_ablated_vs_non_ablated_aggregated_compOutsOfUSGSFinetunedModels0223FocOverPlots0228_v03022026")
    file1_root_folder = "~/results/aggregate_compOutsOfUSGSFinetunedModelsV0223_FocusedOverviewPlots_v02282026/"
    file2_root_folder = "~/results/aggregate_compOutsOfUSGSFinetunedModelsV0223_FocusedOverviewPlots_v02282026_ABLATED/"

    for diff_column in DIFF_COLUMNS: # diff_column = "best_match_ratio"#'H'
        print("__MAIN__: processing results for diff_column == ", diff_column, flush=True)
        file1 = os.path.expanduser(file1_root_folder+diff_column+CSV_FILE_NAME_TAIL)#"~/results/aggregate_compOutsWithValOverlapFilteringAndTSAblations_ICML2026_FocusedOverviewPlots_v01302026/best_match_ratio_eval_edge_groupings_rawEdgeFocussedV01162026_csv/aggregated_samples.csv") 
        file2 = os.path.expanduser(file2_root_folder+diff_column+CSV_FILE_NAME_TAIL)#"~/results/aggregate_compOutsWithValOverlapFilteringAndTSAblations_ICML2026_FocusedOverviewPlots_v01302026_ABLATED/best_match_ratio_eval_edge_groupings_rawEdgeFocussedV01162026_csv/aggregated_samples.csv")

        # Read in csvs
        print("__MAIN__: \t reading in csv files ", flush=True)
        os.makedirs(save_dir, exist_ok=True)
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        for filter_col_target_value in FILTER_COL_TARGET_VALUES:
            print("__MAIN__: \t\t now applying filter_col_target_value == ", filter_col_target_value, " to analyses", flush=True)
            # Filter by filter_column
            df1_f = df1[df1[FILTER_COLUMN] == filter_col_target_value].copy() 
            df2_f = df2[df2[FILTER_COLUMN] == filter_col_target_value].copy()

            # Build match key across MATCH_COLS
            for d in (df1_f, df2_f): 
                d["Key"] = d[MATCH_COLS].astype(str).apply(lambda r: "|".join(r), axis=1)

            # Merge and compute diff_column difference, Add any df2_f columns you want to compare side-by-side.
            df2_cols_to_add = ["Key", diff_column, "pred_text"]
            merged = df1_f.merge(df2_f[df2_cols_to_add], on="Key", how="inner", suffixes=("_1", "_2"))
            merged[diff_column+"_diff"] = merged[diff_column+"_1"] - merged[diff_column+"_2"]

            # Save result
            save_str = filter_col_target_value.strip().replace(' ','_')
            merged.to_csv(save_dir+os.sep+"matched_differences_for_edge_value_"+save_str+".csv", index=False)

            # Plot grouped diffs with error bars
            group_cols = ["base_model_name", "pretraining_stage"]
            stats = (merged.groupby(group_cols)[diff_column + "_diff"].agg(["mean", "std", "count"]).reset_index())
            stage_order = {stage: idx for idx, stage in enumerate(PRETRAINING_STAGES)}
            stats["_pretraining_stage_order"] = (stats["pretraining_stage"].map(stage_order).fillna(len(stage_order)).astype(int))
            stats = stats.sort_values(["base_model_name", "_pretraining_stage_order", "pretraining_stage"])
            stats["sem"] = stats["std"] / stats["count"].pow(0.5)
            stats["group_label"] = (stats["base_model_name"].astype(str)+" | "+stats["pretraining_stage"].astype(str))
            stats = stats.drop(columns="_pretraining_stage_order")

            fig, ax = plt.subplots(figsize=(8, 8))
            x = np.arange(len(stats))
            ax.errorbar(x, stats["mean"], yerr=stats["sem"], fmt="none", ecolor="black", elinewidth=1, capsize=3,)
            ax.scatter(x, stats["mean"], s=40, zorder=3)
            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_title(DIFF_COLUMNS_TO_TITLE_STR_MAP[diff_column]+" Difference By Model/Stage/Batch Size")
            ax.set_ylabel("Average Difference in "+str(DIFF_COLUMNS_TO_TITLE_STR_MAP[diff_column])+" ± SEM)")
            ax.set_xlabel("Base Model | Pretraining Stage")
            ax.set_xticks(x)
            ax.set_xticklabels(stats["group_label"], rotation=45, ha="right", fontsize=12)
            fig.tight_layout()
            fig.savefig(save_dir+os.sep+diff_column+"_diff_by_group_"+save_str+".png", dpi=200, bbox_inches="tight",)#save_dir+os.sep+"best_match_ratio_diff_by_group_"+save_str+".png", dpi=200, bbox_inches="tight",)
            plt.close(fig)
    
    print("__MAIN__: DONE !!! ", flush=True)
    pass

"""
The aggregation script produces a csv structured as: 
A           B           C                   D           E           F           G                   H                   I               I2                J       K           L                       M                   N       O           P           Q           R S T U V W X Y Z
pkl_path	model_id	query_relation_edge	sample_id	pred_len	targ_len	best_num_matches	best_match_ratio	base_model_name	num_stations	kg_size	batch_size	base_model_reformatted	pretraining_stage	fold_id	input_text	target_text	pred_text

With the values of query_relation_edge resembling: 

has an origin characterized by a Number of Stations with a value of 
occured in an area with estimated population density (per square km) of 
was located 612.3 km (380.4 mi) SW of 
is located 310 km (192.6 mi) SE of 
...

These values can be used to construct FILTER_COL_TARGET_VALUES
"""
