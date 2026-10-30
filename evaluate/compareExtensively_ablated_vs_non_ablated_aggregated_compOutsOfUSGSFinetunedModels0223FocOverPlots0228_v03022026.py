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

def set_symlog_yscale(ax, values, min_linthresh=1e-8):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    nonzero_abs = np.abs(vals[vals != 0])
    if nonzero_abs.size == 0:
        linthresh = 1.0
    else:
        # Keep a small linear region around 0 and log-like behavior away from 0.
        linthresh = max(nonzero_abs.min() / 10.0, min_linthresh)
    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
    pass


if __name__ == "__main__":
    # set up primary arguments
    print("__MAIN__: START ", flush=True)
    save_dir = os.path.expanduser("~/results/compareExtensively_ablated_vs_non_ablated_aggregated_compOutsOfUSGSFinetunedModels0223FocOverPlots0228_v03022026")
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
            metric_cols = [diff_column+"_1", diff_column+"_2", diff_column+"_diff"]
            for metric_col in metric_cols:
                merged[metric_col] = pd.to_numeric(merged[metric_col], errors="coerce")

            finite_mask = np.isfinite(merged[metric_cols]).all(axis=1)
            merged_drop = merged.loc[finite_mask].copy()

            merged_replace = merged.copy()
            max_values = merged[metric_cols].replace([np.inf, -np.inf], np.nan).max(skipna=True)
            for metric_col in metric_cols:
                replacement_value = max_values[metric_col]
                if pd.isna(replacement_value):
                    replacement_value = 0.0
                non_finite_mask = ~np.isfinite(merged_replace[metric_col])
                merged_replace.loc[non_finite_mask, metric_col] = replacement_value

            save_str = filter_col_target_value.strip().replace(' ','_')
            merged.to_csv(save_dir+os.sep+"matched_differences_for_edge_value_"+save_str+".csv", index=False)
            merged_drop.to_csv(save_dir+os.sep+"matched_differences_for_edge_value_"+save_str+"_NONFINITE_DROPPED.csv", index=False)
            merged_replace.to_csv(save_dir+os.sep+"matched_differences_for_edge_value_"+save_str+"_NONFINITE_CAPPED.csv", index=False)

            stats_rows = []
            merged_variants = [
                ("merged", merged),
                ("merged_drop", merged_drop),
                ("merged_replace", merged_replace),
            ]
            for variant_name, variant_df in merged_variants:
                for metric_col in metric_cols:
                    series = pd.to_numeric(variant_df[metric_col], errors="coerce")
                    if "_drop" in variant_name or "_replace" in variant_name:
                        series = series[np.isfinite(series)] # finite_series = series[np.isfinite(series)]
                    stats_rows.append({
                        "diff_column": diff_column,
                        "filter_col_target_value": filter_col_target_value,
                        "variant_name": variant_name,
                        "metric_col": metric_col,
                        "mean": series.mean(),#finite_series.mean(),
                        "std": series.std(),#finite_series.std(),
                        "count": int(series.count()),#int(finite_series.count()),
                        "median": series.median(),#finite_series.median(),
                        "q1": series.quantile(0.25),#finite_series.quantile(0.25),
                        "q2": series.quantile(0.50),#finite_series.quantile(0.50),
                        "q3": series.quantile(0.75),#finite_series.quantile(0.75),
                    })
            stats_df = pd.DataFrame(stats_rows)
            stats_df.to_csv(
                save_dir+os.sep+diff_column+"_summary_stats_for_edge_value_"+save_str+".csv",
                index=False,
            )
            print("__MAIN__: \t\t summary statistics for ", diff_column, " and edge ", filter_col_target_value, ":\n", stats_df, flush=True)

            stage_order = {stage: idx for idx, stage in enumerate(PRETRAINING_STAGES)}
            group_cols = ["base_model_name", "pretraining_stage"]
            for variant_name, variant_df in merged_variants:
                # Save result
                variant_save_str = save_str+"_"+variant_name
                variant_df.to_csv(
                    save_dir+os.sep+"matched_differences_for_edge_value_"+variant_save_str+".csv",
                    index=False,
                )

                # Plot grouped diffs with error bars
                stats = (variant_df.groupby(group_cols)[diff_column + "_diff"].agg(["mean", "std", "count"]).reset_index())
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
                ax.set_title(DIFF_COLUMNS_TO_TITLE_STR_MAP[diff_column]+" Difference By Model/Stage/Batch Size ("+variant_name+")")
                ax.set_ylabel("Average Difference in "+str(DIFF_COLUMNS_TO_TITLE_STR_MAP[diff_column])+" \u00b1 SEM)")
                ax.set_xlabel("Base Model | Pretraining Stage")
                ax.set_xticks(x)
                ax.set_xticklabels(stats["group_label"], rotation=45, ha="right", fontsize=12)
                if "euclidean" in diff_column.lower():
                    set_symlog_yscale(ax, stats["mean"])
                fig.tight_layout()
                fig.savefig(
                    save_dir+os.sep+diff_column+"_diff_by_group_"+variant_save_str+".png",
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close(fig)

                # Plot grouped means +/- SEM for diff_column_1 and diff_column_2
                stats_1 = variant_df.groupby(group_cols)[diff_column+"_1"].agg(["mean", "std", "count"]).reset_index()
                stats_2 = variant_df.groupby(group_cols)[diff_column+"_2"].agg(["mean", "std", "count"]).reset_index()
                stats_1 = stats_1.rename(columns={"mean": "mean_1", "std": "std_1", "count": "count_1"})
                stats_2 = stats_2.rename(columns={"mean": "mean_2", "std": "std_2", "count": "count_2"})
                stats_12 = stats_1.merge(stats_2, on=group_cols, how="inner")
                stats_12["_pretraining_stage_order"] = (
                    stats_12["pretraining_stage"].map(stage_order).fillna(len(stage_order)).astype(int)
                )
                stats_12 = stats_12.sort_values(["base_model_name", "_pretraining_stage_order", "pretraining_stage"])
                stats_12["sem_1"] = stats_12["std_1"] / stats_12["count_1"].pow(0.5)
                stats_12["sem_2"] = stats_12["std_2"] / stats_12["count_2"].pow(0.5)
                stats_12["group_label"] = (
                    stats_12["base_model_name"].astype(str)+" | "+stats_12["pretraining_stage"].astype(str)
                )

                fig, ax = plt.subplots(figsize=(8, 8))
                x = np.arange(len(stats_12))
                width = 0.35
                ax.errorbar(
                    x - width/2,
                    stats_12["mean_1"],
                    yerr=stats_12["sem_1"],
                    fmt="o",
                    ecolor="tab:blue",
                    color="tab:blue",
                    elinewidth=1,
                    capsize=3,
                    label=diff_column+"_1",
                )
                ax.errorbar(
                    x + width/2,
                    stats_12["mean_2"],
                    yerr=stats_12["sem_2"],
                    fmt="o",
                    ecolor="tab:orange",
                    color="tab:orange",
                    elinewidth=1,
                    capsize=3,
                    label=diff_column+"_2",
                )
                ax.set_title(DIFF_COLUMNS_TO_TITLE_STR_MAP[diff_column]+" Values By Model/Stage/Batch Size ("+variant_name+")")
                ax.set_ylabel("Average Value (\u00b1 SEM)")
                ax.set_xlabel("Base Model | Pretraining Stage")
                ax.set_xticks(x)
                ax.set_xticklabels(stats_12["group_label"], rotation=45, ha="right", fontsize=12)
                if "euclidean" in diff_column.lower():
                    set_symlog_yscale(ax, np.concatenate([stats_12["mean_1"].values, stats_12["mean_2"].values]))
                ax.legend()
                fig.tight_layout()
                fig.savefig(
                    save_dir+os.sep+diff_column+"_1_vs_2_by_group_"+variant_save_str+".png",
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close(fig)

                # Box-and-whisker plots grouped the same way as the means +/- SEM plots.
                grouped = variant_df[group_cols + metric_cols].copy()
                for metric_col in metric_cols:
                    grouped[metric_col] = pd.to_numeric(grouped[metric_col], errors="coerce")

                group_df = grouped[group_cols].drop_duplicates()
                group_df["_pretraining_stage_order"] = (
                    group_df["pretraining_stage"].map(stage_order).fillna(len(stage_order)).astype(int)
                )
                group_df = group_df.sort_values(["base_model_name", "_pretraining_stage_order", "pretraining_stage"])
                group_df["group_label"] = (
                    group_df["base_model_name"].astype(str)+" | "+group_df["pretraining_stage"].astype(str)
                )
                ordered_groups = list(group_df[group_cols].itertuples(index=False, name=None))
                group_labels = group_df["group_label"].tolist()

                if len(ordered_groups) > 0:
                    x = np.arange(len(ordered_groups))
                    colors = ["tab:blue", "tab:orange", "tab:green"]

                    for metric_idx, metric_col in enumerate(metric_cols):
                        metric_box_data = []
                        metric_positions = []
                        metric_q0 = []
                        metric_q100 = []
                        for group_idx, (base_model_name, pretraining_stage) in enumerate(ordered_groups):
                            group_mask = (
                                (grouped["base_model_name"] == base_model_name) &
                                (grouped["pretraining_stage"] == pretraining_stage)
                            )
                            vals = grouped.loc[group_mask, metric_col]
                            if "_drop" in metric_col or "_replace" in metric_col: 
                                vals = vals[np.isfinite(vals)]
                            if len(vals) > 0:
                                metric_box_data.append(vals.values)
                                metric_positions.append(group_idx)
                                metric_q0.append(vals.quantile(0.0))
                                metric_q100.append(vals.quantile(1.0))

                        if len(metric_box_data) == 0:
                            continue

                        fig, ax = plt.subplots(figsize=(max(8, len(ordered_groups) * 0.7), 8))
                        color = colors[metric_idx % len(colors)]
                        is_euclidean = "euclidean" in diff_column.lower()
                        boxes = ax.boxplot(
                            metric_box_data,
                            positions=metric_positions,
                            widths=0.6,
                            patch_artist=True,
                            showfliers=not is_euclidean,
                        )
                        for patch in boxes["boxes"]:
                            patch.set_facecolor(color)
                            patch.set_alpha(0.5)
                            patch.set_edgecolor(color)
                        for median in boxes["medians"]:
                            median.set_color("black")

                        ax.set_title(
                            DIFF_COLUMNS_TO_TITLE_STR_MAP[diff_column] +
                            " Distribution By Model/Stage/Batch Size (" + variant_name + ", " + metric_col + ")"
                        )
                        ax.set_ylabel("Value")
                        ax.set_xlabel("Base Model | Pretraining Stage")
                        ax.set_xticks(x)
                        ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=12)
                        if is_euclidean:# and len(metric_q0) > 0 and len(metric_q100) > 0:
                            set_symlog_yscale(ax, grouped[metric_col].values)
                            # y_min = float(np.min(metric_q0))
                            # y_max = float(np.max(metric_q100))
                            # if np.isfinite(y_min) and np.isfinite(y_max):
                            #     if y_min == y_max:
                            #         pad = 0.05 * (abs(y_min) + 1.0)
                            #         ax.set_ylim(y_min - pad, y_max + pad)
                            #     else:
                            #         ax.set_ylim(y_min, y_max)
                        fig.tight_layout()
                        fig.savefig(
                            save_dir+os.sep+diff_column+"_boxplot_"+metric_col+"_"+variant_save_str+".png",
                            dpi=200,
                            bbox_inches="tight",
                        )
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
