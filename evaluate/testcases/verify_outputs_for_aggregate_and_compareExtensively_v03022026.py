#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = [
    "best_normalized_Damerau_Levenshtein_dist",
    "best_normalized_hamming_dist",
    "best_euclidean_dist_avgd_across_str",
]
EDGES = [
    "has ISO value of ",
    "caused shaking level of 1 in ",
]
KEY_COLS = [
    "query_relation_edge",
    "sample_id",
    "base_model_name",
    "num_stations",
    "kg_size",
    "batch_size",
    "base_model_reformatted",
    "pretraining_stage",
    "fold_id",
]
SAMPLE_COMPARE_COLS = KEY_COLS + [
    "model_id",
    "best_normalized_Damerau_Levenshtein_dist",
    "best_normalized_hamming_dist",
    "best_euclidean_dist_avgd_across_str",
]
SUMMARY_COMPARE_COLS = [
    "base_model_name",
    "pretraining_stage",
    "num_stations",
    "kg_size",
    "batch_size",
    "fold_id",
    "query_relation_edge",
    "model_id",
    "avg_best_normalized_Damerau_Levenshtein_dist",
    "std_best_normalized_Damerau_Levenshtein_dist",
    "median_best_normalized_Damerau_Levenshtein_dist",
    "max_best_normalized_Damerau_Levenshtein_dist",
    "min_best_normalized_Damerau_Levenshtein_dist",
    "avg_best_normalized_hamming_dist",
    "std_best_normalized_hamming_dist",
    "median_best_normalized_hamming_dist",
    "max_best_normalized_hamming_dist",
    "min_best_normalized_hamming_dist",
    "avg_best_euclidean_dist_avgd_across_str",
    "std_best_euclidean_dist_avgd_across_str",
    "median_best_euclidean_dist_avgd_across_str",
    "max_best_euclidean_dist_avgd_across_str",
    "min_best_euclidean_dist_avgd_across_str",
]


def _assert_close_df(actual, expected, key_cols, numeric_cols, atol=1e-9):
    a = actual.copy()
    e = expected.copy()
    for col in key_cols:
        a[col] = a[col].astype(str)
        e[col] = e[col].astype(str)
    merged = a.merge(e, on=key_cols, how="outer", suffixes=("_actual", "_expected"), indicator=True)
    if not (merged["_merge"] == "both").all():
        missing_actual = merged[merged["_merge"] == "right_only"][key_cols]
        missing_expected = merged[merged["_merge"] == "left_only"][key_cols]
        raise AssertionError(
            "Row-key mismatch. Missing in actual:\n"
            + missing_actual.to_string(index=False)
            + "\nMissing in expected:\n"
            + missing_expected.to_string(index=False)
        )

    for col in numeric_cols:
        left = pd.to_numeric(merged[f"{col}_actual"], errors="coerce")
        right = pd.to_numeric(merged[f"{col}_expected"], errors="coerce")
        mismatch = ~np.isclose(left, right, atol=atol, equal_nan=True)
        if mismatch.any():
            bad = merged.loc[mismatch, key_cols + [f"{col}_actual", f"{col}_expected"]]
            raise AssertionError(f"Numeric mismatch in {col}:\n{bad.to_string(index=False)}")


def _verify_aggregate_outputs(scenario, aggregate_root, expected_root):
    expected_samples = pd.read_csv(expected_root / f"expected_{scenario}_aggregated_samples.csv")
    expected_summary = pd.read_csv(expected_root / f"expected_{scenario}_aggregated_summary.csv")

    for metric in METRICS:
        metric_dir = aggregate_root / f"{metric}_eval_edge_groupings_rawEdgeFocussedV02252026_csv"
        samples_path = metric_dir / "aggregated_samples.csv"
        summary_path = metric_dir / "aggregated_summary.csv"
        plot_bar = metric_dir / "overview_plots" / f"{metric}_bar.png"
        plot_heatmap = metric_dir / "overview_plots" / f"{metric}_edge_group_x_model_heatmap.png"

        if not samples_path.is_file():
            raise FileNotFoundError(f"Missing aggregated_samples: {samples_path}")
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing aggregated_summary: {summary_path}")
        if not plot_bar.is_file():
            raise FileNotFoundError(f"Missing bar plot: {plot_bar}")
        if not plot_heatmap.is_file():
            raise FileNotFoundError(f"Missing heatmap plot: {plot_heatmap}")

        actual_samples = pd.read_csv(samples_path)
        actual_summary = pd.read_csv(summary_path)

        _assert_close_df(
            actual_samples[SAMPLE_COMPARE_COLS],
            expected_samples[SAMPLE_COMPARE_COLS],
            key_cols=KEY_COLS + ["model_id"],
            numeric_cols=[
                "best_normalized_Damerau_Levenshtein_dist",
                "best_normalized_hamming_dist",
                "best_euclidean_dist_avgd_across_str",
            ],
        )

        _assert_close_df(
            actual_summary[SUMMARY_COMPARE_COLS],
            expected_summary[SUMMARY_COMPARE_COLS],
            key_cols=[
                "base_model_name",
                "pretraining_stage",
                "num_stations",
                "kg_size",
                "batch_size",
                "fold_id",
                "query_relation_edge",
                "model_id",
            ],
            numeric_cols=[c for c in SUMMARY_COMPARE_COLS if c.startswith(("avg_", "std_", "median_", "max_", "min_"))],
        )


def _verify_compare_outputs(compare_root, expected_root):
    expected_stats = pd.read_csv(expected_root / "expected_compareExtensively_summary_stats.csv")

    for metric in METRICS:
        for edge in EDGES:
            save_str = edge.strip().replace(" ", "_")
            stats_path = compare_root / f"{metric}_summary_stats_for_edge_value_{save_str}.csv"
            if not stats_path.is_file():
                raise FileNotFoundError(f"Missing compare summary stats: {stats_path}")

            actual = pd.read_csv(stats_path)
            expected = expected_stats[
                (expected_stats["diff_column"] == metric)
                & (expected_stats["filter_col_target_value"] == edge)
            ].copy()

            _assert_close_df(
                actual[[
                    "diff_column",
                    "filter_col_target_value",
                    "variant_name",
                    "metric_col",
                    "mean",
                    "std",
                    "count",
                    "median",
                    "q1",
                    "q2",
                    "q3",
                ]],
                expected[[
                    "diff_column",
                    "filter_col_target_value",
                    "variant_name",
                    "metric_col",
                    "mean",
                    "std",
                    "count",
                    "median",
                    "q1",
                    "q2",
                    "q3",
                ]],
                key_cols=["diff_column", "filter_col_target_value", "variant_name", "metric_col"],
                numeric_cols=["mean", "std", "count", "median", "q1", "q2", "q3"],
            )

            required_plots = [
                compare_root / f"{metric}_diff_by_group_{save_str}_merged.png",
                compare_root / f"{metric}_1_vs_2_by_group_{save_str}_merged.png",
                compare_root / f"{metric}_boxplot_{metric}_diff_{save_str}_merged.png",
            ]
            for plot_path in required_plots:
                if not plot_path.is_file():
                    raise FileNotFoundError(f"Missing compare plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify aggregate + compareExtensively outputs against synthetic testcase expectations.")
    parser.add_argument(
        "--expected-root",
        type=str,
        default="~/results/test_compOuts_expected_v03022026",
    )
    parser.add_argument(
        "--aggregate-non-root",
        type=str,
        default="~/results/test03022026_aggregate_compOutsOfUSGSFinetunedModelsV0223_FocusedOverviewPlots_v02282026",
    )
    parser.add_argument(
        "--aggregate-ablated-root",
        type=str,
        default="~/results/test03022026_aggregate_compOutsOfUSGSFinetunedModelsV0223_FocusedOverviewPlots_v02282026_ABLATED",
    )
    parser.add_argument(
        "--compare-root",
        type=str,
        default="~/results/test03022026_compareExtensively_ablated_vs_non_ablated_aggregated_compOutsOfUSGSFinetunedModels0223FocOverPlots0228_v03022026",
    )
    args = parser.parse_args()

    expected_root = Path(args.expected_root).expanduser()
    aggregate_non_root = Path(args.aggregate_non_root).expanduser()
    aggregate_ablated_root = Path(args.aggregate_ablated_root).expanduser()
    compare_root = Path(args.compare_root).expanduser()

    _verify_aggregate_outputs("non_ablated", aggregate_non_root, expected_root)
    _verify_aggregate_outputs("ablated", aggregate_ablated_root, expected_root)
    _verify_compare_outputs(compare_root, expected_root)

    print("Verification passed for aggregate and compareExtensively outputs.")


if __name__ == "__main__":
    main()
