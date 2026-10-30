#!/usr/bin/env python3
import argparse
import os
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

main_script_file_path = Path(__file__)
absolute_main_script_file_path = main_script_file_path.resolve()
repo_directory = absolute_main_script_file_path.parent.parent.parent
sys.path.append(str(repo_directory))
from evaluate.eval_utils import (  # noqa: E402
    _char_level_normalized_Damerau_Levenshtein_dist,
    _char_level_normalized_hamming_dist,
    _euclidean_dist_avgd_across_str,
)

EDGE_TARGETS = {
    "has ISO value of ": "ISO 100",
    "caused shaking level of 1 in ": "Foo City",
}

PREDS_BY_SCENARIO_STAGE_EDGE = {
    "non_ablated": {
        "ecg-flamingo": {
            "has ISO value of ": ["ISO 100", "ISO 101", "ISO 110"],
            "caused shaking level of 1 in ": ["Foo City", "Foofooville", "Foopolis"],
        },
        "tsqa-flamingo": {
            "has ISO value of ": ["ISO 100", "ISO 100", "ISO 100"],
            "caused shaking level of 1 in ": ["Foo City", "Foo City", "Foofooville"],
        },
    },
    "ablated": {
        "ecg-flamingo": {
            "has ISO value of ": ["ISO 111", "ISO 211", "ISO 999"],
            "caused shaking level of 1 in ": ["Footown", "Isle of Foo", "Los FooFoo"],
        },
        "tsqa-flamingo": {
            "has ISO value of ": ["ISO 100", "ISO 101", "ISO 100"],
            "caused shaking level of 1 in ": ["Point Foo", "Foopolis", "Los FooFoo"],
        },
    },
}

CONFIGS = [
    {
        "base_model_name": "GemmaV3P270M",
        "base_model_reformatted": "gemma-3-270m",
        "pretraining_stage": "ecg-flamingo",
        "num_stations": 4,
        "kg_size": 4,
        "batch_size": 4,
        "fold_id": 0,
    },
    {
        "base_model_name": "GemmaV3P270M",
        "base_model_reformatted": "gemma-3-270m",
        "pretraining_stage": "tsqa-flamingo",
        "num_stations": 4,
        "kg_size": 4,
        "batch_size": 4,
        "fold_id": 0,
    },
]

TOKENIZER_ID = "google/gemma-3-270m"
MODEL_ID_TEMPLATE = "OpenTSLM_{base_model_reformatted}-{pretraining_stage}"
DATE_STR = "02182026"

DIFF_COLUMNS = [
    "best_normalized_Damerau_Levenshtein_dist",
    "best_normalized_hamming_dist",
    "best_euclidean_dist_avgd_across_str",
]
MATCH_COLS = [
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


def _encode_text(tokenizer, text):
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(token_ids) == 0:
        raise ValueError(f"Tokenizer produced empty ids for text: {text!r}")
    return np.array([token_ids], dtype=np.int64)


def _metric_row_from_ids(tokenizer, target_ids, pred_ids):
    target_text = tokenizer.decode(target_ids[0].tolist(), skip_special_tokens=False)
    pred_text = tokenizer.decode(pred_ids[0].tolist(), skip_special_tokens=False)
    return {
        "best_normalized_Damerau_Levenshtein_dist": _char_level_normalized_Damerau_Levenshtein_dist(pred_text, target_text),
        "best_normalized_hamming_dist": _char_level_normalized_hamming_dist(pred_text, target_text),
        "best_euclidean_dist_avgd_across_str": _euclidean_dist_avgd_across_str(pred_text, target_text),
        "target_text": target_text,
        "pred_text": pred_text,
    }


def _build_eval_sample(tokenizer, edge, pred_text):
    input_text = f"Question: {edge}"
    target_text = EDGE_TARGETS[edge]

    input_ids = _encode_text(tokenizer, input_text)
    target_ids = _encode_text(tokenizer, target_text)
    pred_ids = _encode_text(tokenizer, pred_text)

    sample = {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "pred_ids": pred_ids,
        "pred_probs": [1.0 for _ in range(pred_ids.shape[1])],
        "results_by_seq_pos": [],
    }
    metric_row = _metric_row_from_ids(tokenizer, target_ids, pred_ids)
    return sample, input_text, metric_row


def _summary_df_from_samples_df(samples_df):
    group_cols = [
        "base_model_name",
        "pretraining_stage",
        "num_stations",
        "kg_size",
        "batch_size",
        "fold_id",
        "query_relation_edge",
        "model_id",
    ]
    return (
        samples_df.groupby(group_cols)
        .agg(
            avg_best_normalized_Damerau_Levenshtein_dist=("best_normalized_Damerau_Levenshtein_dist", "mean"),
            std_best_normalized_Damerau_Levenshtein_dist=("best_normalized_Damerau_Levenshtein_dist", "std"),
            median_best_normalized_Damerau_Levenshtein_dist=("best_normalized_Damerau_Levenshtein_dist", "median"),
            max_best_normalized_Damerau_Levenshtein_dist=("best_normalized_Damerau_Levenshtein_dist", "max"),
            min_best_normalized_Damerau_Levenshtein_dist=("best_normalized_Damerau_Levenshtein_dist", "min"),
            avg_best_normalized_hamming_dist=("best_normalized_hamming_dist", "mean"),
            std_best_normalized_hamming_dist=("best_normalized_hamming_dist", "std"),
            median_best_normalized_hamming_dist=("best_normalized_hamming_dist", "median"),
            max_best_normalized_hamming_dist=("best_normalized_hamming_dist", "max"),
            min_best_normalized_hamming_dist=("best_normalized_hamming_dist", "min"),
            avg_best_euclidean_dist_avgd_across_str=("best_euclidean_dist_avgd_across_str", "mean"),
            std_best_euclidean_dist_avgd_across_str=("best_euclidean_dist_avgd_across_str", "std"),
            median_best_euclidean_dist_avgd_across_str=("best_euclidean_dist_avgd_across_str", "median"),
            max_best_euclidean_dist_avgd_across_str=("best_euclidean_dist_avgd_across_str", "max"),
            min_best_euclidean_dist_avgd_across_str=("best_euclidean_dist_avgd_across_str", "min"),
        )
        .reset_index()
    )


def _build_expected_compare_stats(non_abl_df, abl_df):
    rows = []
    for diff_column in DIFF_COLUMNS:
        for edge in EDGE_TARGETS:
            df1_f = non_abl_df[non_abl_df["query_relation_edge"] == edge].copy()
            df2_f = abl_df[abl_df["query_relation_edge"] == edge].copy()
            for d in (df1_f, df2_f):
                d["Key"] = d[MATCH_COLS].astype(str).apply(lambda r: "|".join(r), axis=1)

            merged = df1_f.merge(
                df2_f[["Key", diff_column, "pred_text"]],
                on="Key",
                how="inner",
                suffixes=("_1", "_2"),
            )
            merged[f"{diff_column}_diff"] = merged[f"{diff_column}_1"] - merged[f"{diff_column}_2"]
            metric_cols = [f"{diff_column}_1", f"{diff_column}_2", f"{diff_column}_diff"]
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

            for variant_name, variant_df in [
                ("merged", merged),
                ("merged_drop", merged_drop),
                ("merged_replace", merged_replace),
            ]:
                for metric_col in metric_cols:
                    series = pd.to_numeric(variant_df[metric_col], errors="coerce")
                    if "_drop" in variant_name or "_replace" in variant_name:
                        series = series[np.isfinite(series)]
                    rows.append(
                        {
                            "diff_column": diff_column,
                            "filter_col_target_value": edge,
                            "variant_name": variant_name,
                            "metric_col": metric_col,
                            "mean": series.mean(),
                            "std": series.std(),
                            "count": int(series.count()),
                            "median": series.median(),
                            "q1": series.quantile(0.25),
                            "q2": series.quantile(0.50),
                            "q3": series.quantile(0.75),
                        }
                    )
    return pd.DataFrame(rows)


def _write_pkl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pkl.dump(obj, handle)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic generative_eval_results PKLs + expected stats for aggregate/compareExtensively scripts."
    )
    parser.add_argument(
        "--non-ablated-root",
        type=str,
        default="~/results/test_compOuts_non_ablated_v03022026",
    )
    parser.add_argument(
        "--ablated-root",
        type=str,
        default="~/results/test_compOuts_ablated_v03022026",
    )
    parser.add_argument(
        "--expected-root",
        type=str,
        default="~/results/test_compOuts_expected_v03022026",
    )
    args = parser.parse_args()

    non_root = Path(args.non_ablated_root).expanduser()
    abl_root = Path(args.ablated_root).expanduser()
    expected_root = Path(args.expected_root).expanduser()
    expected_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    samples_rows = {"non_ablated": [], "ablated": []}

    for scenario, root, pkl_name, maybe_ts_subdir in [
        ("non_ablated", non_root, "generative_eval_results.pkl", None),
        ("ablated", abl_root, "generative_eval_results_ts_ablated.pkl", "ts_ablated"),
    ]:
        for cfg in CONFIGS:
            stage = cfg["pretraining_stage"]
            base_model_reformatted = cfg["base_model_reformatted"]
            model_id = MODEL_ID_TEMPLATE.format(
                base_model_reformatted=base_model_reformatted,
                pretraining_stage=stage,
            )
            eval_results = {}

            for edge, pred_texts in PREDS_BY_SCENARIO_STAGE_EDGE[scenario][stage].items():
                samples = []
                for sample_id, pred_text in enumerate(pred_texts):
                    sample, input_text, metric_row = _build_eval_sample(tokenizer, edge, pred_text)
                    samples.append(sample)
                    samples_rows[scenario].append(
                        {
                            "model_id": model_id,
                            "query_relation_edge": edge,
                            "sample_id": sample_id,
                            "base_model_name": cfg["base_model_name"],
                            "num_stations": cfg["num_stations"],
                            "kg_size": cfg["kg_size"],
                            "batch_size": cfg["batch_size"],
                            "base_model_reformatted": cfg["base_model_reformatted"],
                            "pretraining_stage": stage,
                            "fold_id": cfg["fold_id"],
                            "input_text": input_text,
                            "target_text": metric_row["target_text"],
                            "pred_text": metric_row["pred_text"],
                            "best_normalized_Damerau_Levenshtein_dist": metric_row[
                                "best_normalized_Damerau_Levenshtein_dist"
                            ],
                            "best_normalized_hamming_dist": metric_row["best_normalized_hamming_dist"],
                            "best_euclidean_dist_avgd_across_str": metric_row[
                                "best_euclidean_dist_avgd_across_str"
                            ],
                        }
                    )
                eval_results[edge] = samples

            pkl_obj = {model_id: eval_results}

            rel_dir = (
                f"compOuts_of_usgs_finetuned_models_v02232026_TSLM{cfg['base_model_name']}"
                f"/usgs_LP50hzSW4096StSub{cfg['num_stations']}Kg{cfg['kg_size']}_"
                f"seismic_data_LatLonRegSplit_v02162026_batchSize{cfg['batch_size']}_v{DATE_STR}"
                f"/OpenTSLM_{base_model_reformatted}-{stage}_fold_{cfg['fold_id']}"
            )
            out_dir = root / rel_dir
            if maybe_ts_subdir:
                out_dir = out_dir / maybe_ts_subdir
            _write_pkl(out_dir / pkl_name, pkl_obj)

    non_samples_df = pd.DataFrame(samples_rows["non_ablated"]).sort_values(MATCH_COLS + ["model_id"]).reset_index(drop=True)
    abl_samples_df = pd.DataFrame(samples_rows["ablated"]).sort_values(MATCH_COLS + ["model_id"]).reset_index(drop=True)

    non_summary_df = _summary_df_from_samples_df(non_samples_df)
    abl_summary_df = _summary_df_from_samples_df(abl_samples_df)
    compare_stats_df = _build_expected_compare_stats(non_samples_df, abl_samples_df)

    non_samples_df.to_csv(expected_root / "expected_non_ablated_aggregated_samples.csv", index=False)
    abl_samples_df.to_csv(expected_root / "expected_ablated_aggregated_samples.csv", index=False)
    non_summary_df.to_csv(expected_root / "expected_non_ablated_aggregated_summary.csv", index=False)
    abl_summary_df.to_csv(expected_root / "expected_ablated_aggregated_summary.csv", index=False)
    compare_stats_df.to_csv(expected_root / "expected_compareExtensively_summary_stats.csv", index=False)

    print("Synthetic testcase generation complete.")
    print("non-ablated PKLs:", non_root)
    print("ablated PKLs:", abl_root)
    print("expected files:", expected_root)


if __name__ == "__main__":
    main()
