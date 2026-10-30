""" Written Using GPT-5.2-Codex (Early 2026) """
import os
import re
import sys
import argparse

import numpy as np
import pickle as pkl

from pathlib import Path
from transformers import AutoTokenizer

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError("pandas is required for this script") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib is required for this script") from exc

main_script_file_path = Path(__file__)
absolute_main_script_file_path = main_script_file_path.resolve()
repo_directory = absolute_main_script_file_path.parent.parent
sys.path.append(str(repo_directory))
opentslm_src_directory = repo_directory / "OpenTSLM" / "src"
sys.path.append(str(opentslm_src_directory))
from evaluate.eval_utils import _char_level_normalized_hamming_dist, _char_level_normalized_Damerau_Levenshtein_dist, _euclidean_dist_avgd_across_str

DECODE_TEXT = True
METRICS_FOR_EVAL = ["best_normalized_Damerau_Levenshtein_dist", "best_normalized_hamming_dist", "best_euclidean_dist_avgd_across_str", ]
EDGE_GROUPING_CSVs_FOR_EVAL = ["eval_edge_groupings_rawEdgeFocussedV02252026.csv",]
PRETRAINING_STAGES = ["tsqa-flamingo", "m4-flamingo", "har-flamingo", "sleep-flamingo", "ecg-flamingo",]
EDGE_GROUP_PLOT_PRESET = "edge_group_x_model"
TOKENIZER_ID_BY_BASE_MODEL = {
    "gemma-3-270m": "google/gemma-3-270m",
    "gemma-3-1b-pt": "google/gemma-3-1b-pt",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
}
STRES_DATASET_HYPERPARAMS_TO_NAME_MAP = {
    "4/4": "Small-Uninformed", #"8/4": "Medium-Uninformed",
    "16/4": "Large-Uninformed",
    "4/10": "Small-Informed", #"8/10": "Medium-Informed",
    "16/10": "Large-Informed",
}


def _normalize_pkl_path(path_obj):
    if path_obj.is_file():
        return path_obj
    if path_obj.suffix == "" and path_obj.with_suffix(".pkl").is_file():
        return path_obj.with_suffix(".pkl")
    return None

def _parse_metadata_from_path(pkl_path):
    # Example: compOuts_of_usgs_finetuned_models_v02232026_TSLMGemmaV3P270M/usgs_LP50hzSW4096StSub16Kg10_seismic_data_LatLonRegSplit_v02162026_batchSize1_v02182026/OpenTSLM_gemma-3-270m-ecg-flamingo_fold_0
    path_str = str(pkl_path)
    stage_regex = "|".join(PRETRAINING_STAGES)
    patterns = [
        re.compile(
            r"compOuts_of_usgs_finetuned_models_v\d+_TSLM"
            r"(?P<base_model_name>[^/\\]+)"
            r"[\\/]+usgs_LP50hzSW4096StSub(?P<num_stations>\d+)Kg(?P<kg_size>\d+)"
            r"_seismic_data_LatLonRegSplit_v\d+_batchSize(?P<batch_size>\d+)_v\d+"
            r"[\\/]+OpenTSLM_(?P<base_model_reformatted>.+?)-(?P<pretraining_stage>"
            + stage_regex +
            r")_fold_(?P<fold_id>\d+)"
        ),
    ]
    match = None
    for pattern in patterns:
        match = pattern.search(path_str)
        if match:
            break
    if not match:
        raise ValueError("Could not parse metadata from path: "+str(path_str))
    md = match.groupdict()
    md["num_stations"] = int(md["num_stations"])
    md["kg_size"] = int(md["kg_size"])
    md["batch_size"] = int(md["batch_size"])
    md["fold_id"] = int(md["fold_id"])
    return md

def _resolve_tokenizer_id(metadata):
    base_model = metadata.get("base_model_reformatted")
    print("_resolve_tokenizer_id: base_model == ", base_model, flush=True)
    if not base_model:
        raise ValueError("Tokenizer is None")
        return None
    for prefix, tok_id in TOKENIZER_ID_BY_BASE_MODEL.items():
        print("_resolve_tokenizer_id: prefix == ", prefix, flush=True)
        if base_model.startswith(prefix):
            print("_resolve_tokenizer_id: tok_id == ", tok_id, flush=True)
            return tok_id
    raise ValueError("Tokenizer is None")

def _get_tokenizer(tokenizer_id, tokenizer_cache):
    if tokenizer_id in tokenizer_cache:
        return tokenizer_cache[tokenizer_id]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_cache[tokenizer_id] = tokenizer
    return tokenizer

def _decode_ids(tokenizer, ids, skip_special_tokens):
    if tokenizer is None:
        return None
    if ids is None:
        return None
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

def _iter_model_eval_results(pkl_obj):
    if isinstance(pkl_obj, dict):
        # If dict values are lists, treat as a single eval_results object.
        if all(isinstance(v, list) for v in pkl_obj.values()):
            yield "unknown_model", pkl_obj
            return
        # If dict values are dicts whose values are lists, treat as model_id -> eval_results.
        for model_id, maybe_eval_res in pkl_obj.items():
            if isinstance(maybe_eval_res, dict) and all(isinstance(v, list) for v in maybe_eval_res.values()):
                yield model_id, maybe_eval_res
            else: # Skip unknown structure.
                continue
    pass


def _collect_rows_from_eval_results(
    pkl_path, pkl_obj, decode_skip_special_tokens=True, tokenizer_cache=None, tokenizer_warn_cache=None,
):
    metadata = _parse_metadata_from_path(pkl_path)

    # set up tokenizer
    tokenizer = None
    assert DECODE_TEXT
    tok_id = _resolve_tokenizer_id(metadata)
    if tok_id is None:
        warn_key = "missing_tokenizer_id"
        if warn_key not in tokenizer_warn_cache:
            print("eval: no tokenizer_id resolved; skipping decoding unless --decode-tokenizer-id is set", flush=True)
            tokenizer_warn_cache.add(warn_key)
    else:
        tokenizer = _get_tokenizer(tok_id, tokenizer_cache)

    rows = []
    for model_id, eval_results in _iter_model_eval_results(pkl_obj):
        for query_relation_edge, samples in eval_results.items():
            for sample_idx, sample in enumerate(samples):
                # get textual representation(s) of preds and targets
                pred_ids = sample.get("pred_ids", None)
                targ_ids = sample.get("target_ids", None)
                input_ids = sample.get("input_ids", None)
                if pred_ids is None or targ_ids is None:
                    continue
                pred_list = pred_ids[0].tolist()
                targ_list = targ_ids[0].tolist()
                assert tokenizer is not None
                pred_text = _decode_ids(tokenizer, pred_ids[0], decode_skip_special_tokens)
                targ_text = _decode_ids(tokenizer, targ_ids[0], decode_skip_special_tokens)
                
                # compare text with metrics
                best_text_normalized_Damerau_Levenshtein_dist = None
                best_text_normalized_hamming_dist = None
                best_text_euclidean_dist_avgd_across_str = None
                if "best_normalized_Damerau_Levenshtein_dist" in METRICS_FOR_EVAL: 
                    best_text_normalized_Damerau_Levenshtein_dist = _char_level_normalized_Damerau_Levenshtein_dist(pred_text, targ_text)
                if "best_normalized_hamming_dist" in METRICS_FOR_EVAL: 
                    best_text_normalized_hamming_dist = _char_level_normalized_hamming_dist(pred_text, targ_text)
                if "best_euclidean_dist_avgd_across_str" in METRICS_FOR_EVAL: 
                    best_text_euclidean_dist_avgd_across_str = _euclidean_dist_avgd_across_str(pred_text, targ_text)
                
                # record the results
                row = {
                    "pkl_path": str(pkl_path),
                    "model_id": model_id,
                    "query_relation_edge": query_relation_edge,
                    "sample_id": sample_idx,
                    "pred_len": len(pred_list),
                    "targ_len": len(targ_list), 
                    "best_normalized_Damerau_Levenshtein_dist": best_text_normalized_Damerau_Levenshtein_dist, 
                    "best_normalized_hamming_dist": best_text_normalized_hamming_dist, 
                    "best_euclidean_dist_avgd_across_str": best_text_euclidean_dist_avgd_across_str, 
                    **metadata,
                }
                assert DECODE_TEXT
                row["input_text"] = _decode_ids(tokenizer, input_ids[0], decode_skip_special_tokens)
                row["target_text"] = targ_text
                row["pred_text"] = pred_text
                rows.append(row)

                # if sample_idx == 10: # FOR DEBUGGING PURPOSES
                #     print("BREAKING FOR DEBUGGING PURPOSES") # FOR DEBUGGING PURPOSES
                #     break # FOR DEBUGGING PURPOSES

            # print("BREAKING FOR DEBUGGING PURPOSES") # FOR DEBUGGING PURPOSES
            # break # FOR DEBUGGING PURPOSES
            
        # print("BREAKING FOR DEBUGGING PURPOSES") # FOR DEBUGGING PURPOSES
        # break # FOR DEBUGGING PURPOSES
    return rows


def _summarize_dataframe(df, group_cols):
    assert "best_normalized_Damerau_Levenshtein_dist" in METRICS_FOR_EVAL
    assert "best_normalized_hamming_dist" in METRICS_FOR_EVAL
    assert "best_euclidean_dist_avgd_across_str" in METRICS_FOR_EVAL
    return (
        df.groupby(group_cols).agg(
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
        ).reset_index()
    )

def _compute_group_spans(values):
    spans = []
    if not values:
        return spans
    start = 0
    current = values[0]
    for idx, value in enumerate(values[1:], start=1):
        if value != current:
            spans.append((current, start, idx - 1))
            start = idx
            current = value
    spans.append((current, start, len(values) - 1))
    return spans

def _format_leaf_label(row, leaf_cols):
    num_stations = row["num_stations"] if "num_stations" in leaf_cols else None
    kg_size = row["kg_size"] if "kg_size" in leaf_cols else None
    batch_size = row["batch_size"] if "batch_size" in leaf_cols else None
    if num_stations is not None and kg_size is not None and batch_size is not None:
        map_key = f"{num_stations}/{kg_size}"
        dataset_name = STRES_DATASET_HYPERPARAMS_TO_NAME_MAP.get(map_key, map_key)
        return f"{dataset_name}/{batch_size}"
    return "/".join(str(row[col]) for col in leaf_cols)

def _sort_grouped_by_columns(grouped, group_by):
    if "pretraining_stage" not in group_by:
        return grouped.sort_values(group_by)
    order_map = {stage: idx for idx, stage in enumerate(PRETRAINING_STAGES)}
    grouped = grouped.copy()
    grouped["_pretraining_stage_order"] = (grouped["pretraining_stage"].map(order_map).fillna(len(order_map)).astype(int))
    sort_cols = []
    for col in group_by:
        if col == "pretraining_stage":
            sort_cols.append("_pretraining_stage_order")
        sort_cols.append(col)
    return grouped.sort_values(sort_cols).drop(columns="_pretraining_stage_order")

def _add_hierarchical_xtick_labels(ax, grouped, base_col, stage_col, leaf_cols):
    x = np.arange(len(grouped))
    leaf_labels = [_format_leaf_label(grouped.loc[i], leaf_cols) for i in grouped.index]
    ax.set_xticks(x)
    ax.set_xticklabels(leaf_labels, rotation=60, ha="right",fontsize=12)
    stage_keys = list(zip(grouped[base_col].tolist(), grouped[stage_col].tolist()))
    base_values = grouped[base_col].tolist()
    stage_spans = _compute_group_spans(stage_keys)
    base_spans = _compute_group_spans(base_values)
    stage_y = -0.55#-0.18
    base_y = -0.70#-0.34
    for (stage_key, start, end) in stage_spans:
        center = (start + end) / 2.0
        ax.text(center, stage_y, str(stage_key[1]), ha="center", va="top", transform=ax.get_xaxis_transform(), clip_on=False, fontsize=14)
    for (base_value, start, end) in base_spans:
        center = (start + end) / 2.0
        ax.text(center, base_y, str(base_value), ha="center", va="top", transform=ax.get_xaxis_transform(), clip_on=False, fontsize=18)
        if end < len(base_values) - 1:
            ax.axvline(end + 0.5, color="gray", linewidth=0.6, alpha=0.4)
    pass

def _split_group_label(label, expected_parts):
    parts = str(label).split("|")
    if expected_parts:
        return parts[: len(expected_parts)]
    return parts


def _order_labels_by_pretraining_stage(labels, x_group_cols):
    if not x_group_cols or "pretraining_stage" not in x_group_cols:
        return labels
    order_map = {stage: idx for idx, stage in enumerate(PRETRAINING_STAGES)}

    def sort_key(label):
        parts = _split_group_label(label, x_group_cols)
        parts_by_col = dict(zip(x_group_cols, parts))
        key = []
        for col in x_group_cols:
            if col == "pretraining_stage":
                key.append(order_map.get(parts_by_col.get(col, ""), len(order_map)))
            else:
                key.append(parts_by_col.get(col, ""))
        return tuple(key)

    return sorted(labels, key=sort_key)


def _add_hierarchical_xtick_labels_from_parts(ax, base_values, stage_values, leaf_labels):
    x = np.arange(len(leaf_labels))
    ax.set_xticks(x)
    ax.set_xticklabels(leaf_labels, rotation=60, ha="right",fontsize=12)
    stage_keys = list(zip(base_values, stage_values))
    stage_spans = _compute_group_spans(stage_keys)
    base_spans = _compute_group_spans(base_values)
    stage_y = -0.55#-0.18
    base_y = -0.70#-0.34
    for (stage_key, start, end) in stage_spans:
        center = (start + end) / 2.0
        ax.text(center, stage_y, str(stage_key[1]), ha="center", va="top", transform=ax.get_xaxis_transform(), clip_on=False, fontsize=14)
    for (base_value, start, end) in base_spans:
        center = (start + end) / 2.0
        ax.text(center, base_y, str(base_value), ha="center", va="top", transform=ax.get_xaxis_transform(), clip_on=False, fontsize=18)
        if end < len(base_values) - 1:
            ax.axvline(end + 0.5, color="gray", linewidth=0.6, alpha=0.4)
    pass

def _plot_bar(df, metric, group_by, output_path, title=None, figsize=(27, 7)):
    grouped = (df.groupby(group_by).agg(metric_mean=(metric, "mean"), metric_std=(metric, "std")).reset_index())
    grouped = _sort_grouped_by_columns(grouped, group_by).reset_index(drop=True)
    leaf_cols = [c for c in ["num_stations", "kg_size", "batch_size"] if c in grouped.columns]
    use_hierarchical = all(c in grouped.columns for c in ["base_model_name", "pretraining_stage"]) and leaf_cols
    if use_hierarchical:
        labels = [_format_leaf_label(grouped.loc[i], leaf_cols) for i in grouped.index]
    else:
        labels = ["|".join(str(grouped.loc[i, col]) for col in group_by) for i in grouped.index]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        x,
        grouped["metric_mean"].values,
        yerr=grouped["metric_std"].values,
        capsize=4,
        color="steelblue",
        ecolor="gray",
    )
    if use_hierarchical:
        _add_hierarchical_xtick_labels(ax, grouped, "base_model_name", "pretraining_stage", leaf_cols)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right",fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title or f"{metric} by " + ", ".join(group_by))
    fig.tight_layout()
    if use_hierarchical:
        fig.subplots_adjust(bottom=0.44)
    fig.savefig(output_path)
    plt.close(fig)
    pass

def _sanitize_label_for_filename(label):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_")

def _plot_bar_by_edge_group(df, metric, group_by, plot_dir, edge_group_col):
    if edge_group_col not in df.columns:
        raise ValueError("edge_group_col not found in DataFrame")
    edge_groups = df[edge_group_col].dropna().unique().tolist()
    for edge_group in edge_groups:
        subset_df = df[df[edge_group_col] == edge_group]
        if subset_df.empty:
            continue
        safe_edge_group = _sanitize_label_for_filename(edge_group)
        output_path = plot_dir / f"{metric}_bar_{edge_group_col}_{safe_edge_group}.png"
        title = f"{metric} by " + ", ".join(group_by) + f" ({edge_group_col}={edge_group})"
        _plot_bar(subset_df, metric, group_by, output_path, title=title)
    pass

def _plot_heatmap(df, metric, group_by, output_path, title=None, figsize=(27, 7), cmap="viridis", x_group_cols=None):
    if len(group_by) != 2:
        raise ValueError("heatmap requires exactly two group_by fields")
    pivot = (df.groupby(group_by).agg(metric_mean=(metric, "mean")).reset_index().pivot(index=group_by[0], columns=group_by[1], values="metric_mean"))
    x_labels = [str(c) for c in pivot.columns]
    if x_group_cols:
        ordered_labels = _order_labels_by_pretraining_stage(x_labels, x_group_cols)
        pivot = pivot.reindex(columns=ordered_labels)
        x_labels = ordered_labels
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    fig.colorbar(im, label=metric)
    if x_group_cols:
        leaf_cols = [c for c in ["num_stations", "kg_size", "batch_size"] if c in x_group_cols]
        if "base_model_name" in x_group_cols and "pretraining_stage" in x_group_cols and leaf_cols:
            base_values = []
            stage_values = []
            leaf_labels = []
            for label in x_labels:
                parts = _split_group_label(label, x_group_cols)
                parts_by_col = dict(zip(x_group_cols, parts))
                base_values.append(parts_by_col.get("base_model_name", ""))
                stage_values.append(parts_by_col.get("pretraining_stage", ""))
                num_stations = parts_by_col.get("num_stations", "")
                kg_size = parts_by_col.get("kg_size", "")
                batch_size = parts_by_col.get("batch_size", "")
                map_key = f"{num_stations}/{kg_size}"
                dataset_name = STRES_DATASET_HYPERPARAMS_TO_NAME_MAP.get(map_key, map_key)
                leaf_labels.append(f"{dataset_name}/{batch_size}")
            _add_hierarchical_xtick_labels_from_parts(ax, base_values, stage_values, leaf_labels)
        else:
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=60, ha="right",fontsize=12)
    else:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=60, ha="right",fontsize=12)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index],fontsize=12)
    ax.set_title(title or f"{metric} heatmap by " + ", ".join(group_by))
    fig.tight_layout()
    if x_group_cols and "base_model_name" in x_group_cols and "pretraining_stage" in x_group_cols:
        fig.subplots_adjust(bottom=0.44)
    fig.savefig(output_path)
    plt.close(fig)
    pass

def _merge_edge_groups(df, edge_group_path, edge_group_col):
    edge_group_df = pd.read_csv(edge_group_path)
    required_cols = {"query_relation_edge", edge_group_col}
    if not required_cols.issubset(set(edge_group_df.columns)):
        raise ValueError("edge group file must include columns: "+ ", ".join(sorted(required_cols)))
    df = df.copy()
    df[edge_group_col] = None
    for _, row in edge_group_df.iterrows():
        prefix = str(row["query_relation_edge"]).strip().lower()
        if prefix == "":
            continue
        mask = df["query_relation_edge"].str.strip().str.lower().str.startswith(prefix, na=False)
        df.loc[mask, edge_group_col] = row[edge_group_col]
    return df

def _add_group_label_column(df, cols, new_col):
    df = df.copy()
    df[new_col] = df[cols].astype(str).agg("|".join, axis=1)
    return df

def _run_plot_preset(df, plot_dir, preset_name, metric, edge_group_col, x_group_cols, cmap):
    if preset_name == "edge_group_x_model":
        if edge_group_col not in df.columns:
            raise ValueError("edge_group_x_model preset requires edge group column in DataFrame")
        df = _add_group_label_column(df, x_group_cols, "x_group_label")
        _plot_heatmap(
            df,
            metric,
            [edge_group_col, "x_group_label"],
            plot_dir / f"{metric}_edge_group_x_model_heatmap.png",
            title=f"{metric} heatmap by {edge_group_col} and x_group_label",
            cmap=cmap,
            x_group_cols=x_group_cols,
        )
    else:
        raise ValueError("Unrecognized preset_name == "+str(preset_name))
    pass


def eval(args, pkl_paths, base_csv_dir, plot_dir, plot_metric, edge_group_path): 
    # unpack args ----------------------------------------------------------------------------------------
    plot_group_by = [c.strip() for c in args.plot_group_by.split(",") if c.strip()]
    edge_group_x_cols = [c.strip() for c in args.edge_group_x_cols.split(",") if c.strip()]

    # aggregate results into dataframes ------------------------------------------------------------------
    print("eval: cultivating rows", flush=True)
    rows = []
    tokenizer_cache = {}
    tokenizer_warn_cache = set()
    for p, pkl_path in enumerate(pkl_paths):
        print("eval: \t now processing pkl_path p == ", p, " of ", len(pkl_paths), flush=True)
        with open(pkl_path, "rb") as infile:
            pkl_obj = pkl.load(infile)
        rows.extend(
            _collect_rows_from_eval_results(
                pkl_path,
                pkl_obj, 
                decode_skip_special_tokens=not args.decode_keep_special_tokens,
                tokenizer_cache=tokenizer_cache,
                tokenizer_warn_cache=tokenizer_warn_cache,
            )
        )
        # if p == 2: # FOR DEBUGGING PURPOSES
        #     print("BREAKING FOR DEBUGGING PURPOSES", flush=True) # FOR DEBUGGING PURPOSES
        #     break # FOR DEBUGGING PURPOSES

    print("eval: defining df", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid rows found in PKL files.")

    edge_merged_df = None
    if edge_group_path is not None:
        print("eval: merging edge groups", flush=True)
        edge_merged_df = _merge_edge_groups(df, edge_group_path, args.edge_group_col)

    print("eval: defining csvs", flush=True)
    base_csv_dir.mkdir(parents=True, exist_ok=True)
    output_csv = base_csv_dir / "aggregated_samples.csv"
    summary_csv = base_csv_dir / "aggregated_summary.csv"
    df.to_csv(output_csv, index=False)

    edge_merged_output_csv = None
    edge_merged_summary_csv = None
    if edge_merged_df is not None:
        edge_merged_output_csv = (base_csv_dir / "aggregated_samples_with_merged_edges.csv")
        edge_merged_summary_csv = (base_csv_dir / "aggregated_summary_with_merged_edges.csv")
        edge_merged_df.to_csv(edge_merged_output_csv, index=False)
    print("eval: edge_merged_output_csv == ", edge_merged_output_csv, flush=True)
    print("eval: edge_merged_summary_csv == ", edge_merged_summary_csv, flush=True)

    print("eval: defining summary df", flush=True)
    summary_group_cols = [
        "base_model_name", "pretraining_stage", "num_stations", "kg_size",
        "batch_size", "fold_id", "query_relation_edge", "model_id",
    ]
    summary_df = _summarize_dataframe(df, summary_group_cols)
    summary_df.to_csv(summary_csv, index=False)

    edge_merged_summary_df = None
    if edge_merged_df is not None: 
        assert edge_group_path is not None
        summary_group_cols.append(args.edge_group_col)
        edge_merged_summary_df = _summarize_dataframe(edge_merged_df, summary_group_cols)
        edge_merged_summary_df.to_csv(edge_merged_summary_csv, index=False)
    # ----------------------------------------------------------------------------------------------------

    print("eval: generating relevant plots", flush=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    _plot_bar(df, plot_metric, plot_group_by, plot_dir / f"{plot_metric}_bar.png",)
    
    if edge_merged_df is not None: 
        _run_plot_preset(
            edge_merged_df,
            plot_dir,
            EDGE_GROUP_PLOT_PRESET,
            plot_metric,
            args.edge_group_col,
            edge_group_x_cols,
            args.heatmap_cmap,
        )
        if args.plot_bar_by_edge_group:
            _plot_bar_by_edge_group(
                edge_merged_df, 
                plot_metric, 
                plot_group_by, 
                plot_dir, 
                args.edge_group_col
            )

    print(f"Saved sample-level CSV: {output_csv}")
    print(f"Saved summary CSV: {summary_csv}", flush=True)
    print(f"Saved plots to: {plot_dir}", flush=True)
    pass


def main():
    parser = argparse.ArgumentParser(description="Aggregate OpenTSLM generative eval PKLs into a DataFrame and plot summaries.")
    parser.add_argument("--results-dir", type=str, required=True, help="Root directory containing compOuts outputs.",)
    parser.add_argument("--csv-root-save-folder", type=str, required=True, help="Folder to save CSVs to.",) 
    parser.add_argument("--plot-subdir", type=str, required=True, help="Foldername for plots.",)
    parser.add_argument("--edge-group-configs-root-dir", type=str, required=True, help="Path to dir where CSVs define query_relation_edge -> edge group mapping.",)
    parser.add_argument("--glob", type=str, default="compOuts_TSLM*/OpenTSLM_*_fold_*", help="Glob pattern for locating pkl files (relative to results-dir).",)
    parser.add_argument("--decode-keep-special-tokens", action="store_true", help="Keep special tokens when decoding text.",)
    parser.add_argument("--edge-group-x-cols", type=str, default="base_model_name,pretraining_stage,num_stations,kg_size,batch_size", help="Comma-separated columns to combine for heatmap x-axis.",)
    parser.add_argument("--heatmap-cmap", type=str, default="plasma", help="Colormap for heatmap plots.",)
    parser.add_argument("--plot-group-by", type=str, default="base_model_name,pretraining_stage,num_stations,kg_size", help="Comma-separated group-by fields for plotting.",)
    parser.add_argument("--plot-bar-by-edge-group", action="store_true", help="Generate one bar plot per edge group.",)
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of files to read.",)
    parser.add_argument("--edge-group-col", type=str, default="edge_group", help="Column name in edge-group CSV for group label.",)
    args = parser.parse_args()

    print("main: START", flush=True)
    print("main: --- ARGS --- ")
    print("main: \t args.results-dir == ", args.results_dir)
    print("main: \t args.glob == ", args.glob)
    print("main: \t DECODE_TEXT == ", DECODE_TEXT)
    print("main: \t args.decode-keep-special-tokens == ", args.decode_keep_special_tokens)
    print("main: \t args.csv-root-save-folder == ", args.csv_root_save_folder) 
    print("main: \t args.plot-subdir == ", args.plot_subdir)
    print("main: \t METRICS_FOR_EVAL == ", METRICS_FOR_EVAL)
    print("main: \t args.plot-group-by == ", args.plot_group_by)
    print("main: \t args.plot-bar-by-edge-group == ", args.plot_bar_by_edge_group)
    print("main: \t args.edge-group-x-cols == ", args.edge_group_x_cols)
    print("main: \t args.heatmap-cmap == ", args.heatmap_cmap)
    print("main: \t args.max-files == ", args.max_files)
    print("main: \t args.edge-group-configs-root-dir == ", args.edge_group_configs_root_dir)
    print("main: \t args.edge-group-col == ", args.edge_group_col)
    print("main: ------------ ", flush=True)

    results_dir = Path(args.results_dir).expanduser()

    # identify results files to process ------------------------------------------------------------------
    matches = sorted(results_dir.glob(args.glob))
    print("main: matches == ", matches, flush=True)
    pkl_paths = []
    for m in matches:
        normalized = _normalize_pkl_path(m)
        if normalized is not None:
            pkl_paths.append(normalized)
    if args.max_files is not None:
        pkl_paths = pkl_paths[: args.max_files]
    print("main: pkl_paths == ", pkl_paths, flush=True)

    if not pkl_paths:
        raise FileNotFoundError(f"No PKL files found under {results_dir} with glob {args.glob}")
    # ----------------------------------------------------------------------------------------------------

    for metric in METRICS_FOR_EVAL:
        print("\n\nmain: now performing evaluations using metric == ", metric, flush=True)
        plot_metric = metric
        for edge_csv in EDGE_GROUPING_CSVs_FOR_EVAL:
            curr_eval_folder_name = metric + "_" + edge_csv[:-1*len('.csv')]+"_csv"
            edge_group_path = args.edge_group_configs_root_dir + os.sep + edge_csv
            print("\nmain: \t kicking off evaluation using edge groupings defined in edge_group_path == ", edge_group_path, flush=True)
            base_csv_dir = Path(args.csv_root_save_folder+os.sep+curr_eval_folder_name)
            plot_dir = Path(args.csv_root_save_folder+os.sep+curr_eval_folder_name+os.sep+args.plot_subdir)
            eval(args, pkl_paths, base_csv_dir, plot_dir, plot_metric, edge_group_path)
            print("main: \t completed evaluation using edge groupings defined in edge_group_path == ", edge_group_path, flush=True)
        print("\nmain: completed evaluations using metric == ", metric, flush=True)
    print("\n\nmain: STOP", flush=True)
    pass



if __name__ == "__main__":
    print("__MAIN__: START", flush=True)
    main()
    print("__MAIN__: DONE !!!", flush=True)
