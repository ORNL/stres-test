#!/usr/bin/env python3
"""
Report validation samples whose KG query+answer triples overlap with train samples.
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import os
import sys
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np


def decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_text(value.item())
        parts = []
        for elem in value.ravel().tolist():
            parts.append(decode_text(elem))
        return "".join(parts)
    if isinstance(value, (list, tuple)):
        return "".join(decode_text(elem) for elem in value)
    return str(value)


def load_samples(h5_path: str) -> Iterable[Tuple[str, str, str, str, str, str]]:
    with h5py.File(h5_path, "r") as h5f:
        for key in sorted(h5f.keys()):
            if not key.startswith("sample_"):
                continue
            grp = h5f[key]
            kg_grp = grp.get("text_based_knowedge_graph_sample")
            if kg_grp is None:
                continue
            query = decode_text(kg_grp["query"][()])
            answer = decode_text(kg_grp["answer"][()])
            general_kg = ""
            event_kg = ""
            if "general_knowledge" in kg_grp:
                general_kg = decode_text(kg_grp["general_knowledge"][()])
            if "event_knowledge" in kg_grp:
                event_kg = decode_text(kg_grp["event_knowledge"][()])
            event_id = str(grp.attrs.get("event_id", ""))
            yield key, event_id, query, answer, general_kg, event_kg


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def parse_kg_triples(text: str) -> List[Tuple[str, str, str]]:
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    triples: List[Tuple[str, str, str]] = []
    if isinstance(parsed, (list, tuple, set)):
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                triples.append((str(item[0]), str(item[1]), str(item[2])))
    return triples


def triple_text_from_parts(subject: str, predicate: str, obj: str, normalize: bool) -> str:
    text = f"{subject} {predicate} {obj}"
    return normalize_text(text) if normalize else text


def triple_text_from_query_answer(query: str, answer: str, normalize: bool) -> str:
    text = f"{query} {answer}"
    return normalize_text(text) if normalize else text


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check for overlapping KG queries between train and val splits."
    )
    parser.add_argument(
        "--datasets-root",
        default=os.path.expanduser("~/datasets"),
        help="Root folder containing usgs_LP50hzSW4096* datasets.",
    )
    parser.add_argument(
        "--pattern",
        default="usgs_LP50hzSW4096*spe5_seismic_data",
        help="Dataset folder glob pattern under datasets-root.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.getcwd(), "query_overlap_reports"),
        help="Directory for CSV reports.",
    )
    parser.add_argument(
        "--no-normalize-queries",
        action="store_true",
        help="Disable whitespace normalization for query comparison.",
    )
    return parser.parse_args()


def main() -> int:
    print("main(): START ", flush=True)
    args = parse_args()
    datasets_root = os.path.expanduser(args.datasets_root)
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    ensure_dir(output_dir)

    dataset_paths = sorted(glob.glob(os.path.join(datasets_root, args.pattern)))
    print("main(): dataset_paths == ", dataset_paths, flush=True)
    if not dataset_paths:
        print(f"No datasets found under {datasets_root} matching {args.pattern}", file=sys.stderr)
        return 1
    print("main(): len(dataset_paths) == ", len(dataset_paths), flush=True)

    overlap_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    normalize = not args.no_normalize_queries

    for dp_ind, dataset_path in enumerate(dataset_paths):
        print("main(): \t now processing dp_ind == ", dp_ind, flush=True)
        dataset_name = os.path.basename(dataset_path)
        folds_root = os.path.join(dataset_path, "all_sensor_recorded_events")
        if not os.path.isdir(folds_root):
            warnings.append(f"Missing all_sensor_recorded_events in {dataset_name}")
            continue

        fold_dirs = sorted(glob.glob(os.path.join(folds_root, "fold_*")))
        if not fold_dirs:
            warnings.append(f"No fold_* directories in {dataset_name}")
            continue

        for fold_dir in fold_dirs:
            fold_name = os.path.basename(fold_dir)
            train_path = os.path.join(fold_dir, "train_data.h5")
            val_path = os.path.join(fold_dir, "val_data.h5")
            if not os.path.exists(train_path):
                warnings.append(f"Missing train_data.h5 in {dataset_name}/{fold_name}")
                continue
            if not os.path.exists(val_path):
                warnings.append(f"Missing val_data.h5 in {dataset_name}/{fold_name}")
                continue

            train_index: Dict[str, List[Dict[str, str]]] = {}
            train_samples: List[Dict[str, str]] = []
            train_count = 0
            for sample_id, event_id, query, answer, general_kg, event_kg in load_samples(
                train_path
            ):
                query_key = normalize_text(query) if normalize else query
                query_triple = triple_text_from_query_answer(query, answer, normalize)
                sample_info = {
                    "sample_id": sample_id,
                    "event_id": event_id,
                    "query": query,
                    "answer": answer,
                    "query_triple": query_triple,
                    "general_knowledge": general_kg,
                    "event_knowledge": event_kg,
                }
                train_samples.append(sample_info)
                train_index.setdefault(query_key, []).append(sample_info)
                train_count += 1

            val_count = 0
            overlap_count = 0
            overlap_query_count = 0
            overlap_general_count = 0
            overlap_event_count = 0
            overlap_query_keys: set[str] = set()
            val_query_keys: set[str] = set()
            general_match_cache: Dict[str, List[Dict[str, str]]] = {}
            event_match_cache: Dict[str, List[Dict[str, str]]] = {}
            for (
                val_sample_id,
                val_event_id,
                val_query,
                val_answer,
                _,
                _,
            ) in load_samples(val_path):
                val_count += 1
                val_key = normalize_text(val_query) if normalize else val_query
                val_triple_text = triple_text_from_query_answer(val_query, val_answer, normalize)
                val_query_keys.add(val_key)
                query_matches = [
                    match for match in train_index.get(val_key, []) if match["query_triple"] == val_triple_text
                ]
                if val_triple_text in general_match_cache:
                    general_matches = general_match_cache[val_triple_text]
                else:
                    general_matches = []
                    if val_triple_text:
                        for sample_info in train_samples:
                            for triple in parse_kg_triples(sample_info["general_knowledge"]):
                                if triple_text_from_parts(*triple, normalize=normalize) == val_triple_text:
                                    general_matches.append(sample_info)
                                    break
                    general_match_cache[val_triple_text] = general_matches
                if val_triple_text in event_match_cache:
                    event_matches = event_match_cache[val_triple_text]
                else:
                    event_matches = []
                    if val_triple_text:
                        for sample_info in train_samples:
                            for triple in parse_kg_triples(sample_info["event_knowledge"]):
                                if triple_text_from_parts(*triple, normalize=normalize) == val_triple_text:
                                    event_matches.append(sample_info)
                                    break
                    event_match_cache[val_triple_text] = event_matches

                if not query_matches and not general_matches and not event_matches:
                    continue

                overlap_count += 1
                overlap_query_keys.add(val_key)
                if query_matches:
                    overlap_query_count += 1
                if general_matches:
                    overlap_general_count += 1
                if event_matches:
                    overlap_event_count += 1

                seen_ids = set()
                union_matches: List[Dict[str, str]] = []
                for match_list in (query_matches, general_matches, event_matches):
                    for match in match_list:
                        if match["sample_id"] in seen_ids:
                            continue
                        seen_ids.add(match["sample_id"])
                        union_matches.append(match)

                train_sample_ids = "|".join(match["sample_id"] for match in union_matches)
                train_event_ids = "|".join(match["event_id"] for match in union_matches)
                train_answers = "|".join(match["answer"] for match in union_matches)
                train_query_match_sample_ids = "|".join(
                    match["sample_id"] for match in query_matches
                )
                train_general_match_sample_ids = "|".join(
                    match["sample_id"] for match in general_matches
                )
                train_event_match_sample_ids = "|".join(
                    match["sample_id"] for match in event_matches
                )
                train_query_match_triples = "|".join(
                    match["query_triple"] for match in query_matches
                )
                train_general_match_triples = "|".join(
                    val_triple_text for _ in general_matches
                )
                train_event_match_triples = "|".join(
                    val_triple_text for _ in event_matches
                )

                overlap_rows.append(
                    {
                        "dataset": dataset_name,
                        "fold": fold_name,
                        "val_sample_id": val_sample_id,
                        "val_event_id": val_event_id,
                        "val_query": val_query,
                        "val_answer": val_answer,
                        "val_triple": val_triple_text,
                        "train_match_count": len(union_matches),
                        "train_sample_ids": train_sample_ids,
                        "train_event_ids": train_event_ids,
                        "train_answers": train_answers,
                        "train_query_match_sample_ids": train_query_match_sample_ids,
                        "train_general_match_sample_ids": train_general_match_sample_ids,
                        "train_event_match_sample_ids": train_event_match_sample_ids,
                        "train_query_match_triples": train_query_match_triples,
                        "train_general_match_triples": train_general_match_triples,
                        "train_event_match_triples": train_event_match_triples,
                    }
                )

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "fold": fold_name,
                    "num_train_samples": train_count,
                    "num_val_samples": val_count,
                    "num_overlapping_val_samples": overlap_count,
                    "num_overlapping_val_samples_query_match": overlap_query_count,
                    "num_overlapping_val_samples_general_kg_match": overlap_general_count,
                    "num_overlapping_val_samples_event_kg_match": overlap_event_count,
                    "num_unique_val_queries": len(val_query_keys),
                    "num_unique_overlapping_val_queries": len(overlap_query_keys),
                }
            )

    overlap_csv_path = os.path.join(output_dir, "val_train_query_overlaps.csv")
    summary_csv_path = os.path.join(output_dir, "val_train_query_overlap_summary.csv")

    print("main(): writing overlap_rows if necessary", flush=True)
    if overlap_rows:
        with open(overlap_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(overlap_rows[0].keys()))
            writer.writeheader()
            writer.writerows(overlap_rows)

    print("main(): writing summary_rows if necessary", flush=True)
    if summary_rows:
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print("main(): writing warnings if necessary", flush=True)
    if warnings:
        warn_path = os.path.join(output_dir, "val_train_query_overlap_warnings.txt")
        with open(warn_path, "w") as f:
            f.write("\n".join(warnings))

    print(f"main(): Wrote overlap CSV: {overlap_csv_path}")
    print(f"main(): Wrote summary CSV: {summary_csv_path}")
    if warnings:
        print(f"Wrote warnings: {os.path.join(output_dir, 'val_train_query_overlap_warnings.txt')}")
    print("main(): STOP ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
