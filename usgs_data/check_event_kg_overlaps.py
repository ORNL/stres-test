#!/usr/bin/env python3
"""
Scan USGS datasets for event KG overlap with event-specific edge sets and
the universal general info KG, and write CSV summaries + optional heatmap.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import sys
from typing import Iterable, List, Dict, Set, Tuple


Triple = Tuple[str, str, str]


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_triples(obj) -> Set[Triple]:
    triples: Set[Triple] = set()
    if isinstance(obj, set) or isinstance(obj, list) or isinstance(obj, tuple):
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                triples.add((str(item[0]), str(item[1]), str(item[2])))
    elif hasattr(obj, "edges"):
        # Best-effort for graph-like objects.
        for edge in obj.edges(data=True):
            if len(edge) >= 3 and isinstance(edge[2], dict):
                rel = edge[2].get("relation") or edge[2].get("predicate") or edge[2].get("label")
                if rel is not None:
                    triples.add((str(edge[0]), str(rel), str(edge[1])))
    return triples


def extract_edge_types(obj) -> Set[str]:
    edge_types: Set[str] = set()
    if isinstance(obj, (set, list, tuple)):
        for item in obj:
            if isinstance(item, str):
                edge_types.add(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                if isinstance(item[1], str):
                    edge_types.add(item[1])
    elif hasattr(obj, "edges"):
        for edge in obj.edges(data=True):
            if len(edge) >= 3 and isinstance(edge[2], dict):
                rel = edge[2].get("relation") or edge[2].get("predicate") or edge[2].get("label")
                if isinstance(rel, str):
                    edge_types.add(rel)
    return edge_types


def extract_predicates(triples: Iterable[Triple]) -> Set[str]:
    return {t[1] for t in triples if len(t) >= 2}


def find_event_ids(viz_dir: str) -> List[str]:
    pattern = os.path.join(viz_dir, "*_event_knowledge_graph.pkl")
    ids = []
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        ids.append(name.replace("_event_knowledge_graph.pkl", ""))
    return sorted(ids)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_shared_triples_report(
    output_path: str,
    dataset_name: str,
    event_id: str,
    shared_triples: Iterable[Triple],
    other_label: str,
) -> None:
    with open(output_path, "a") as f:
        f.write(f"[{dataset_name}] {event_id} vs {other_label}\n")
        for triple in sorted(shared_triples):
            f.write(f"  {triple}\n")
        f.write("\n")


def write_shared_predicates_report(
    output_path: str,
    dataset_name: str,
    event_id: str,
    predicates: Iterable[str],
    other_label: str,
) -> None:
    with open(output_path, "a") as f:
        f.write(f"[{dataset_name}] {event_id} vs {other_label}\n")
        for predicate in sorted(predicates):
            f.write(f"  {predicate}\n")
        f.write("\n")


def write_predicate_matched_triples_report(
    output_path: str,
    dataset_name: str,
    event_id: str,
    triples: Iterable[Triple],
    other_label: str,
) -> None:
    with open(output_path, "a") as f:
        f.write(f"[{dataset_name}] {event_id} predicates in {other_label}\n")
        for triple in sorted(triples):
            f.write(f"  {triple}\n")
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check overlap between event KGs and edge sets across USGS datasets."
    )
    parser.add_argument(
        "--datasets-root",
        default=os.path.expanduser("~/datasets"),
        help="Root folder containing usgs_LP50hzSW4096* datasets.",
    )
    parser.add_argument(
        "--pattern",
        default="usgs_LP50hzSW4096*",
        help="Dataset folder glob pattern under datasets-root.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.getcwd(), "kg_overlap_reports"),
        help="Directory for CSVs and heatmap output.",
    )
    parser.add_argument(
        "--heatmap-metric",
        choices=["count", "pct"],
        default="pct",
        help="Metric for heatmap values.",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Disable heatmap generation.",
    )
    parser.add_argument(
        "--shared-triples-report",
        action="store_true",
        help="Write text reports of shared triples and predicate overlaps.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets_root = os.path.expanduser(args.datasets_root)
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    ensure_dir(output_dir)

    dataset_paths = sorted(glob.glob(os.path.join(datasets_root, args.pattern)))
    if not dataset_paths:
        print(f"No datasets found under {datasets_root} matching {args.pattern}", file=sys.stderr)
        return 1

    event_rows: List[Dict[str, object]] = []
    dataset_rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    shared_report_path = os.path.join(output_dir, "kg_overlap_shared_triples.txt")
    shared_predicates_path = os.path.join(output_dir, "kg_overlap_shared_predicates.txt")
    predicate_matched_triples_path = os.path.join(
        output_dir, "kg_overlap_predicate_matched_triples.txt"
    )
    if args.shared_triples_report and os.path.exists(shared_report_path):
        os.remove(shared_report_path)
    if args.shared_triples_report and os.path.exists(shared_predicates_path):
        os.remove(shared_predicates_path)
    if args.shared_triples_report and os.path.exists(predicate_matched_triples_path):
        os.remove(predicate_matched_triples_path)

    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        viz_dir = os.path.join(dataset_path, "all_events_visualizations")
        meta_dir = os.path.join(dataset_path, "all_events_meta_data")
        if not os.path.isdir(viz_dir):
            warnings.append(f"Missing all_events_visualizations in {dataset_name}")
            continue

        universal_path = os.path.join(meta_dir, "UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH.pkl")
        if not os.path.exists(universal_path):
            warnings.append(f"Missing UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH.pkl in {dataset_name}")
            universal_triples: Set[Triple] = set()
        else:
            universal_triples = extract_triples(load_pickle(universal_path))

        event_ids = find_event_ids(viz_dir)
        if not event_ids:
            warnings.append(f"No event_knowledge_graph.pkl files in {dataset_name}")
            continue

        ds_counts = {
            "dataset": dataset_name,
            "num_events": len(event_ids),
            "num_usgs_json_overlap": 0,
            "num_general_info_overlap": 0,
            "num_universal_overlap": 0,
            "num_is_located_overlap": 0,
        }

        for event_id in event_ids:
            event_kg_path = os.path.join(viz_dir, f"{event_id}_event_knowledge_graph.pkl")
            usgs_edges_path = os.path.join(viz_dir, f"{event_id}_usgs_json_kg_edge_set.pkl")
            general_edges_path = os.path.join(viz_dir, f"{event_id}_general_info_kg_edge_set.pkl")
            general_kg_path = os.path.join(viz_dir, f"{event_id}_general_info_knowledge_graph.pkl")

            event_triples = extract_triples(load_pickle(event_kg_path))
            event_predicates = extract_predicates(event_triples)

            if os.path.exists(usgs_edges_path):
                usgs_edge_types = extract_edge_types(load_pickle(usgs_edges_path))
            else:
                usgs_edge_types = set()
                warnings.append(f"Missing usgs_json_kg_edge_set.pkl for {event_id} in {dataset_name}")

            if os.path.exists(general_edges_path):
                general_edge_types = extract_edge_types(load_pickle(general_edges_path))
            else:
                general_edge_types = set()
                warnings.append(f"Missing general_info_kg_edge_set.pkl for {event_id} in {dataset_name}")

            overlap_usgs = event_predicates.intersection(usgs_edge_types)
            overlap_general = event_predicates.intersection(general_edge_types)
            overlap_universal = event_triples.intersection(universal_triples)

            if os.path.exists(general_kg_path):
                general_triples = extract_triples(load_pickle(general_kg_path))
            else:
                general_triples = set()
                warnings.append(f"Missing general_info_knowledge_graph.pkl for {event_id} in {dataset_name}")

            overlap_general_triples = event_triples.intersection(general_triples)
            predicate_matched_general_triples = {
                t for t in event_triples if t[1] in general_edge_types
            }
            event_is_located = {t for t in event_triples if t[1].startswith("is located ")}
            general_is_located = {t for t in general_triples if t[1].startswith("is located ")}
            overlap_is_located = event_is_located.intersection(general_is_located)

            has_usgs = len(overlap_usgs) > 0
            has_general = len(overlap_general) > 0
            has_universal = len(overlap_universal) > 0
            has_is_located = len(overlap_is_located) > 0

            if has_usgs:
                ds_counts["num_usgs_json_overlap"] += 1
            if has_general:
                ds_counts["num_general_info_overlap"] += 1
            if has_universal:
                ds_counts["num_universal_overlap"] += 1
            if has_is_located:
                ds_counts["num_is_located_overlap"] += 1

            if args.shared_triples_report:
                if overlap_general:
                    write_shared_predicates_report(
                        shared_predicates_path,
                        dataset_name,
                        event_id,
                        overlap_general,
                        "general_info_kg_edge_set",
                    )
                if predicate_matched_general_triples:
                    write_predicate_matched_triples_report(
                        predicate_matched_triples_path,
                        dataset_name,
                        event_id,
                        predicate_matched_general_triples,
                        "general_info_kg_edge_set",
                    )
                if overlap_general_triples:
                    write_shared_triples_report(
                        shared_report_path,
                        dataset_name,
                        event_id,
                        overlap_general_triples,
                        "general_info_knowledge_graph",
                    )
                if overlap_universal:
                    write_shared_triples_report(
                        shared_report_path,
                        dataset_name,
                        event_id,
                        overlap_universal,
                        "UNIVERSAL_GENERAL_INFO_KNOWLEDGE_GRAPH",
                    )

            event_rows.append(
                {
                    "dataset": dataset_name,
                    "event_id": event_id,
                    "event_triple_count": len(event_triples),
                    "event_predicate_count": len(event_predicates),
                    "event_is_located_triple_count": len(event_is_located),
                    "general_is_located_triple_count": len(general_is_located),
                    "usgs_json_edge_type_count": len(usgs_edge_types),
                    "general_info_edge_type_count": len(general_edge_types),
                    "overlap_usgs_json_predicate_count": len(overlap_usgs),
                    "overlap_general_info_predicate_count": len(overlap_general),
                    "overlap_general_info_triple_count": len(overlap_general_triples),
                    "overlap_general_info_predicate_matched_triple_count": len(
                        predicate_matched_general_triples
                    ),
                    "overlap_universal_triple_count": len(overlap_universal),
                    "overlap_is_located_triple_count": len(overlap_is_located),
                    "has_usgs_json_overlap": has_usgs,
                    "has_general_info_overlap": has_general,
                    "has_universal_overlap": has_universal,
                    "has_is_located_overlap": has_is_located,
                }
            )

        dataset_rows.append(ds_counts)

    event_csv_path = os.path.join(output_dir, "kg_overlap_events.csv")
    dataset_csv_path = os.path.join(output_dir, "kg_overlap_datasets.csv")

    if event_rows:
        with open(event_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(event_rows[0].keys()))
            writer.writeheader()
            writer.writerows(event_rows)

    if dataset_rows:
        with open(dataset_csv_path, "w", newline="") as f:
            fieldnames = [
                "dataset",
                "num_events",
                "num_usgs_json_overlap",
                "num_general_info_overlap",
                "num_universal_overlap",
                "num_is_located_overlap",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset_rows)

    if warnings:
        warn_path = os.path.join(output_dir, "kg_overlap_warnings.txt")
        with open(warn_path, "w") as f:
            f.write("\n".join(warnings))

    if not args.no_heatmap:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            if dataset_rows:
                datasets = [row["dataset"] for row in dataset_rows]
                if args.heatmap_metric == "count":
                    values = [
                        [
                            row["num_usgs_json_overlap"],
                            row["num_general_info_overlap"],
                            row["num_universal_overlap"],
                        ]
                        for row in dataset_rows
                    ]
                else:
                    values = []
                    for row in dataset_rows:
                        num_events = row["num_events"] or 1
                        values.append(
                            [
                                row["num_usgs_json_overlap"] / num_events,
                                row["num_general_info_overlap"] / num_events,
                                row["num_universal_overlap"] / num_events,
                            ]
                        )

                fig, ax = plt.subplots(figsize=(8, max(4, len(datasets) * 0.4)))
                im = ax.imshow(values, aspect="auto", cmap="viridis")
                ax.set_yticks(range(len(datasets)))
                ax.set_yticklabels(datasets)
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(
                    ["usgs_json", "general_info", "universal"], rotation=30, ha="right"
                )
                ax.set_title(
                    "Event KG Overlap (" + ("count" if args.heatmap_metric == "count" else "pct") + ")"
                )
                fig.colorbar(im, ax=ax, shrink=0.8)
                fig.tight_layout()
                heatmap_path = os.path.join(output_dir, "kg_overlap_heatmap.png")
                fig.savefig(heatmap_path, dpi=200)
                plt.close(fig)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Heatmap skipped: {exc}", file=sys.stderr)

    print(f"Wrote event CSV: {event_csv_path}")
    print(f"Wrote dataset CSV: {dataset_csv_path}")
    if warnings:
        print(f"Wrote warnings: {os.path.join(output_dir, 'kg_overlap_warnings.txt')}")
    if not args.no_heatmap:
        print(f"Wrote heatmap: {os.path.join(output_dir, 'kg_overlap_heatmap.png')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
