#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _iter_samples(data_loader, max_samples):
    seen = 0
    for batch in data_loader:
        for sample in batch:
            yield sample
            seen += 1
            if max_samples is not None and seen >= max_samples:
                return


def _format_prompt(sample):
    lines = []
    pre_prompt = sample.get("pre_prompt", "")
    if pre_prompt:
        lines.append(pre_prompt)
    ts_texts = sample.get("time_series_text") or []
    if ts_texts:
        lines.append("Time series descriptions:")
        for idx, text in enumerate(ts_texts, start=1):
            lines.append(f"  {idx}. {text}")
    post_prompt = sample.get("post_prompt", "")
    if post_prompt:
        lines.append(post_prompt)
    return "\n".join(lines).strip()


def _print_split(split_name, data_loader, max_samples):
    print(f"=== {split_name} samples ===")
    for idx, sample in enumerate(_iter_samples(data_loader, max_samples), start=1):
        prompt_text = _format_prompt(sample)
        answer_text = sample.get("answer", "")
        print(f"\n--- {split_name} sample {idx} ---")
        print("Prompt:")
        print(prompt_text)
        print("Answer:")
        print(answer_text, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load train/val HDF5 datasets with get_text_time_series_hdf5_data_loader "
            "and print each sample's prompt/input text and target answer."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to a fold directory containing train/val HDF5 files.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="train_data.h5",
        help="Train HDF5 filename inside --data-dir.",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="val_data.h5",
        help="Validation HDF5 filename inside --data-dir.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (default: <data-dir>/print_dataset_prompts.log).",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default="</s>",
        help="EOS token used by the dataset when appending missing EOS to answers.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the DataLoader.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=4,
        help="Patch size used by the collate function.",
    )
    parser.add_argument(
        "--time-series-sample-rt",
        type=int,
        default=50,
        help="Time-series sampling rate (Hz).",
    )
    parser.add_argument(
        "--include-gen-knowledge",
        action="store_true",
        help="Include general knowledge in the post prompt.",
    )
    parser.add_argument(
        "--no-condense-ts-descriptions",
        action="store_true",
        help="Keep full time-series descriptions instead of condensed stats.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the number of samples printed per split.",
    )
    args = parser.parse_args()

    data_dir = Path(os.path.expanduser(args.data_dir)).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    log_file = args.log_file
    if log_file is None:
        log_file = str(data_dir / "print_dataset_prompts.log")

    train_path = data_dir / args.train_file
    val_path = data_dir / args.val_file

    _ensure_repo_on_path()
    from usgs_data.text_time_series_dataset import (  # pylint: disable=import-error
        get_text_time_series_hdf5_data_loader,
    )

    train_loader, _ = get_text_time_series_hdf5_data_loader(
        str(train_path),
        log_file,
        args.eos_token,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        patch_size=args.patch_size,
        time_series_sample_rt=args.time_series_sample_rt,
        include_gen_knowledge=args.include_gen_knowledge,
        condense_ts_descriptions=not args.no_condense_ts_descriptions,
    )
    val_loader, _ = get_text_time_series_hdf5_data_loader(
        str(val_path),
        log_file,
        args.eos_token,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        patch_size=args.patch_size,
        time_series_sample_rt=args.time_series_sample_rt,
        include_gen_knowledge=args.include_gen_knowledge,
        condense_ts_descriptions=not args.no_condense_ts_descriptions,
    )

    _print_split("train", train_loader, args.max_samples)
    _print_split("val", val_loader, args.max_samples)


if __name__ == "__main__":
    main()
