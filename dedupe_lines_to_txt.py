#!/usr/bin/env python3
from pathlib import Path
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read a text file, deduplicate lines with a set, and write unique lines to a new .txt file."
    )
    parser.add_argument("input_file", type=Path, help="Path to the input .txt file.")
    parser.add_argument(
        "-o",
        "--output_file",
        type=Path,
        help="Path to output .txt file. Default: <input_stem>_unique.txt",
    )
    args = parser.parse_args()

    input_path = args.input_file
    output_path = args.output_file or input_path.with_name(f"{input_path.stem}_unique.txt")

    with input_path.open("r", encoding="utf-8") as f:
        unique_lines = set(line.rstrip("\n") for line in f)

    with output_path.open("w", encoding="utf-8") as f:
        for line in sorted(unique_lines):
            f.write(f"{line}\n")

    print(f"Wrote {len(unique_lines)} unique lines to: {output_path}")


if __name__ == "__main__":
    main()
