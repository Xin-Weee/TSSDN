import argparse
import csv
from pathlib import Path
from utils.dataloder import load_processed_arrays, split_patch_data


def main():
    parser = argparse.ArgumentParser(description="Create a stratified train/validation/test split.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    names, labels = load_processed_arrays(args.processed_dir)
    split = split_patch_data(names, labels, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "order", "patch_name", "label"])
        for part in ("train", "val", "test"):
            for i, (name, label) in enumerate(zip(split[part + "_names"], split[part + "_labels"])):
                writer.writerow([part, i, name, int(label)])
    print(output)


if __name__ == "__main__":
    main()
