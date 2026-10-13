import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.TSSDN import TSSDNClassifier
from utils.config import parse_args as training_args
from utils.losses import FocalLossWithSmoothing
from utils.metrics import evaluate_graph_loader
from utils.dataloder import build_graph_dataset_bundle
from utils.reproducibility import export_metrics, record_run, resolve_device, seed_everything, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate a TSSDN checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--output-dir", required=True, help="New directory; existing output is never overwritten.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=None, help="Explicit exploratory override; recorded in report.")
    cli = parser.parse_args()
    args = training_args(["--config", cli.config, "--device", cli.device, "--batch-size", str(cli.batch_size)])
    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=True)
    for key in ("model_name", "node_feat_dim", "hidden_dim", "dropout", "num_classes"):
        actual = checkpoint.get(key)
        expected = getattr(args, "model" if key == "model_name" else key)
        if actual is not None and actual != expected:
            raise ValueError("Checkpoint/config mismatch: {} ({} != {})".format(key, actual, expected))
    threshold = float(checkpoint.get("thr", args.threshold)) if cli.threshold is None else cli.threshold
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1].")
    model = TSSDNClassifier(args.node_feat_dim, args.hidden_dim, args.num_classes, args.dropout).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    bundle = build_graph_dataset_bundle(
        args.processed_dir, args.feature_root, args.aum_root, args.seed,
        years=args.years, node_feat_dim=args.node_feat_dim,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        split_file=args.split_file)
    if cli.split == "all":
        from utils.dataloder import GraphChangeDataset
        names = bundle.all_patch_names
        dataset = GraphChangeDataset(names, bundle.all_labels, args.feature_root, args.aum_root,
                                              years=args.years, node_feat_dim=args.node_feat_dim)
    else:
        dataset = getattr(bundle, cli.split + "_dataset")
        names = getattr(bundle, cli.split + "_names")
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    record = record_run(args, output)
    criterion = FocalLossWithSmoothing(args.alpha, args.gamma, args.smoothing)
    loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False, num_workers=0)
    metrics = evaluate_graph_loader(model, loader, criterion, device, threshold=threshold, use_amp=False)
    report = export_metrics(metrics, names, output, prefix=cli.split)
    report.update({
        "split": cli.split, "threshold": threshold,
        "threshold_override": cli.threshold is not None, "checkpoint_sha256": sha256(cli.checkpoint),
        "input_sha256": record["inputs"]["ordered_input_sha256"],
        "interpretation": "Held-out test" if cli.split == "test" else "Not a held-out test score",
    })
    write_json(output / (cli.split + "_metrics.json"), report)
    print(report)


if __name__ == "__main__":
    main()
