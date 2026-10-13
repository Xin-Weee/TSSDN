import argparse
import json
import os
from pathlib import Path

import numpy as np


ROOT_DIR = str(Path(__file__).resolve().parents[1])
FIXED_THRESHOLD = 0.5
PATH_KEYS = ("processed_dir", "feature_root", "aum_root", "output_root", "split_file")


def configure_parser(parser, argv=None):
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        config_path = Path(preliminary.config).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        known = {a.dest for a in parser._actions}
        unknown = set(config) - known
        if unknown:
            parser.error("Unknown configuration keys: " + ", ".join(sorted(unknown)))
        for key in PATH_KEYS:
            if config.get(key):
                config[key] = str((config_path.parent / config[key]).resolve())
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if args.model != "TSSDN":
        parser.error("model must be TSSDN.")
    if args.config:
        args.config = str(Path(args.config).resolve())
    if len(args.years) != 4 or len(set(args.years)) != 4:
        parser.error("Exactly four distinct ordered years are required.")
    if args.num_classes != 2 or args.node_feat_dim != 32:
        parser.error("TSSDN requires binary labels and 32-dimensional node features.")
    if args.hidden_dim < 2 or args.batch_size < 1 or args.epochs < 1:
        parser.error("hidden_dim >= 2, batch_size >= 1 and epochs >= 1 are required.")
    if not 0 <= args.num_workers or args.save_every < 1 or args.patience < 1:
        parser.error("Invalid worker count, save interval or patience.")
    if not (0 < args.lr and args.weight_decay >= 0 and args.grad_clip > 0):
        parser.error("Invalid optimizer settings.")
    if not (0 <= args.dropout < 1 and 0 <= args.smoothing < 1 and 0 <= args.alpha <= 1 and args.gamma >= 0):
        parser.error("Invalid dropout or loss settings.")
    if not 0 <= args.threshold <= 1:
        parser.error("Threshold must be between zero and one.")
    if any(r <= 0 for r in (args.train_ratio, args.val_ratio, args.test_ratio)) or not np.isclose(
        args.train_ratio + args.val_ratio + args.test_ratio, 1):
        parser.error("Positive split ratios summing to one are required.")
    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train TSSDN.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--model", choices=("TSSDN",), default="TSSDN")
    parser.add_argument("--feature-root", default=os.path.join(ROOT_DIR, "data", "custom", "features"))
    parser.add_argument("--aum-root", default=os.path.join(ROOT_DIR, "data", "custom", "aum"))
    parser.add_argument("--processed-dir", default=os.path.join(ROOT_DIR, "data", "custom", "processed"))
    parser.add_argument("--output-root", default=os.path.join(ROOT_DIR, "runs"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--node-feat-dim", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--threshold", type=float, default=FIXED_THRESHOLD)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--years", nargs="+", default=["2022", "2023", "2024", "2025"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--track-grad-norm", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA.")
    parser.add_argument("--test-every", type=int, default=1, help="Evaluate test set every N epochs. 0 means only evaluate at the end.")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return configure_parser(parser, argv)
