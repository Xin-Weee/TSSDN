import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu.")
    return device


def environment():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "scikit-learn")},
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def validate_inputs(args):
    from utils.dataloder import load_processed_arrays, load_graph_inputs
    names, labels = load_processed_arrays(args.processed_dir)
    h = hashlib.sha256()
    count = 0
    for name in ("patch_names.npy", "labels.npy"):
        h.update(name.encode())
        h.update(bytes.fromhex(sha256(Path(args.processed_dir) / name)))
        count += 1
    for name in names:
        for year in args.years:
            load_graph_inputs(str(name), year, args.feature_root, args.aum_root, 6, args.node_feat_dim)
            for kind, root, suffix in (("features", args.feature_root, "_features.npy"), ("aum", args.aum_root, "_aum.npy")):
                key = "{}/{}/{}{}".format(kind, year, name, suffix)
                h.update(key.encode())
                h.update(bytes.fromhex(sha256(Path(root) / year / (str(name) + suffix))))
                count += 1
    return {
        "samples": len(names), "positive_samples": int(labels.sum()),
        "input_file_count": count, "ordered_input_sha256": h.hexdigest(),
        "split_sha256": sha256(args.split_file) if args.split_file else None,
    }


def record_run(args, run_dir):
    root = Path(__file__).resolve().parents[1]
    record = {
        "arguments": vars(args), "environment": environment(),
        "inputs": validate_inputs(args),
        "code_sha256": {str(p.relative_to(root)).replace("\\", "/"): sha256(p)
                        for pattern in ("*.py", "models/*.py", "layers/*.py", "utils/*.py")
                        for p in sorted(root.glob(pattern))},
        "selection": "maximum validation F1; latest epoch wins ties",
        "decision_rule": "softmax(logits)[:, 1] > threshold",
    }
    write_json(Path(run_dir) / "run_config.json", record)
    return record


def export_metrics(metrics, patch_names, output_dir, prefix="test"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / (prefix + "_predictions.csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patch_name", "label", "probability", "prediction"])
        for row in zip(patch_names, metrics["labels"], metrics["probs"], metrics["preds"]):
            writer.writerow([str(row[0]), int(row[1]), float(row[2]), int(row[3])])
    scalar = {k: (float(v) if np.isfinite(v) else None) for k, v in metrics.items()
              if k not in ("labels", "probs", "preds")}
    scalar["sample_count"] = len(patch_names)
    write_json(output_dir / (prefix + "_metrics.json"), scalar)
    return scalar


def save_split_bundle(bundle, path):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "order", "patch_name", "label"])
        for part in ("train", "val", "test"):
            for i, (name, label) in enumerate(zip(getattr(bundle, part + "_names"), getattr(bundle, part + "_labels"))):
                writer.writerow([part, i, name, int(label)])
