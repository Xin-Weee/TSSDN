import os
import csv
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

DEFAULT_YEARS = ("2022", "2023", "2024", "2025")


@dataclass


class DatasetBundle:
    all_patch_names: np.ndarray
    all_labels: np.ndarray
    train_names: np.ndarray
    train_labels: np.ndarray
    val_names: np.ndarray
    val_labels: np.ndarray
    test_names: np.ndarray
    test_labels: np.ndarray
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset


@dataclass


class DataLoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def load_processed_arrays(processed_dir):
    patch_names = np.load(os.path.join(processed_dir, "patch_names.npy"), allow_pickle=False)
    labels = np.load(os.path.join(processed_dir, "labels.npy"), allow_pickle=False)
    if patch_names.ndim != 1 or labels.ndim != 1 or len(patch_names) != len(labels) or not len(labels):
        raise ValueError("Names and labels must be nonempty aligned one-dimensional arrays.")
    if patch_names.dtype.kind not in ("U", "S"):
        raise ValueError("Patch names must be strings, not object arrays.")
    patch_names = patch_names.astype(str)
    if len(set(patch_names)) != len(patch_names):
        raise ValueError("Duplicate patch names.")
    if any(not n or n in (".", "..") or "/" in n or chr(92) in n or ":" in n for n in patch_names):
        raise ValueError("Patch names must not contain paths.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary 0 or 1.")
    return patch_names, labels.astype(np.int64)


def load_fixed_split(split_file, patch_names, labels):
    known = dict(zip(patch_names.tolist(), labels.tolist()))
    grouped = {part: [] for part in ("train", "val", "test")}
    seen = set()
    with open(split_file, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"split", "order", "patch_name", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Split CSV requires split, order, patch_name, label.")
        for row in reader:
            name = row["patch_name"]
            part = row["split"]
            if part not in grouped or name not in known or name in seen or int(row["label"]) != known[name]:
                raise ValueError("Invalid/overlapping/mismatched split entry: " + str(row))
            grouped[part].append((int(row["order"]), name, known[name]))
            seen.add(name)
    if seen != set(known):
        raise ValueError("Split does not cover every labeled sample exactly once.")
    result = {}
    for part, rows in grouped.items():
        rows.sort()
        if not rows or [r[0] for r in rows] != list(range(len(rows))):
            raise ValueError("Nonempty splits with contiguous zero-based order are required.")
        result[part + "_names"] = np.array([r[1] for r in rows])
        result[part + "_labels"] = np.array([r[2] for r in rows], dtype=np.int64)
    return result


def split_patch_data(
    patch_names,
    labels,
    random_seed,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
):
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total_ratio:.4f}"
        )

    train_names, temp_names, train_labels, temp_labels = train_test_split(
        patch_names,
        labels,
        test_size=1.0 - train_ratio,
        random_state=random_seed,
        stratify=labels,
    )
    test_size_within_temp = test_ratio / (val_ratio + test_ratio)
    val_names, test_names, val_labels, test_labels = train_test_split(
        temp_names,
        temp_labels,
        test_size=test_size_within_temp,
        random_state=random_seed,
        stratify=temp_labels,
    )
    return {
        "train_names": train_names,
        "train_labels": train_labels,
        "val_names": val_names,
        "val_labels": val_labels,
        "test_names": test_names,
        "test_labels": test_labels,
    }


def load_graph_inputs(patch_name, year, feature_root, aum_root, num_nodes, node_feat_dim):
    feature_file = os.path.join(feature_root, year, patch_name + "_features.npy")
    aum_file = os.path.join(aum_root, year, patch_name + "_aum.npy")
    node_features = np.load(feature_file, allow_pickle=False)
    node_aum = np.load(aum_file, allow_pickle=False)
    for value, expected, path in (
        (node_features, (num_nodes, node_feat_dim), feature_file),
        (node_aum, (num_nodes, num_nodes), aum_file),
    ):
        if value.shape != expected or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError("Invalid shape/type/non-finite input: " + path)
    if (node_aum < 0).any() or not np.allclose(node_aum, node_aum.T, rtol=1e-5, atol=1e-6):
        raise ValueError("AUM must be nonnegative and symmetric: " + aum_file)
    return node_features.astype(np.float32), node_aum.astype(np.float32)


def build_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    num_workers,
    device_type,
    drop_last_train=True,
):
    if drop_last_train and len(train_dataset) < batch_size:
        raise ValueError("Training set is smaller than one complete batch; reduce batch_size.")
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device_type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=drop_last_train,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_kwargs,
    )
    return DataLoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


class GraphChangeDataset(Dataset):
    def __init__(
        self,
        patch_names,
        labels,
        feature_root,
        aum_root,
        years=None,
        num_nodes=6,
        node_feat_dim=32,
    ):
        self.patch_names = np.array(patch_names)
        self.labels = np.array(labels)
        self.feature_root = feature_root
        self.aum_root = aum_root
        self.years = list(years) if years is not None else list(DEFAULT_YEARS)
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        patch_name = self.patch_names[idx]
        label = self.labels[idx]
        seq_features, seq_aums = [], []

        for year in self.years:
            node_features, node_aum = load_graph_inputs(
                patch_name, year, self.feature_root, self.aum_root, self.num_nodes, self.node_feat_dim
            )
            seq_features.append(node_features)
            seq_aums.append(node_aum)

        x_features = np.stack(seq_features, axis=0)
        x_aums = np.stack(seq_aums, axis=0)
        return torch.FloatTensor(x_features), torch.FloatTensor(x_aums), torch.LongTensor([label])


def build_graph_dataset_bundle(
    processed_dir,
    feature_root,
    aum_root,
    random_seed,
    years=None,
    num_nodes=6,
    node_feat_dim=32,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    split_file=None,
):
    years = list(years) if years is not None else list(DEFAULT_YEARS)
    all_patch_names, all_labels = load_processed_arrays(processed_dir)
    if len(years) != 4 or len(set(years)) != 4:
        raise ValueError("Exactly four ordered, distinct dates are required.")
    if split_file:
        split_data = load_fixed_split(split_file, all_patch_names, all_labels)
    else:
        split_data = split_patch_data(
            all_patch_names, all_labels, random_seed,
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)


    dataset_kwargs = {
        "feature_root": feature_root,
        "aum_root": aum_root,
        "years": years,
        "num_nodes": num_nodes,
        "node_feat_dim": node_feat_dim,
    }
    train_dataset = GraphChangeDataset(
        split_data["train_names"],
        split_data["train_labels"],
        **dataset_kwargs,
    )
    val_dataset = GraphChangeDataset(
        split_data["val_names"],
        split_data["val_labels"],
        **dataset_kwargs,
    )
    test_dataset = GraphChangeDataset(
        split_data["test_names"],
        split_data["test_labels"],
        **dataset_kwargs,
    )

    return DatasetBundle(
        all_patch_names=all_patch_names,
        all_labels=all_labels,
        train_names=split_data["train_names"],
        train_labels=split_data["train_labels"],
        val_names=split_data["val_names"],
        val_labels=split_data["val_labels"],
        test_names=split_data["test_names"],
        test_labels=split_data["test_labels"],
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )
