import csv
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

import torch

from models.TSSDN import TSSDNClassifier
from utils.config import FIXED_THRESHOLD, parse_args
from utils.losses import FocalLossWithSmoothing
from utils.metrics import evaluate_graph_loader
from utils.dataloder import build_dataloaders, build_graph_dataset_bundle
from utils.reproducibility import seed_everything, resolve_device, record_run, export_metrics, save_split_bundle


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def training_parameters(model):
    scale = model.encoder.grf.scale
    return [scale] + [param for param in model.parameters() if param is not scale]


def compute_grad_norm(model):
    total_norm = 0.0
    for param in training_parameters(model):
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def train_graph_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp=False, track_grad_norm=False, grad_clip=0.05):

    model.train()
    train_loss = 0.0
    epoch_grad_norm = 0.0

    for x, aum, y in train_loader:
        x = x.to(device, non_blocking=True)
        aum = aum.to(device, non_blocking=True)
        y = y.squeeze(-1).to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(x, aum)
            loss = criterion(output, y)
        scaler.scale(loss).backward()

        total_norm = compute_grad_norm(model) if track_grad_norm else 0.0
        epoch_grad_norm += total_norm

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(training_parameters(model), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    return train_loss / len(train_loader), epoch_grad_norm / len(train_loader)


def save_history_csv(history, csv_path):
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "test_loss",
        "grad_norm",
        "val_acc",
        "val_kappa",
        "val_f1",
        "val_precision",
        "val_recall",
        "val_score",
        "test_acc",
        "test_kappa",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_score",
        "threshold",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def build_run_dir(root_dir, model_name):
    run_name = f"{model_name}_{datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')}"
    run_dir = os.path.join(root_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def print_dataset_summary(train_dataset, val_dataset, test_dataset):
    train_pos = int(sum(train_dataset.labels == 1))
    val_pos = int(sum(val_dataset.labels == 1))
    test_pos = int(sum(test_dataset.labels == 1))
    print(
        f"Train: {len(train_dataset)}(change:{train_pos}) | "
        f"Val: {len(val_dataset)}(change:{val_pos}) | "
        f"Test: {len(test_dataset)}(change:{test_pos})"
    )


def print_run_configuration(args, device, use_amp, run_dir):
    print("Runtime configuration:")
    print(f"  device: {device}")
    print(f"  run_dir: {run_dir}")
    print(f"  amp_enabled: {use_amp}")
    print(f"  decision_threshold: {args.threshold:.2f}")
    print("Arguments:")
    for key, value in vars(args).items():
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item) for item in value)
        print(f"  {key}: {value}")


def build_training_components(args, device):
    model = TSSDNClassifier(args.node_feat_dim, args.hidden_dim, args.num_classes, args.dropout).to(device)
    criterion = FocalLossWithSmoothing(alpha=args.alpha, gamma=args.gamma, smoothing=args.smoothing)
    optimizer = torch.optim.AdamW(training_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)
    return model, criterion, optimizer, scheduler


def build_checkpoint_payload(model, args, threshold):
    return {
        "model": model.state_dict(),
        "thr": threshold,
        "model_name": args.model,
        "hidden_dim": args.hidden_dim,
        "node_feat_dim": args.node_feat_dim,
        "dropout": args.dropout,
        "num_classes": args.num_classes,
        "years": args.years,
        "seed": args.seed,
    }


def build_empty_test_metrics():
    return {
        "loss": float("nan"),
        "acc": float("nan"),
        "kappa": float("nan"),
        "f1": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "score": float("nan"),
    }


def run_training_loop(args, device, use_amp, scaler, run_dir, csv_path, best_model_path, dataset_bundle, loader_bundle):
    model, criterion, optimizer, scheduler = build_training_components(args, device)
    best_f1_global = -1.0
    run_threshold = args.threshold
    best_thr = run_threshold
    patience_counter = 0
    history = []

    print_dataset_summary(dataset_bundle.train_dataset, dataset_bundle.val_dataset, dataset_bundle.test_dataset)
    print(model)
    print(
        f"{'Epoch':<6} {'T_Loss':<8} {'V_Loss':<8} {'Te_Loss':<8} {'Grad':<8} "
        f"{'V_Acc':<8} {'V_Kap':<8} {'V_F1':<8} {'V_P':<8} {'V_R':<8} "
        f"{'Te_Acc':<8} {'Te_Kap':<8} {'Te_F1':<8} {'Te_P':<8} {'Te_R':<8} {'Thr':<6}"
    )
    print("-" * 140)

    for epoch in range(args.epochs):
        train_loss, epoch_grad_norm = train_graph_epoch(
            model=model,
            train_loader=loader_bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            track_grad_norm=args.track_grad_norm,
            grad_clip=args.grad_clip,
        )

        val_metrics = evaluate_graph_loader(model, loader_bundle.val_loader, criterion, device, threshold=run_threshold, use_amp=use_amp)
        if args.test_every > 0 and (epoch + 1) % args.test_every == 0:
            test_metrics = evaluate_graph_loader(model, loader_bundle.test_loader, criterion, device, threshold=run_threshold, use_amp=use_amp)
        else:
            test_metrics = build_empty_test_metrics()

        print(
            f"{epoch + 1:<6} {train_loss:<8.4f} {val_metrics['loss']:<8.4f} {test_metrics['loss']:<8.4f} "
            f"{epoch_grad_norm:<8.4f} {val_metrics['acc']:<8.4f} {val_metrics['kappa']:<8.4f} "
            f"{val_metrics['f1']:<8.4f} {val_metrics['precision']:<8.4f} {val_metrics['recall']:<8.4f} "
            f"{test_metrics['acc']:<8.4f} {test_metrics['kappa']:<8.4f} {test_metrics['f1']:<8.4f} "
            f"{test_metrics['precision']:<8.4f} {test_metrics['recall']:<8.4f} {run_threshold:<6.2f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "test_loss": test_metrics["loss"],
                "grad_norm": epoch_grad_norm,
                "val_acc": val_metrics["acc"],
                "val_kappa": val_metrics["kappa"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_score": val_metrics["score"],
                "test_acc": test_metrics["acc"],
                "test_kappa": test_metrics["kappa"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_score": test_metrics["score"],
                "threshold": run_threshold,
            }
        )

        if val_metrics["f1"] >= best_f1_global:
            best_f1_global = val_metrics["f1"]
            best_thr = run_threshold
            torch.save(build_checkpoint_payload(model, args, best_thr), best_model_path)
            print(f"Epoch {epoch + 1}: best updated! Val F1: {val_metrics['f1']:.4f} (Thr: {best_thr:.2f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % args.save_every == 0:
            epoch_ckpt = os.path.join(run_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(build_checkpoint_payload(model, args, run_threshold), epoch_ckpt)
            print(f"Epoch {epoch + 1}: checkpoint saved -> {epoch_ckpt}")

        scheduler.step(val_metrics["f1"])
        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    save_history_csv(history, csv_path)

    best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model"])
    best_thr = float(best_checkpoint.get("thr", FIXED_THRESHOLD))
    best_test_metrics = evaluate_graph_loader(model, loader_bundle.test_loader, criterion, device, threshold=best_thr, use_amp=use_amp)


    print("\nBest checkpoint test results:")
    print(f"Best threshold: {best_thr:.2f}")
    print(f"Test Loss: {best_test_metrics['loss']:.4f}")
    print(f"Test Acc: {best_test_metrics['acc']:.4f}")
    print(f"Test Kappa: {best_test_metrics['kappa']:.4f}")
    print(f"Test F1: {best_test_metrics['f1']:.4f}")
    print(f"Test Precision: {best_test_metrics['precision']:.4f}")
    print(f"Test Recall: {best_test_metrics['recall']:.4f}")

    print("\nSaved artifacts:")
    print(f"Metrics CSV: {csv_path}")
    print(f"Best model: {best_model_path}")
    export_metrics(best_test_metrics, dataset_bundle.test_names, run_dir)
    return best_test_metrics


def run_graph_training(args, device, use_amp, scaler, run_dir, csv_path, best_model_path):
    print_run_configuration(args=args, device=device, use_amp=use_amp, run_dir=run_dir)
    record_run(args, run_dir)
    dataset_bundle = build_graph_dataset_bundle(
        processed_dir=args.processed_dir,
        feature_root=args.feature_root,
        aum_root=args.aum_root,
        random_seed=args.seed,
        years=args.years,
        num_nodes=6,
        node_feat_dim=args.node_feat_dim,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_file=args.split_file,
    )
    print(f"Base samples: {len(dataset_bundle.all_patch_names)}")
    save_split_bundle(dataset_bundle, os.path.join(run_dir, "split.csv"))

    loader_bundle = build_dataloaders(
        train_dataset=dataset_bundle.train_dataset,
        val_dataset=dataset_bundle.val_dataset,
        test_dataset=dataset_bundle.test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_type=device.type,
        drop_last_train=True,
    )
    return run_training_loop(args, device, use_amp, scaler, run_dir, csv_path, best_model_path, dataset_bundle, loader_bundle)


def train(args):
    seed_everything(args.seed, args.deterministic)

    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_root, exist_ok=True)
    run_dir = build_run_dir(args.output_root, args.model)
    log_path = os.path.join(run_dir, "train.log")
    csv_path = os.path.join(run_dir, "metrics.csv")
    best_model_path = os.path.join(run_dir, "best_model_f1.pth")

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"Log: {log_path}")
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            run_graph_training(args, device, use_amp, scaler, run_dir, csv_path, best_model_path)


if __name__ == "__main__":
    train(parse_args())
