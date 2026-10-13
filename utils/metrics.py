import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score


def summarize_binary_metrics(labels, probs, loss, threshold):
    preds = (probs > threshold).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    kappa = cohen_kappa_score(labels, preds)
    penalty = min(precision, recall) / max(precision, recall) if max(precision, recall) > 0 else 0.0
    score = f1 * penalty

    return {
        "loss": loss,
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "kappa": kappa,
        "score": score,
        "labels": labels,
        "preds": preds,
        "probs": probs,
    }


def evaluate_graph_loader(model, loader, criterion, device, threshold=0.5, use_amp=False):

    model.eval()
    total_loss = 0.0
    probs, labels = [], []

    with torch.no_grad():
        for x, aum, y in loader:
            x = x.to(device, non_blocking=True)
            aum = aum.to(device, non_blocking=True)
            y = y.squeeze(-1).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(x, aum)
                loss = criterion(output, y)
            total_loss += loss.item()
            probs.extend(torch.softmax(output, dim=1)[:, 1].cpu().numpy())
            labels.extend(y.cpu().numpy())

    return summarize_binary_metrics(np.array(labels), np.array(probs), total_loss / len(loader), threshold)
