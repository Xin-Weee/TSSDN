import torch
import torch.nn as nn


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.5, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        log_pt = torch.log_softmax(inputs, dim=1)
        target_smooth = torch.full_like(inputs, self.smoothing / (inputs.size(1) - 1))
        target_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        ce_loss = -(target_smooth * log_pt).sum(dim=1)
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
