import torch
import torch.nn as nn

from models.TSS import TSS
from layers.layers import (
    DLTD,
    FeatureFusion,
    GlobalDifferenceBranch,
    GRF,
    LocalDifferenceBranch,
    SymmetricReadout,
)


class TSSDN(nn.Module):


    def __init__(self, node_feat_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()

        self.tss = TSS(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim)
        self.dltd = DLTD(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.local_branch = LocalDifferenceBranch(hidden_dim=hidden_dim, dropout=dropout)
        self.global_branch = GlobalDifferenceBranch(hidden_dim)
        self.diff_fusion = FeatureFusion(hidden_dim * 2, hidden_dim, dropout)
        self.grf = GRF(hidden_dim, dropout)
        self.graph_readout = SymmetricReadout(hidden_dim, hidden_dim, dropout)

        self.dltd = DLTD(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, inputs, adjs):
        tss_node = self.tss(inputs, adjs)
        pairs = self.dltd(inputs, adjs)
        local_node = self.local_branch(pairs)
        global_node = self.global_branch(pairs["14"]["feat"], pairs["14"]["adj"])
        diff_node = self.diff_fusion(torch.cat([local_node, global_node], dim=-1))
        fused_node = self.grf(tss_node, diff_node)
        graph_features = self.graph_readout(fused_node)

        return graph_features


class TSSDNClassifier(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = TSSDN(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, aum):
        graph_features = self.encoder(x, aum)
        return self.classifier(graph_features)


__all__ = ["TSSDN", "TSSDNClassifier"]
