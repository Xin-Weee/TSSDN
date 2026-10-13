import torch
import torch.nn as nn
import torch.nn.functional as F

_FOUR_STEP_PAIR_SPECS = (
    ("12", 0, 1),
    ("23", 1, 2),
    ("34", 2, 3),
    ("14", 0, 3),
)

_LOCAL_PAIR_NAMES = ("12", "23", "34")


def normalize_difference_adjacency(adj, eps: float = 1e-6):
    adj_max = torch.max(adj.reshape(adj.size(0), -1), dim=1)[0].reshape(-1, 1, 1).clamp_min(eps)
    normalized_adj = adj / adj_max
    identity = torch.eye(adj.size(1), device=adj.device, dtype=adj.dtype).unsqueeze(0)
    normalized_adj = normalized_adj + identity

    degree = torch.sum(normalized_adj, dim=2)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    degree_matrix = torch.diag_embed(degree_inv_sqrt)
    return degree_matrix @ normalized_adj @ degree_matrix


class DifferenceGraphConvolution(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, node_feat_dim: int, bias: float = 0.0):
        super().__init__()
        self.weights = nn.Parameter(torch.FloatTensor(num_gru_units + node_feat_dim, output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(output_dim))
        self.layer_norm = nn.LayerNorm(output_dim)
        self.bias_init = bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self.bias_init)

    def forward(self, inputs, hidden_state, adj):
        concatenated = torch.cat((inputs, hidden_state), dim=2)
        normalized_adj = normalize_difference_adjacency(adj)
        propagated = torch.bmm(normalized_adj, concatenated)
        outputs = propagated @ self.weights + self.biases
        return self.layer_norm(outputs)


class DifferenceRecurrentCell(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_reset_conv = DifferenceGraphConvolution(hidden_dim, hidden_dim * 2, node_feat_dim, bias=1.0)
        self.candidate_conv = DifferenceGraphConvolution(hidden_dim, hidden_dim, node_feat_dim, bias=0.0)

    def forward(self, inputs, hidden_state, adj):
        gates = torch.sigmoid(self.update_reset_conv(inputs, hidden_state, adj))
        reset_gate, update_gate = torch.chunk(gates, chunks=2, dim=2)
        candidate_state = torch.tanh(self.candidate_conv(inputs, reset_gate * hidden_state, adj))
        return update_gate * hidden_state + (1.0 - update_gate) * candidate_state


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        normalized_adj = normalize_difference_adjacency(adj)
        h = torch.bmm(normalized_adj, x)
        h = self.linear(h)
        h = self.norm(h)
        h = F.relu(h, inplace=True)
        return self.dropout(h)


class SharedStateEncoder(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.gcn1 = GCNLayer(node_feat_dim, hidden_dim, dropout)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim, dropout)

    def forward(self, x, adj):
        h1 = self.gcn1(x, adj)
        h2 = self.gcn2(h1, adj)
        return h1 + h2


class TemporalStateEncoder(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.state_encoder = SharedStateEncoder(node_feat_dim, hidden_dim, dropout)

    def forward(self, inputs, adjs):
        if inputs.size(1) != adjs.size(1):
            raise ValueError(f"inputs/adjs temporal dimensions must match, got {inputs.size(1)} and {adjs.size(1)}")

        states = {}
        for step in range(inputs.size(1)):
            states[f"s{step + 1}"] = self.state_encoder(inputs[:, step], adjs[:, step])
        return states


class ScaledDiffPairBuilder(nn.Module):
    rel_dim = 5

    def __init__(self, diff_scale_init: float = 0.2, abs_diff_scale_init: float = 0.2):
        super().__init__()
        diff_scale_init = float(min(max(diff_scale_init, 1e-4), 1.0 - 1e-4))
        abs_diff_scale_init = float(min(max(abs_diff_scale_init, 1e-4), 1.0 - 1e-4))
        self.diff_scale_logit = nn.Parameter(torch.logit(torch.tensor(diff_scale_init)))
        self.abs_diff_scale_logit = nn.Parameter(torch.logit(torch.tensor(abs_diff_scale_init)))

    def forward(self, s_i, s_j, A_i, A_j):
        ds = s_j - s_i
        ads = torch.abs(ds)
        diff_scale = torch.sigmoid(self.diff_scale_logit)
        abs_diff_scale = torch.sigmoid(self.abs_diff_scale_logit)
        rel_dA = A_j - A_i
        adA = torch.abs(rel_dA)

        rel = torch.cat(
            [
                rel_dA.sum(dim=-1, keepdim=True),
                adA.sum(dim=-1, keepdim=True),
                adA.mean(dim=-1, keepdim=True),
                adA.max(dim=-1, keepdim=True).values,
                torch.diagonal(rel_dA, dim1=-2, dim2=-1).unsqueeze(-1),
            ],
            dim=-1,
        )
        u = torch.cat([s_i, s_j, diff_scale * ds, abs_diff_scale * ads, rel], dim=-1)
        return u, 0.5 * (A_i + A_j)


class PairEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, u):
        return self.proj(u)


class DifferenceSequence(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.cell = DifferenceRecurrentCell(hidden_dim, hidden_dim)

    def forward(self, pair_feats, pair_adjs):
        batch_size, num_nodes, hidden_dim = pair_feats[0].shape
        hidden = torch.zeros(batch_size, num_nodes, hidden_dim, device=pair_feats[0].device)
        outs = []
        for x, a in zip(pair_feats, pair_adjs):
            hidden = self.cell(x, hidden, a)
            outs.append(hidden)
        return torch.stack(outs, dim=1)


class GlobalDifferenceBranch(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.cell = DifferenceRecurrentCell(hidden_dim, hidden_dim)

    def forward(self, feat_14, adj_14):
        batch_size, num_nodes, hidden_dim = feat_14.shape
        hidden = torch.zeros(batch_size, num_nodes, hidden_dim, device=feat_14.device)
        return self.cell(feat_14, hidden, adj_14)


class FeatureFusion(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fuse(x)


class GRF(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        in_dim = hidden_dim * 3
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.correction = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, tss_node, diff_node):
        context = torch.cat([tss_node, diff_node, torch.abs(tss_node - diff_node)], dim=-1)
        gate = self.gate(context)
        correction = self.correction(context)
        scale = torch.sigmoid(self.scale)
        delta = scale * gate * correction
        return tss_node + delta


class SymmetricReadout(nn.Module):
    def __init__(self, node_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(node_dim * 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, node_features):
        mean_feat = node_features.mean(dim=1)
        max_feat = node_features.max(dim=1).values
        std_feat = node_features.std(dim=1, unbiased=False)
        graph_feat = self.proj(torch.cat([mean_feat, max_feat, std_feat], dim=-1))
        return graph_feat


class DLTD(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.state_encoder = TemporalStateEncoder(node_feat_dim, hidden_dim, dropout)
        self.pair_builder = ScaledDiffPairBuilder(diff_scale_init=0.2, abs_diff_scale_init=0.2)
        self.pair_encoder = PairEncoder(hidden_dim * 4 + self.pair_builder.rel_dim, hidden_dim, dropout)

    def build_pair(self, s_i, s_j, adj_i, adj_j):
        pair_inputs, pair_adj = self.pair_builder(s_i, s_j, adj_i, adj_j)
        return {
            "feat": self.pair_encoder(pair_inputs),
            "adj": pair_adj,
        }

    def forward(self, inputs, adjs):
        if inputs.size(1) != 4 or adjs.size(1) != 4:
            raise ValueError(f"DLTD expects 4 temporal steps, got {inputs.size(1)} and {adjs.size(1)}")

        states = self.state_encoder(inputs, adjs)
        pairs = {}
        for pair_name, start_idx, end_idx in _FOUR_STEP_PAIR_SPECS:
            pairs[pair_name] = self.build_pair(
                states[f"s{start_idx + 1}"],
                states[f"s{end_idx + 1}"],
                adjs[:, start_idx],
                adjs[:, end_idx],
            )
        return pairs


class LocalDifferenceBranch(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.temporal = DifferenceSequence(hidden_dim)
        self.proj = FeatureFusion(hidden_dim, hidden_dim, dropout)

    def forward(self, pairs):
        pair_feats = [pairs[name]["feat"] for name in _LOCAL_PAIR_NAMES]
        pair_adjs = [pairs[name]["adj"] for name in _LOCAL_PAIR_NAMES]
        local_seq = self.temporal(pair_feats, pair_adjs)
        local_seq = self.proj(local_seq)
        return local_seq.mean(dim=1)


__all__ = [
    "GRF",
    "FeatureFusion",
    "GlobalDifferenceBranch",
    "DLTD",
    "LocalDifferenceBranch",
    "ScaledDiffPairBuilder",
    "SymmetricReadout",
]
