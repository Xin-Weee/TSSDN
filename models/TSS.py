import torch
import torch.nn as nn


def normalize_adjacency(adj):
    identity = torch.eye(adj.size(1), device=adj.device, dtype=adj.dtype).unsqueeze(0)
    normalized_adj = adj + identity
    degree = torch.sum(normalized_adj, dim=2)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    degree_matrix = torch.diag_embed(degree_inv_sqrt)
    return degree_matrix @ normalized_adj @ degree_matrix


class TSSGraphConvolution(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, node_feat_dim: int, bias: float = 0.0):
        super().__init__()
        self.weights = nn.Parameter(torch.FloatTensor(num_gru_units + node_feat_dim, output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(output_dim))
        self.bias_init = bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self.bias_init)

    def forward(self, inputs, hidden_state, adj):
        concatenated = torch.cat((inputs, hidden_state), dim=2)
        normalized_adj = normalize_adjacency(adj)
        propagated = torch.bmm(normalized_adj, concatenated)
        return propagated @ self.weights + self.biases


class TSSRecurrentCell(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_reset_conv = TSSGraphConvolution(hidden_dim, hidden_dim * 2, node_feat_dim, bias=1.0)
        self.candidate_conv = TSSGraphConvolution(hidden_dim, hidden_dim, node_feat_dim, bias=0.0)

    def forward(self, inputs, hidden_state, adj):
        gates = torch.sigmoid(self.update_reset_conv(inputs, hidden_state, adj))
        reset_gate, update_gate = torch.chunk(gates, chunks=2, dim=2)
        candidate_state = torch.tanh(self.candidate_conv(inputs, reset_gate * hidden_state, adj))
        new_hidden_state = update_gate * hidden_state + (1.0 - update_gate) * candidate_state
        return new_hidden_state


class TSS(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = TSSRecurrentCell(node_feat_dim, hidden_dim)

    def forward(self, inputs, adjs):
        batch_size, seq_len, num_nodes, _ = inputs.shape
        hidden_state = torch.zeros(
            batch_size,
            num_nodes,
            self.hidden_dim,
            device=inputs.device,
            dtype=inputs.dtype,
        )

        for time_step in range(seq_len):
            hidden_state = self.cell(inputs[:, time_step], hidden_state, adjs[:, time_step])
        return hidden_state


__all__ = ["TSS", "TSSRecurrentCell", "TSSGraphConvolution", "normalize_adjacency"]
