import torch
import torch.nn as nn

from .gnn import *
from .regvd import build_graph


class DevignForSequence(nn.Module):

    def __init__(self, transformer, config, tokenizer,
                 feature_size=768, hidden_size=256, window_size=5,
                 num_classes=2, num_gnn_layers=2, fp16=False):
        super(DevignForSequence, self).__init__()
        self.transformer = transformer
        self.config = config
        self.tokenizer = tokenizer

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_classes = num_classes
        self.num_gnn_layers = num_gnn_layers

        self.fp16 = fp16

        self.w_embeddings = self.transformer.roberta.embeddings.word_embeddings \
            .weight.data.clone().detach().to('cpu', torch.float16).numpy()
        self.tokenizer = tokenizer

        self.gnn = GatedGNN(feature_size=feature_size,
                            hidden_size=hidden_size,
                            num_gnn_layers=num_gnn_layers,
                            dropout=config.hidden_dropout_prob)

        self.conv_l1 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 3),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2)
        )
        self.conv_l2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )

        self.concat_dim = feature_size + hidden_size
        self.conv_l1_for_concat = nn.Sequential(
            nn.Conv1d(self.concat_dim, self.concat_dim, 3),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2)
        )
        self.conv_l2_for_concat = nn.Sequential(
            nn.Conv1d(self.concat_dim, self.concat_dim, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=self.num_classes)
        self.mlp_y = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, labels=None):
        # construct graph
        x_adj, x_feature = build_graph(input_ids.detach().cpu().numpy(), self.w_embeddings)

        # initilization
        x_adj, adj_mask = preprocess_adjs(x_adj)
        adj_feature = preprocess_features(x_feature)
        x_adj = torch.from_numpy(x_adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)

        # GNN layer
        device = next(self.parameters()).device
        dtype = torch.float16 if self.fp16 else torch.float32
        outputs = self.gnn(adj_feature.to(device, dtype),
                           x_adj.to(device, dtype),
                           adj_mask.to(device, dtype)).to(dtype)
        # Conv layer
        Y_1 = self.conv_l1(outputs.transpose(1, 2))
        Y_2 = self.conv_l2(Y_1).transpose(1, 2)
        # Conv layer (after concat)
        concat = torch.cat((outputs, adj_feature.to(device, dtype)), dim=-1)
        Z_1 = self.conv_l1_for_concat(concat.transpose(1, 2))
        Z_2 = self.conv_l2_for_concat(Z_1).transpose(1, 2)
        # Dense layer
        logits = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2)).mean(dim=1)
        prob = self.sigmoid(logits)

        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, logits
        else:
            return logits
