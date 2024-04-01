import torch
import torch.nn as nn
import torch.nn.functional as F

from models.old.gnn import *
from utils import preprocess_features, preprocess_adj


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = outputs
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class PredictionClassification(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args, input_size=None):
        super().__init__()
        # self.dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        if input_size is None:
            input_size = args.hidden_size
        self.dense = nn.Linear(input_size, args.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(args.hidden_size, args.num_classes)

    def forward(self, features):  #
        x = features
        x = self.dropout(x)
        x = self.dense(x.float())
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ReGVD(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(ReGVD, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings \
            .weight.data.clone().detach().cpu().numpy()
        self.tokenizer = tokenizer
        if args.gnn == "ResGatedGNN":
            self.gnn = ResGatedGNN(feature_dim_size=args.feature_dim_size,
                                   hidden_size=args.hidden_size,
                                   num_GNN_layers=args.num_GNN_layers,
                                   dropout=config.hidden_dropout_prob,
                                   residual=not args.remove_residual,
                                   att_op=args.att_op)
        else:
            self.gnn = ResGCN(feature_dim_size=args.feature_dim_size,
                              hidden_size=args.hidden_size,
                              num_GNN_layers=args.num_GNN_layers,
                              dropout=config.hidden_dropout_prob,
                              residual=not args.remove_residual,
                              att_op=args.att_op)
        gnn_out_dim = self.gnn.out_dim
        self.classifier = PredictionClassification(config, args, input_size=gnn_out_dim)

    def forward(self, input_ids=None, labels=None):
        # construct graph
        if self.args.format == "uni":
            adj, x_feature = build_graph(input_ids.detach().cpu().numpy(),
                                         self.w_embeddings, window_size=self.args.window_size)
        else:
            adj, x_feature = build_graph_text(input_ids.detach().cpu().numpy(),
                                              self.w_embeddings, window_size=self.args.window_size)
        # initilizatioin
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)
        # run over GNNs
        outputs = self.gnn(adj_feature.to(self.args.device, torch.float64),
                           adj.to(self.args.device, torch.float64),
                           adj_mask.to(self.args.device, torch.float64))
        logits = self.classifier(outputs)
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class Devign(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Devign, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings \
            .weight.data.clone().detach().cpu().numpy()
        self.tokenizer = tokenizer

        self.gnn = GatedGNN(feature_dim_size=args.feature_dim_size,
                            hidden_size=args.hidden_size,
                            num_GNN_layers=args.num_GNN_layers,
                            num_classes=args.num_classes,
                            dropout=config.hidden_dropout_prob)

        self.conv_l1 = torch.nn.Conv1d(args.hidden_size, args.hidden_size, 3).to(torch.float64)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2).to(torch.float64)
        self.conv_l2 = torch.nn.Conv1d(args.hidden_size, args.hidden_size, 1).to(torch.float64)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2).to(torch.float64)

        self.concat_dim = args.feature_dim_size + args.hidden_size
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3).to(torch.float64)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2).to(torch.float64)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1).to(torch.float64)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2).to(torch.float64)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=args.num_classes).to(torch.float64)
        self.mlp_y = nn.Linear(in_features=args.hidden_size, out_features=args.num_classes).to(torch.float64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids=None, labels=None):
        # construct graph
        if self.args.format == "uni":
            adj, x_feature = build_graph(input_ids.detach().cpu().numpy(), self.w_embeddings)
        else:
            adj, x_feature = build_graph_text(input_ids.detach().cpu().numpy(), self.w_embeddings)
        # initilization
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature).to(self.args.device).to(torch.float64)
        # run over GGGN
        outputs = self.gnn(
            adj_feature.to(self.args.device).to(torch.float64),
            adj.to(self.args.device).to(torch.float64),
            adj_mask.to(self.args.device).to(torch.float64)).to(torch.float64)
        #
        c_i = torch.cat((outputs, adj_feature), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(nn.functional.relu(self.conv_l1(outputs.transpose(1, 2))))
        Y_2 = self.maxpool2(nn.functional.relu(self.conv_l2(Y_1))).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(nn.functional.relu(self.conv_l1_for_concat(c_i.transpose(1, 2))))
        Z_2 = self.maxpool2_for_concat(nn.functional.relu(self.conv_l2_for_concat(Z_1))).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        prob = self.sigmoid(avg)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
