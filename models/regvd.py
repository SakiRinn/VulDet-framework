import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import *


def normalize_adj(adj: sp.csr_matrix):
    """ Symmetrically normalize adjacency matrix. """
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(np.array(adj.sum(axis=1)), -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt_diag = np.diag(d_inv_sqrt)
    return adj.dot(d_inv_sqrt_diag).transpose().dot(d_inv_sqrt_diag)


def preprocess_adjs(adjs: List[sp.csr_matrix]):
    """ Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation. """
    max_length = max([a.shape[0] for a in adjs])
    mask = np.zeros((len(adjs), max_length, 1))         # mask for padding

    for i, adj in enumerate(adjs):
        adj_normalized = normalize_adj(adj)             # no self-loop
        pad = max_length - adj_normalized.shape[0]      # padding for each epoch

        adjs[i] = np.pad(adj_normalized, ((0, pad), (0, pad)), mode='constant')
        mask[i, :adj.shape[0], :] = 1.
    return np.array(adjs), mask


def preprocess_features(features: List[np.ndarray]):
    """ Row-normalize feature matrix and convert to tuple representation """
    max_length = max([len(feature) for feature in features])

    for i, feature in enumerate(features):
        feature = np.array(feature)
        pad = max_length - feature.shape[0]         # padding for each epoch

        features[i] = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
    return np.array(features)


def build_graph(sentences, word_embedding, window_size=3,
                weighted_graph=False, word_unique=True):
    adjs, features = [], []
    for sentence in sentences:
        words = list(set(sentence)) if word_unique else sentence
        word_to_inds = {words[i]: i for i in range(len(words))}

        # sliding windows
        windows = (sentence[i: i + window_size] if len(sentence) > window_size else sentence
                   for i in range(len(sentence) - window_size + 1))

        # word co-occurrences as weights
        pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    if word_unique and window[p] == window[q]:
                        continue
                    # forward
                    pair = (window[p], window[q])
                    pair_count[pair] = pair_count.get(pair, 0.) + 1.
                    # reverse
                    pair = (window[q], window[p])
                    pair_count[pair] = pair_count.get(pair, 0.) + 1.

        row, col = [], []
        weight = []
        for pair in pair_count.keys():
            p, q = pair
            row.append(word_to_inds[p])
            col.append(word_to_inds[q])
            weight.append(pair_count[pair] if weighted_graph else 1.)

        adj = sp.csr_matrix((weight, (row, col)), shape=(len(words), len(words)))
        feature = [word_embedding[v] for v in words]
        adjs.append(adj)
        features.append(feature)

    return adjs, features


class PredictionClassification(nn.Module):

    """ Head for sentence-level classification tasks. """

    def __init__(self, config,
                 input_size=-1, hidden_size=256, num_classes=2):
        super().__init__()
        if input_size < 0:
            input_size = hidden_size
        self.dense = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            # nn.Linear(hidden_size * 2, hidden_size),
            nn.Linear(input_size, hidden_size),
            nn.Tanh()
        )
        self.out_proj = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, features):
        x = features.to(torch.float32)
        x = self.dense(x)
        x = self.out_proj(x)
        return x


class ReGVD(nn.Module):

    def __init__(self, encoder, config, tokenizer,
                 feature_size=768, hidden_size=256, window_size=5,
                 num_classes=2, num_gnn_layers=2, alpha_weight=1.,
                 gnn="ResGatedGNN", att_op="mul", remove_residual=False):
        super(ReGVD, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_classes = num_classes
        self.num_gnn_layers = num_gnn_layers

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings \
            .weight.data.clone().detach().cpu().numpy()
        self.tokenizer = tokenizer
        if gnn == "ResGatedGNN":
            self.gnn = ResGatedGNN(feature_size=feature_size,
                                   hidden_size=hidden_size,
                                   num_gnn_layers=num_gnn_layers,
                                   dropout=config.hidden_dropout_prob,
                                   residual=not remove_residual,
                                   att_op=att_op)
        else:
            self.gnn = ResGCN(feature_size=feature_size,
                              hidden_size=hidden_size,
                              num_gnn_layers=num_gnn_layers,
                              dropout=config.hidden_dropout_prob,
                              residual=not remove_residual,
                              att_op=att_op)

        self.classifier = PredictionClassification(config, self.gnn.out_dim, hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, labels=None):
        # construct graph
        x_adj, x_feature = build_graph(inputs.detach().cpu().numpy(),
                                       self.w_embeddings,
                                       window_size=self.window_size)

        # initilizatioin
        x_adj, adj_mask = preprocess_adjs(x_adj)
        adj_feature = preprocess_features(x_feature)
        x_adj = torch.from_numpy(x_adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)

        # GNN layer
        device = next(self.parameters()).device
        outputs = self.gnn(adj_feature.to(device, torch.float64),
                           x_adj.to(device, torch.float64),
                           adj_mask.to(device, torch.float64))
        # Classification layer
        logits = self.classifier(outputs)
        prob = self.softmax(logits)

        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class Devign(nn.Module):

    def __init__(self, encoder, config, tokenizer,
                 feature_size=768, hidden_size=256, window_size=5,
                 num_classes=2, num_gnn_layers=2):
        super(Devign, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer

        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_classes = num_classes
        self.num_gnn_layers = num_gnn_layers

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings \
            .weight.data.clone().detach().cpu().numpy()
        self.tokenizer = tokenizer

        self.gnn = GatedGNN(feature_size=feature_size,
                            hidden_size=hidden_size,
                            num_gnn_layers=num_gnn_layers,
                            dropout=config.hidden_dropout_prob)

        self.conv_l1 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 3).to(torch.float64),
            nn.ReLU().to(torch.float64),
            nn.MaxPool1d(3, stride=2).to(torch.float64)
        )
        self.conv_l2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1).to(torch.float64),
            nn.ReLU().to(torch.float64),
            nn.MaxPool1d(2, stride=2).to(torch.float64)
        )

        self.concat_dim = feature_size + hidden_size
        self.conv_l1_for_concat = nn.Sequential(
            nn.Conv1d(self.concat_dim, self.concat_dim, 3).to(torch.float64),
            nn.ReLU().to(torch.float64),
            nn.MaxPool1d(3, stride=2).to(torch.float64)
        )
        self.conv_l2_for_concat = nn.Sequential(
            nn.Conv1d(self.concat_dim, self.concat_dim, 1).to(torch.float64),
            nn.ReLU().to(torch.float64),
            nn.MaxPool1d(2, stride=2).to(torch.float64)
        )

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=self.num_classes).to(torch.float64)
        self.mlp_y = nn.Linear(in_features=hidden_size, out_features=num_classes).to(torch.float64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, labels=None):
        # construct graph
        x_adj, x_feature = build_graph(inputs.detach().cpu().numpy(), self.w_embeddings)

        # initilization
        x_adj, adj_mask = preprocess_adjs(x_adj)
        adj_feature = preprocess_features(x_feature)
        x_adj = torch.from_numpy(x_adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)

        # GNN layer
        device = next(self.parameters()).device
        outputs = self.gnn(adj_feature.to(device, torch.float64),
                           x_adj.to(device, torch.float64),
                           adj_mask.to(device, torch.float64)).to(torch.float64)
        # Conv layer
        Y_1 = self.conv_l1(outputs.transpose(1, 2))
        Y_2 = self.conv_l2(Y_1).transpose(1, 2)
        # Conv layer (after concat)
        concat = torch.cat((outputs, adj_feature), dim=-1)
        Z_1 = self.conv_l1_for_concat(concat.transpose(1, 2))
        Z_2 = self.conv_l2_for_concat(Z_1).transpose(1, 2)
        # Dense layer
        logit = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        prob = self.sigmoid(logit.mean(dim=1))

        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
