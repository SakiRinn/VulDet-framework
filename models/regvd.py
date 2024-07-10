import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import *


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

    def __init__(self, transformer, config, tokenizer,
                 feature_size=768, hidden_size=256, window_size=5,
                 num_classes=2, num_gnn_layers=2, fp16=False,
                 remove_residual=False, gnn="ResGatedGNN", att_op="mul"):
        super(ReGVD, self).__init__()
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

    def forward(self, input_ids, labels=None):
        # construct graph
        x_adj, x_feature = build_graph(input_ids.detach().cpu().numpy(),
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
        dtype = torch.float16 if self.fp16 else torch.float32
        outputs = self.gnn(adj_feature.to(device, dtype),
                           x_adj.to(device, dtype),
                           adj_mask.to(device, dtype))
        # Classification layer
        logits = self.classifier(outputs)
        prob = self.softmax(logits)

        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + \
                torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, logits
        else:
            return logits
