import torch
from torch import nn


class BiGRU(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layer):
        super(BiGRU, self).__init__()
        self.num_layers = num_layer
        self.num_directions = 2
        self.hidden_size = hidden_size
        self.features = nn.Sequential(
            nn.GRU(hidden_size=hidden_size, input_size=embedding_size,
                   bidirectional=True, batch_first=True, num_layers=self.num_layers),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size * self.num_layers * self.num_directions, out_features=2),
            # nn.BatchNorm1d(num_features=64),
            # nn.ReLU(True),
            # nn.Linear(in_features=64, out_features=16),
            # nn.ReLU(True),
            # nn.Linear(in_features=16, out_features=2),
        )
        self.drop = nn.Dropout(p=0.2)

    def forward(self, sequences, masks=None):
        embs = sequences
        _, h_n = self.features(embs)
        h_n = self.drop(torch.cat([h_n[ix, :, :] for ix in range(h_n.shape[0])], 1))
        out = self.classifier(h_n)
        return out, h_n


class BiRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layer):
        super(BiRNN, self).__init__()
        self.num_layers = num_layer
        self.num_directions = 2
        self.hidden_size = hidden_size
        self.features = nn.Sequential(
            nn.RNN(hidden_size=hidden_size, input_size=embedding_size,
                   bidirectional=True, batch_first=True, num_layers=self.num_layers),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size * self.num_directions * self.num_layers, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.Linear(in_features=16, out_features=2),
        )
        self.drop = nn.Dropout(p=0.5)

    def forward(self, sequences, masks=None):
        embs = sequences
        _, h_n = self.features(embs)
        h_n = self.drop(torch.cat([h_n[ix, :, :] for ix in range(h_n.shape[0])], 1))
        out = self.classifier(h_n)
        return out


class BiLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layer):
        super(BiLSTM, self).__init__()
        self.num_layers = num_layer
        self.num_directions = 2
        self.hidden_size = hidden_size
        self.features = nn.Sequential(
            nn.LSTM(hidden_size=hidden_size, input_size=embedding_size,
                    bidirectional=True, batch_first=True, num_layers=self.num_layers),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size * self.num_directions * self.num_layers, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.Linear(in_features=16, out_features=2),
        )
        self.drop = nn.Dropout(p=0.5)

    def forward(self, sequences, masks=None):
        embs = sequences
        _, (h_n, c_n) = self.features(embs)
        h_n = self.drop(torch.cat([h_n[ix, :, :] for ix in range(h_n.shape[0])], 1))
        out = self.classifier(h_n)
        return out
