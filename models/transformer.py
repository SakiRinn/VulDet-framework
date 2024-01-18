import torch
import torch.nn as nn
import math
from .attention import Attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, padding_idx=padding_idx)
        self.emb_transform = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=128),
            nn.ReLU(True)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4,
            dim_feedforward=2 * embedding_size,
            dropout=0.2, activation='relu')
        encoder_norm = nn.LayerNorm(128)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=4, norm=encoder_norm)
        # self.features = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=self.num_out_kernel, kernel_size=(9, emb_dim)),
        #     nn.ReLU(True)
        # )
        self.combiner = Attention(128)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=16),
            nn.Linear(in_features=16, out_features=2),
        )
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, sentences, masks=None):
        emb = self.embedding(sentences)
        emb = self.emb_transform(emb)
        encoded = self.encoder(emb.transpose(*(0, 1))).transpose(*(0, 1))
        combined, _ = self.combiner(encoded)
        combined = self.dropout(combined)
        output = self.classifier(combined)
        return output, combined


class TransformerBiGRU(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layer):
        super(TransformerBiGRU, self).__init__()
        self.num_layers = num_layer
        self.num_directions = 2
        self.hidden_size = hidden_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size, nhead=4,
            dim_feedforward=2 * embedding_size,
            dropout=0.2, activation='relu')
        encoder_norm = nn.LayerNorm(embedding_size)
        self.encoder = nn.Sequential(
            PositionalEncoding(embedding_size),
            nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=4, norm=encoder_norm)
        )
        self.features = nn.Sequential(
            nn.GRU(hidden_size=hidden_size, input_size=embedding_size,
                   bidirectional=True, batch_first=True, num_layers=self.num_layers),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size * self.num_layers * self.num_directions, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.Linear(in_features=16, out_features=2),
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, sequences, masks=None):
        embs = sequences
        embs = self.encoder(embs.transpose(*(0, 1))).transpose(*(0, 1))
        output, h_n = self.features(embs)
        h_n = self.dropout(torch.cat([h_n[ix, :, :] for ix in range(h_n.shape[0])], 1))
        out = self.classifier(h_n)
        return out


class TransformerAttention(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers):
        super(TransformerAttention, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=hidden_size),
            nn.ReLU(),
            PositionalEncoding(embedding_size),
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=hidden_size, nhead=4,
                    dim_feedforward=2 * hidden_size,
                    dropout=0.2, activation='relu'
                ),
                num_layers=4,
                norm=nn.LayerNorm(hidden_size)
            )
        )
        self.features = nn.Sequential(
            Attention(embedding_size),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.Linear(in_features=16, out_features=2),
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, sequences, masks=None):
        embs = sequences
        embs = self.encoder(embs.transpose(*(0, 1))).transpose(*(0, 1))
        output, h_n = self.features(embs)
        h_n = self.dropout(output)
        out = self.classifier(h_n)
        return out


class TransformerPool(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers):
        super(TransformerPool, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=hidden_size),
            nn.ReLU(),
            PositionalEncoding(hidden_size),
            nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=hidden_size, nhead=4,
                    dim_feedforward=2 * hidden_size,
                    dropout=0.2, activation='relu'
                ),
                num_layers=4,
                norm=nn.LayerNorm(hidden_size)
            )
        )
        # self.features = nn.Sequential(
        #     Attention(emb_dim),
        # )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.Linear(in_features=16, out_features=2),
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, sequences, masks=None):
        embs = sequences
        embs = self.encoder(embs.transpose(*(0, 1))).transpose(*(0, 1))
        output = embs[:, 0, :]
        h_n = self.dropout(output)
        out = self.classifier(h_n)
        return out
