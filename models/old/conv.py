from torch import nn


class ConvNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, pad_idx):
        super(ConvNet, self).__init__()
        self.num_out_kernel = 512
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=pad_idx)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.num_out_kernel, kernel_size=(9, emb_dim)),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.num_out_kernel, out_features=64),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=16),
            nn.Linear(in_features=16, out_features=2),
        )
        self.drop = nn.Dropout(p=0.5)

    def forward(self, sentences, masks=None):
        emb = self.embedding(sentences)
        emb = emb.unsqueeze(dim=1)
        cs = self.features(emb)
        cs = cs.view(sentences.shape[0], self.num_out_kernel, -1)
        rs = self.drop(nn.functional.max_pool1d(cs, kernel_size=cs.shape[-1]))
        rs = rs.view(sentences.shape[0], self.num_out_kernel)
        soft = self.classifier(rs)
        return soft, rs
