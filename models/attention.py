import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.ff = nn.Linear(in_features=hidden_size, out_features=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, contexts, masks=None):
        """
        :param contexts: (batch_size, seq_len, n_hid)
        :param masks: (batch_size, seq_len)
        :return: (batch_size, n_hid), (batch_size, seq_len)
        """
        out = self.ff(contexts)
        out = out.view(contexts.size(0), contexts.size(1))
        if masks is not None:
            masked_out = out.masked_fill(masks, float('-inf'))
        else:
            masked_out = out
        attn_weights = self.softmax(masked_out)
        out = attn_weights.unsqueeze(1).bmm(contexts)
        out = out.squeeze(1)
        return out, attn_weights


class AttentionEmbedding(nn.Module):
    def __init__(self, embedding_size=256, hidden_size=256, output_size=2,
                 external_token_embed=True, vocab_size=-1):
        super(AttentionEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.external_token_embed = external_token_embed
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if not self.external_token_embed:
            assert self.vocab_size != -1, 'Please provide vocabulary size to use embedding layer.'
            self.emb_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        self.emb_transform = nn.Linear(in_features=self.embedding_size, out_features=self.hidden_size)
        self.emb_drop = nn.Dropout(0.1)

        encoder_layer = nn.TransformerEncoderLayer(self.hidden_size, 4, 2 * self.hidden_size, 0.1, 'relu')
        encoder_norm = nn.LayerNorm(self.hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, 6, encoder_norm)

        self.combiner = Attention(hidden_size)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, sequences, masks=None):
        """
        :param sequences: (batch_size, seq_len, hidden_dim) is external_token_embed == True else (batch_size, seq_len)
        :param masks: (batch_size, seq_len)
        :return: (batch_size, n_hid)
        """
        if self.external_token_embed:
            assert len(sequences.size()) == 3, 'Must provide a 3 dimension (batch_size * seq_len * hidden_dim) ' \
                'input for using external embedding'
            embedding = sequences.transpose(*(0, 1))
        else:
            assert len(sequences.size()) == 2, 'Must provide a 2 dimension (batch_size * seq_len) ' \
                'input for using external embedding'
            embedding = self.emb_layer(sequences)
            embedding = embedding.transpose(0, 1)
        transformed_emb = self.emb_transform(embedding)
        enc_emb = self.encoder(transformed_emb, src_key_padding_mask=masks)
        enc_emb = enc_emb.transpose(*(0, 1))
        combined_enc, attn_weights = self.combiner(enc_emb, masks)
        out = self.output_layer(combined_enc)
        return out, combined_enc, attn_weights
