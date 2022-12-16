import torch
import torch.nn as nn


class LSTM(nn.Module):
    _hidden_size = 128

    def __init__(self, embedding_matix):
        super(LSTM, self).__init__()
        # number of words in embadding matrix
        num_words = embedding_matix.shape[0]

        # embedding dimension
        embed_dim = embedding_matix.shape[1]

        # input embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embed_dim
        )

        # use embedding matrix as weights of embedding layer
        self.embedding.weight = nn.Parameter(
            torch.tensor(
                embedding_matix,
                dtype=torch.float32
            )
        )

        # not to train the pretrained embeddings
        self.embedding.weight.requires_grad = False

        # bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=self._hidden_size,
            bidirectional=True,
            batch_first=True,
        )

        # output linear layer with one output
        # input (512) = 2*128 + 2*128 for mean and same for max pooling
        # 128 for each direction
        self.out = nn.Linear(4*self._hidden_size, 1)

    def forward(self, x):
        # pass data through embedding layer
        # the input is just the tokens
        x = self.embedding(x)

        # move embedding output to lstm
        x, _ = self.lstm(x)

        # apply mean and max pooling o lstm output
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)

        # concatenate mean and max pooling
        out = torch.cat((avg_pool, max_pool), 1)

        # pass through the output linear layer
        out = self.out(out)

        return out
