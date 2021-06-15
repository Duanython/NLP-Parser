import torch
import torch.nn as nn


class ParserModel(nn.Module):
    def __init__(self, embeddings, in_features=36,
                 hidden_size=200, out_features=3, dropout_prob=0.5):
        """ Initialize the parser model.
        @param embeddings (ndarray): word embeddings (num_words, embedding_size)
        @param in_features (int): number of input features
        @param hidden_size (int): number of hidden units
        @param out_features (int): number of output classes
        @param dropout_prob (float): dropout probability
        """
        super(ParserModel, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.dropout_prob = dropout_prob
        # 词向量长度：50
        self.embed_size = embeddings.shape[1]
        # class torch.nn.Embedding and Embedding.weight
        self.pretrained_embeddings = nn.Embedding(embeddings.shape[0], self.embed_size)
        self.pretrained_embeddings.weight = nn.Parameter(torch.tensor(embeddings))
        self.embeddings = self.pretrained_embeddings.weight

        self.embed_to_hidden = nn.Linear(self.in_features * self.embed_size, self.hidden_size)
        nn.init.xavier_uniform_(self.embed_to_hidden.weight)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.out_features)
        nn.init.xavier_uniform_(self.hidden_to_logits.weight)
        print(self)

    def embedding_lookup(self, t):
        x = self.pretrained_embeddings(t).view(t.shape[0], -1)
        return x

    def forward(self, t):
        embeddings = self.embedding_lookup(t)
        hidden = self.embed_to_hidden(embeddings)
        hidden = nn.ReLU()(hidden)
        hidden = self.dropout(hidden)
        logits = self.hidden_to_logits(hidden)
        return logits
