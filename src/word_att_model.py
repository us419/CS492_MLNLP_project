"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul
from src.multihead_att import MultiHeadAttention
import pandas as pd
import numpy as np
import csv

class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=50):
        super(WordAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

        self.embed_dim = 100
        self.n_heads = 5
        self.dim_per_head = 20
        self.MultiHeadAttention = MultiHeadAttention(self.embed_dim, self.dim_per_head, self.n_heads, 26)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        output = self.lookup(input)
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        output = output.float()
        output = output.permute(1,0,2)
        output = self.MultiHeadAttention(output, output, output)
        output = matrix_mul(output, self.context_weight).permute(1,0)
        output = F.softmax(output)
        output = element_wise_mul(f_output,output.permute(1,0))
        return output, h_output

if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
