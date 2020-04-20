
import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
 
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(10) # np.sqrt(10) 바꿔라
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, n_heads, length):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_k * n_heads)
        self.W_O = nn.Linear(n_heads * d_k, 100)

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_model = d_model
        self.length = length

    def forward(self, Q, K, V):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        batch_size = 1024
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(-1, self.length, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(-1, self.length, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(-1, self.length, self.n_heads, self.d_k).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context = ScaledDotProductAttention()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(-1, self.length, self.n_heads * self.d_k) # context: [batch_size x len_q x n_heads * d_v]
        output = self.W_O(context)
        return output.permute(1,0,2)

