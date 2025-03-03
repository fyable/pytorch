import math
import torch
from torch import nn

def masked_softmax(scores, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(scores, dim=-1)
    max_len = scores.size(1)
    valid_lens = valid_lens.clamp(min=0, max=max_len-1)
    mask = torch.zeros_like(scores).scatter_(1, valid_lens.unsqueeze(1), float('-inf'))
    return nn.functional.softmax(scores + mask, dim=1)

class DotProductAttention(nn.Module):
    def __init__(self, dropout, B, head_dim, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.B = B
        self.head_dim = head_dim
        self.d_low = head_dim // B
        
        self.W_q_low = nn.Linear(head_dim, self.d_low)
        self.W_k_low = nn.Linear(head_dim, self.d_low)
        self.W_up = nn.Linear(self.d_low, head_dim)
        
        # 新增高维投影层
        self.W_q_high = nn.Linear(head_dim, self.d_low)
        self.W_k_high = nn.Linear(head_dim, self.d_low)
    
    def forward(self, queries, keys, values, valid_lens):
        q_low = self.W_q_low(queries)    # [b, s, d_low]
        k_low = self.W_k_low(keys)    # [b, s, d_low]
        
        scores_low = (q_low @ k_low.transpose(1,2)) / math.sqrt(self.d_low)
        
        if valid_lens is not None:
            max_len = scores_low.size(1)
            valid_lens = valid_lens.clamp(min=0, max=max_len-1)
            mask = torch.zeros_like(scores_low).scatter_(1, valid_lens.unsqueeze(1), float('-inf'))
            scores_low_masked = scores_low + mask
        else:
            scores_low_masked = scores_low
        
        topk_indices = None
        if self.B > 0:
            _, topk = torch.topk(scores_low_masked, k=self.B, dim=2)
            topk_indices = topk  # [b, s, B]
         
        if self.B == 0 or topk_indices is None:
            attention_weights = masked_softmax(scores_low, valid_lens)
            return self.dropout(attention_weights) @ values
        
        # 提取高维向量并投影到d_low维度
        b, s, _ = queries.size()
        indices_flat = topk_indices.reshape(b*s, B)  # [b*s, B]
        
        q_high = queries.reshape(b*s, 1, self.head_dim)[indices_flat]  # [b*s*B, 1, head_dim]
        k_high = keys.reshape(b*s, 1, self.head_dim)[indices_flat]  # [b*s*B, 1, head_dim]
        
        q_proj = self.W_q_high(q_high)     # [b*s*B, 1, d_low]
        k_proj = self.W_k_high(k_high)     # [b*s*B, 1, d_low]
        
        score_high = (q_proj @ k_proj.transpose(0,1)) / math.sqrt(self.d_low)
        score_high = score_high.reshape(b, s, B, self.d_low)  # [b, s, B, d_low]
        
        # 使用scatter进行批量修正
        corrected_scores = scores_low_masked.clone()
        corrected_scores.scatter_(2, topk_indices, score_high)
        
        attention_weights = masked_softmax(corrected_scores, valid_lens)
        return self.dropout(attention_weights) @ values

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, B=1, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.B = B
        
        self.head_dim = num_hiddens // num_heads
        assert self.head_dim % B == 0, "Head dimension must be divisible by B after scaling"
        
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        
        self.attention = DotProductAttention(dropout, B=B, head_dim=self.head_dim)
    
    def forward(self, queries, keys, values, valid_lens):
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)
        
        def transpose_qkv(X):
            X = X.reshape(X.size(0), X.size(1), self.num_heads, -1)
            return X.permute(0, 2, 1, 3).contiguous()
        
        queries = transpose_qkv(queries)
        keys = transpose_qkv(keys)
        values = transpose_qkv(values)
        
        if valid_lens is not None:
            valid_lens = valid_lens.unsqueeze(1).repeat(1, self.num_heads).squeeze(1)
            valid_lens = valid_lens.unsqueeze(0)
        
        output = self.attention(queries, keys, values, valid_lens)
        
        def transpose_output(X):
            X = X.reshape(-1, self.num_heads, X.size(1), X.size(2))
            return X.permute(0, 2, 1, 3).contiguous()
        
        output = transpose_output(output)
        return self.W_o(output)

# 使用示例
if __name__ == "__main__":
    input_size = 512
    hidden_size = 2048
    num_heads = 16
    B = 4
    
    mha = MultiHeadAttention(
        key_size=input_size,
        query_size=input_size,
        value_size=input_size,
        num_hiddens=hidden_size,
        num_heads=num_heads,
        dropout=0.1,
        B=B
    )
    
    batch_size = 2
    seq_len = 10
    queries = torch.randn(batch_size, seq_len, input_size)
    keys = torch.randn(batch_size, seq_len, input_size)
    values = torch.randn(batch_size, seq_len, input_size)
    valid_lens = torch.tensor([seq_len]*batch_size)
    
    output = mha(queries, keys, values, valid_lens)
    print(output.shape)  # 应输出: torch.Size([2, 10, 2048])
