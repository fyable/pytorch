import torch

if __name__ == "__main__":
    a, b, c, d, e = 2, 3, 4, 2, 2
    A = torch.arange(a * b * c).reshape(a, b, c)
    B = torch.tensor([
        [[0, 1], [2, 0]],
        [[2, 2], [2, 2]]
    ])
    
    # 构造四维索引张量
    i_indices = torch.arange(a).reshape(-1, 1, 1, 1)      # (a,1,1,1)
    j_indices = torch.arange(d).reshape(1, -1, 1, 1)      # (1,d,1,1)
    k_indices = torch.arange(e).reshape(1, 1, -1, 1)      # (1,1,e,1)
    B_expanded = B.unsqueeze(-1).expand(a, d, e, c)      # (a,d,e,c)
    l_indices = torch.arange(c).reshape(1, 1, 1, -1)      # (1,1,1,c)
    
    # 使用高级索引直接生成结果
    C = A[i_indices, B_expanded, l_indices]
    
    print("C shape:", C.shape)    # (2, 2, 2, 4)
    print("C values:\n", C)
