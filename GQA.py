import os
import sys
import torch
import torch.nn as nn


class GroupedQueryAttention:
    def __init__(
        self,
        dimension: int = 512,
        query_heads: int = 8,
        kv_heads: int = 4,
        dropout: float = 0.1,
    ):
        self.dimension = dimension
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        else:
            pass
        
        
if __name__ == "__main__":
    attention = GroupedQueryAttention(
        dimension=512,
        query_heads=8,
        kv_heads=4,
        dropout=0.1
    )
