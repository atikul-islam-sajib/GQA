import os
import sys
import math
import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        query_heads: int = 8,
        kv_heads: int = 4,
        dropout: float = 0.1,
    ):
        super(GroupedQueryAttention, self).__init__()
        self.dimension = dimension
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout

        assert (
            self.dimension % (self.query_heads) == 0
        ), "dimension must be divisible by (query_heads + kv_heads)"
        assert (
            self.query_heads % self.kv_heads == 0
        ), "query_heads must be divisible by kv_heads"

        self.query = nn.Linear(in_features=self.dimension, out_features=self.dimension)
        self.key = nn.Linear(
            in_features=self.dimension,
            out_features=self.kv_heads * (self.dimension // self.query_heads),
        )
        self.value = nn.Linear(
            in_features=self.dimension,
            out_features=self.kv_heads * (self.dimension // self.query_heads),
        )

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        else:
            query = self.query(x)
            key = self.key(x)
            value = self.value(x)

            query = query.view(
                query.size(0),
                query.size(1),
                self.query_heads,
                self.dimension // self.query_heads,
            )
            key = key.view(
                key.size(0),
                key.size(1),
                self.kv_heads,
                self.dimension // self.query_heads,
            )
            value = value.view(
                value.size(0),
                value.size(1),
                self.kv_heads,
                self.dimension // self.query_heads,
            )

            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)

            key = key.repeat_interleave(repeats=self.query_heads//self.kv_heads, dim=1)
            value = value.repeat_interleave(repeats=self.query_heads//self.kv_heads, dim=1)

            scores = torch.matmul(
                input=query, other=torch.transpose(input=key, dim0=-2, dim1=-1)
            )
            scores = scores / math.sqrt(key.size(-1))
            attention = torch.matmul(
                input=torch.softmax(input=scores, dim=-1), other=value
            )

            attention = attention.view(
                attention.size(0),
                attention.size(2),
                attention.size(1) * attention.size(3),
            )

            return attention


if __name__ == "__main__":
    attention = GroupedQueryAttention(
        dimension=512, query_heads=8, kv_heads=2, dropout=0.1
    )

    texts = torch.randn((64, 128, 512))
    assert (
        attention(texts).size()
    ) == texts.size(), "Grouped Attention Failed".capitalize()
