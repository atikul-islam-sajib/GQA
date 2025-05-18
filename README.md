# Grouped Query Attention (GQA) in PyTorch

This repository contains a simple implementation of **Grouped Query Attention (GQA)** — a variant of multi-head attention where the number of key/value heads is smaller than the number of query heads. This design reduces computation and memory overhead while maintaining expressiveness.

---

## Features

* Implements GQA with customizable number of query and key/value heads.
* Supports attention head grouping for improved efficiency.
* Compatible with batched input data: shape `[batch_size, seq_len, hidden_dim]`.

---

## How It Works

In GQA, we use more **query heads** than **key/value heads**. Key and value heads are **repeated** to match the number of query heads during attention computation. This is beneficial for transformer models where query projection needs more flexibility than keys/values.

---

## Usage

```python
from gqa import GroupedQueryAttention
import torch

# Create attention module
attention = GroupedQueryAttention(dimension=512, query_heads=8, kv_heads=4)

# Input: [batch_size, sequence_length, hidden_dim]
x = torch.randn(64, 128, 512)

# Apply attention
output = attention(x)

# Output shape matches input
print(output.shape)  # torch.Size([64, 128, 512])
```

---

## 🧪 Test

Run the following to verify functionality:

```bash
python GQA.py
``
