```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# torch.nn.attention.flex_attention

```{eval-rst}
.. currentmodule:: torch.nn.attention.flex_attention
```
```{eval-rst}
.. py:module:: torch.nn.attention.flex_attention
```

FlexAttention is easiest to reason about in layers:

- author the math with `score_mod` and `mask_mod`
- compress the mask into a `BlockMask` schedule with `create_block_mask`
- run the compiled path, where Dynamo captures the callables into subgraphs and Inductor lowers `torch.ops.higher_order.flex_attention` to Triton or CPU kernels

```{note}
The diagrams below are representative of the compiled FlexAttention path. They are meant to explain the data structures and lowering stages, not to be exact dumps of one specific run.
```

## A compact example

```python
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def causal_mask(batch, head, q_idx, kv_idx):
    del batch, head
    return q_idx >= kv_idx


def alibi_bias(score, batch, head, q_idx, kv_idx):
    del batch
    slope = (-0.01 * (head + 1)).to(score.dtype)
    return score + (q_idx - kv_idx).to(score.dtype) * slope


q = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.bfloat16)

block_mask = create_block_mask(
    causal_mask,
    B=1,
    H=8,
    Q_LEN=q.size(-2),
    KV_LEN=k.size(-2),
    device="cuda",
)

compiled_flex_attention = torch.compile(flex_attention, fullgraph=True)
out = compiled_flex_attention(q, k, v, score_mod=alibi_bias, block_mask=block_mask)
```

`mask_mod` controls which tiles are visited. `score_mod` changes the math inside every visited tile. Using both lets the kernel skip work before it evaluates the score transform.

## Visual Walkthrough

### 1. `create_block_mask()` converts dense elementwise decisions into a sparse tile schedule

```{figure} _static/img/nn/flex_attention/block_mask_overview.svg
:alt: Dense causal mask converted into full, partial, and skipped FlexAttention tiles.
:width: 100%

A block is "full" when every element in the tile is valid, "partial" when the tile needs per-element masking, and skipped otherwise. The sparse schedule is what the kernel iterates over.
```

### 2. `BlockMask` stores forward and backward traversal metadata

```{figure} _static/img/nn/flex_attention/block_mask_metadata.svg
:alt: FlexAttention BlockMask metadata tables for kv and full_kv traversal.
:width: 100%

Forward kernels reduce along the KV axis, so the primary metadata is `kv_num_blocks` plus `kv_indices`. When available, `full_kv_*` lets the kernel skip calling `mask_mod` on tiles that are already known to be entirely valid. Backward needs the transposed traversal and PyTorch materializes that as `q_*` and `full_q_*`.
```

### 3. In compiled execution, Dynamo captures the user callables as subgraphs

```{figure} _static/img/nn/flex_attention/compiler_path.svg
:alt: FlexAttention compilation pipeline from Python callables to a higher-order operator and backend kernels.
:width: 100%

The user-facing Python call still looks like `flex_attention(q, k, v, score_mod=..., block_mask=...)`. Under `torch.compile`, Dynamo traces the `score_mod` and `mask_mod` bodies into small graphs and packages them into the higher-order operator `torch.ops.higher_order.flex_attention(...)`.
```

### 4. Dynamo rewrites the Python frame around the compiled region

```{figure} _static/img/nn/flex_attention/dynamo_bytecode.svg
:alt: Side-by-side comparison of representative original and Dynamo-rewritten bytecode for a FlexAttention call.
:width: 100%

If you enable `TORCH_LOGS="+dynamo,guards,bytecode"`, the bytecode trace is a useful way to confirm that the outer frame was rewritten to call a compiled function and that Dynamo emitted any needed `__resume_at_<offset>` continuations around graph breaks.
```

### 5. Triton launches one program per query tile and walks only the active KV blocks

```{figure} _static/img/nn/flex_attention/triton_launch.svg
:alt: Triton launch grid and active-block iteration for FlexAttention kernels.
:width: 100%

The Triton path uses a launch grid shaped like `(ceil_div(num_queries, BLOCK_M), batch_size, q_heads)`. Each program instance loads one query tile, iterates over the active block columns from `kv_indices`, and only consults `mask_graph` on partial tiles. Full tiles take the fast path and skipped tiles are never visited.
```

## Debugging Tips

- Use `TORCH_LOGS="+dynamo,guards,bytecode"` to see the bytecode rewrite and guard generation.
- Look for `torch.ops.higher_order.flex_attention(...)`, `sdpa_score0`, and `sdpa_mask0` in compiler logs or captured graphs to confirm that the callables were lowered as subgraphs.
- Start with `torch.compile(flex_attention, fullgraph=True)` when you want the call to stay inside a single compiled region while you inspect the generated graphs.
- The most useful mental split is: `mask_mod` decides which tiles exist, `score_mod` changes values inside those tiles, and `BlockMask` is the bridge between the two.

## API Reference

```{eval-rst}
.. autofunction:: flex_attention
```
```{eval-rst}
.. autoclass:: AuxOutput
```
```{eval-rst}
.. autoclass:: AuxRequest
```

## BlockMask Utilities

```{eval-rst}
.. autofunction:: create_block_mask
```
```{eval-rst}
.. autofunction:: create_mask
```
```{eval-rst}
.. autofunction:: and_masks
```
```{eval-rst}
.. autofunction:: or_masks
```
```{eval-rst}
.. autofunction:: noop_mask
```

## FlexKernelOptions

```{eval-rst}
.. autoclass:: FlexKernelOptions
    :members:
    :undoc-members:
```

## BlockMask

```{eval-rst}
.. autoclass:: BlockMask
    :members:
    :undoc-members:
```
