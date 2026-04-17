(distributed_custom_compute_comm_overlap)=

# Custom Compute/Communication Overlap Kernels

This page is a design-oriented demo of how to package an overlapped
compute/communication region as a PyTorch custom operator.

It is intentionally **not** a drop-in public recipe for extracting NCCL
communicators from `torch.distributed`. That part is backend-specific. The goal
here is to show:

- what the kernel is trying to do,
- where the custom-op boundary belongs,
- how to register it cleanly,
- and how the extension build path relates to `torch.compile` and AOTInductor.

## Why overlap exists

The slow pattern is:

1. launch the entire collective,
2. wait for it to finish,
3. launch the next compute kernel.

The usual overlap trick is to **pipeline chunks** so that chunk `i` is still in
flight on the communication stream while chunk `i - 1` is already being consumed
by compute.

:::{figure} _static/img/distributed_overlap/overlap_timeline.svg
:alt: Timeline showing a sequential launch pattern above and an overlapped launch pattern below.
:width: 100%
Sequential launch versus chunked overlap.
:::

## Pipeline the tensor, not the whole step

In practice, overlap kernels usually:

- partition the logical tensor into chunks,
- launch collective work for one chunk on a communication stream,
- record an event for just that chunk,
- let the compute stream wait on that event,
- and launch local math as soon as the chunk is ready.

:::{figure} _static/img/distributed_overlap/chunk_pipeline.svg
:alt: Diagram of a chunked pipeline where reduce-scatter for chunk i overlaps with compute on chunk i-1.
:width: 100%
Per-chunk communication and compute pipeline.
:::

This is the same idea behind many DDP/FSDP overlap strategies: the unit of
scheduling is a **bucket or chunk**, not the full tensor for the entire step.

## Choose the boundary deliberately

You can place the overlap logic in different layers. The right choice depends on
how much of the communication stack you need to own.

:::{figure} _static/img/distributed_overlap/integration_choices.svg
:alt: Diagram comparing a Python communication hook, a custom operator, and a backend extension as integration boundaries.
:width: 100%
Common integration boundaries for overlap experiments.
:::

- **Python comm hook**:
  good for quickly orchestrating existing collectives, bucketing, or callback
  ordering.
- **Custom operator**:
  good when the overlapped region has a real C++/CUDA implementation that owns
  chunking, streams, events, and launch order.
- **Custom ProcessGroup/backend extension**:
  good when you need to own communicator creation, collective transport, or
  backend lifecycle, not just a fused launch sequence.

For most overlap-kernel prototypes, the cleanest split is:

- let `torch.distributed` continue to own the high-level process group model,
- expose one fused launcher as a custom op,
- and keep communicator lookup behind a backend-specific handle that your C++
  code understands.

## Integrating the kernel as a custom op

For a real C++/CUDA overlap launcher, a good default is:

1. define the op schema with `TORCH_LIBRARY`,
2. register the device implementation with `TORCH_LIBRARY_IMPL`,
3. build and load the shared library,
4. register a fake kernel so `torch.compile` and `torch.export` can reason
   about shapes,
5. and register autograd if the op participates in backward.

:::{figure} _static/img/distributed_overlap/custom_op_stack.svg
:alt: Software stack from Python model code to torch.ops dispatcher, C++ launcher, and CUDA communication and compute streams.
:width: 100%
Where the custom operator sits in the stack.
:::

### C++ registration sketch

```cpp
#include <torch/extension.h>

at::Tensor reduce_scatter_matmul_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    int64_t comm_token,
    int64_t rank,
    int64_t world_size);

TORCH_LIBRARY(overlap_demo, m) {
  m.def(
      "reduce_scatter_matmul("
      "Tensor x, Tensor weight, int comm_token, int rank, int world_size"
      ") -> Tensor");
}

TORCH_LIBRARY_IMPL(overlap_demo, CUDA, m) {
  m.impl("reduce_scatter_matmul", &reduce_scatter_matmul_cuda);
}
```

The `comm_token` above is intentionally abstract. In a real implementation it
would typically identify backend-owned state that can recover the communicator,
stream policy, topology hints, or temporary buffers needed by the launcher.

### CUDA-side launcher sketch

```cpp
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAContext.h>

at::Tensor reduce_scatter_matmul_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    int64_t comm_token,
    int64_t rank,
    int64_t world_size) {
  const auto device_index = x.get_device();
  auto comm_stream = at::cuda::getStreamFromPool(true, device_index);
  auto compute_stream = at::cuda::getDefaultCUDAStream(device_index);

  CommHandle comm = lookup_comm(comm_token); // backend-specific
  auto out = at::empty({x.size(0) / world_size, weight.size(1)}, x.options());
  std::vector<at::cuda::CUDAEvent> ready(world_size);

  for (int64_t chunk = 0; chunk < world_size; ++chunk) {
    launch_reduce_scatter_chunk(comm, x, chunk, rank, world_size, comm_stream);
    ready[chunk].record(comm_stream);
    ready[chunk].block(compute_stream);
    launch_matmul_chunk(out, weight, chunk, compute_stream);
  }

  return out;
}
```

The important idea is not the exact helper names. The important idea is that
the launcher owns:

- the chunk schedule,
- the communication stream,
- the compute stream,
- and the event edges between them.

That is the essence of the overlap kernel.

### Python-side registration sketch

After the shared library is loaded, add the metadata that compiler flows need.

```python
import torch

torch.ops.load_library("/path/to/liboverlap_demo.so")

@torch.library.register_fake("overlap_demo::reduce_scatter_matmul")
def _(x: torch.Tensor,
      weight: torch.Tensor,
      comm_token: int,
      rank: int,
      world_size: int) -> torch.Tensor:
    rows_per_rank = x.size(0) // world_size
    return x.new_empty((rows_per_rank, weight.size(1)))
```

If the operator participates in training, also register autograd using
`torch.library.register_autograd` or explicitly expose a backward op.

If the operator mutates inputs or returns aliased outputs, the schema must say
so accurately. Custom overlap kernels often fail in subtle ways when aliasing or
mutation is hand-waved.

## `custom_op` or `triton_op`?

If the implementation really owns streams, events, or collective side effects,
an opaque custom-op boundary is usually the right abstraction.

- Use `torch.library.custom_op` / `TORCH_LIBRARY` when you want
  `torch.compile` and `torch.export` to treat the op as an external call.
- Use `torch.library.triton_op` when the implementation is fundamentally a
  Triton/PyTorch program and you want compiler subsystems to see into it.

For overlap kernels that coordinate communication and compute, the first option
is usually the safer default.

## How are these kernels compiled?

There are **two different compilation questions** here:

1. how you build the custom-op shared library itself,
2. and how PyTorch compiles the surrounding model.

:::{figure} _static/img/distributed_overlap/compilation_paths.svg
:alt: Diagram comparing extension build paths with torch.compile and AOTInductor model compilation paths.
:width: 100%
Extension build paths versus model compilation paths.
:::

| Topic | What happens |
| --- | --- |
| Build the extension on demand | `torch.utils.cpp_extension.load()` or `load_inline()` compiles the C++/CUDA sources the first time you import/use them. This is "JIT-like" for the extension build itself. |
| Build the extension ahead of time | `CppExtension` / `CUDAExtension`, setuptools, or CMake build a shared library in CI or during wheel creation. This is AOT for the extension itself. |
| Run under `torch.compile` | `torch.compile` captures the surrounding Python model and JIT-compiles the graph around the custom op. The custom op call is usually an opaque node; `torch.compile` does **not** compile your extension source for you. |
| Run under `torch.export` + AOTInductor | AOTInductor AOT-compiles the exported model into generated artifacts. The custom-op shared library still needs to be present and registered at runtime unless you lowered the op into compiler-generated code. |

The practical rule is:

- **extension build path** answers "how does my `.so` get created?",
- **`torch.compile`** answers "how does PyTorch optimize the graph that calls my op?",
- **AOTInductor** answers "how do I package the graph for deployment?".

They are related, but they are not the same thing.

## A small prototype workflow

If you want the fastest path to a working demo PR:

1. start with `load()` or `load_inline()` to iterate quickly,
2. register a fake kernel immediately,
3. validate eager correctness first,
4. then validate `torch.compile`,
5. then decide whether you want an AOT build of the extension and/or an
   AOTInductor package of the model.

For a more production-shaped version:

1. build the extension with `CUDAExtension` or CMake,
2. load it with `torch.ops.load_library`,
3. keep the op schema and fake/autograd registrations stable,
4. and treat `torch.compile` / AOTInductor as consumers of that registered op.

## What to validate

For overlap kernels, validation is usually more important than the code sketch:

- run `torch.library.opcheck` on the eager op,
- verify the fake kernel matches shape, dtype, and device semantics,
- compare eager and compiled numerics,
- confirm the op survives `torch.compile`,
- confirm it survives `torch.export` if you need AOT deployment,
- and inspect an Nsight Systems trace to prove that communication and compute
  actually overlap.

If the trace still shows one long collective followed by one long compute
kernel, then you built a fused op, but not an overlap kernel.
