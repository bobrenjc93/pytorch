"""
vnccl: Virtual NCCL — a distributed backend that resolves collectives
locally using thread synchronization.

Designed for torchmux, where N distributed workers run as threads in a
single process sharing 1 GPU. Each collective blocks the calling thread
until all ranks have arrived, resolves the operation (allreduce = sum,
broadcast = copy from root, etc.), and unblocks all threads.

No GPUs are communicated between — all data movement is in-process
via tensor copy. The numerics are bitwise identical to real NCCL.

Usage:
    import torch.distributed.vnccl  # registers "vnccl" backend
    dist.init_process_group(backend="vnccl", rank=rank, world_size=N, store=store)

Typically used via torchmux rather than directly.
"""

import threading
import weakref
from functools import partial, reduce

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    ReduceOp,
    ReduceScatterOptions,
    ScatterOptions,
    _create_work_from_future,
)
from torch.distributed.distributed_c10d import _store_based_barrier
from torch.futures import Future
from torch.utils import _pytree as pytree


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _completed_work(result):
    fut = Future()
    fut.set_result(result)
    return _create_work_from_future(fut)


def _binop_reduce(tensors, op):
    res = op(torch.stack(tensors), dim=0)
    return res if isinstance(res, torch.Tensor) else res.values


_REDUCE_OPS = {
    ReduceOp.SUM: partial(_binop_reduce, op=torch.sum),
    ReduceOp.AVG: partial(_binop_reduce, op=torch.mean),
    ReduceOp.PRODUCT: partial(_binop_reduce, op=torch.prod),
    ReduceOp.MIN: partial(_binop_reduce, op=torch.min),
    ReduceOp.MAX: partial(_binop_reduce, op=torch.max),
    ReduceOp.BAND: partial(reduce, torch.bitwise_and),
    ReduceOp.BOR: partial(reduce, torch.bitwise_or),
    ReduceOp.BXOR: partial(reduce, torch.bitwise_xor),
}


# ------------------------------------------------------------------ #
# Collective resolution — each class has a work() method that takes
# the data contributed by all ranks and resolves the collective
# in-place.
# ------------------------------------------------------------------ #


class _AllReduce:
    def __init__(self, op):
        self.op = op.op

    @torch.no_grad()
    def work(self, data):
        for i in range(len(data[0])):
            dev = data[0][i].device
            tensors = [data[r][i].to(dev) for r in range(len(data))]
            res = _REDUCE_OPS[self.op](tensors)
            for r in range(len(data)):
                data[r][i].detach().copy_(res.to(data[r][i].device))


class _Broadcast:
    def __init__(self, src):
        self.src = src

    @torch.no_grad()
    def work(self, data):
        src_tensors = pytree.tree_leaves(data[self.src])
        for i in range(len(data)):
            if i == self.src:
                continue
            dst_tensors = pytree.tree_leaves(data[i])
            for s, d in zip(src_tensors, dst_tensors):
                d.detach().copy_(s)


class _AllGather:
    @torch.no_grad()
    def work(self, data):
        for src_rank in range(len(data)):
            src_tensor = data[src_rank][1][0]
            for dest in data:
                dest[0][0][src_rank].detach().copy_(src_tensor)


class _ReduceScatter:
    def __init__(self, op):
        self.op = op

    @torch.no_grad()
    def work(self, data):
        started = [False] * len(data)
        for each in data:
            chunks = each[1][0]
            for i, chunk in enumerate(chunks):
                dst = data[i][0][0]
                if not started[i]:
                    dst.detach().copy_(chunk.to(dst.device))
                    started[i] = True
                else:
                    dst.detach().add_(chunk.to(dst.device))
        if self.op == ReduceOp.AVG:
            for each in data:
                each[0][0].detach().div_(len(data))


class _Scatter:
    def __init__(self, src):
        self.src = src

    @torch.no_grad()
    def work(self, data):
        src_tensors = data[self.src][1][0]
        for rank, each in enumerate(data):
            each[0][0].detach().copy_(src_tensors[rank])


class _Gather:
    def __init__(self, dst):
        self.dst = dst

    @torch.no_grad()
    def work(self, data):
        out_list = data[self.dst][0][0]
        for rank, each in enumerate(data):
            out_list[rank].detach().copy_(each[1][0])


class _AllToAll:
    @torch.no_grad()
    def work(self, data):
        ws = len(data)
        for dst in range(ws):
            out_list, _ = data[dst]
            for src in range(ws):
                _, in_list = data[src]
                out_list[src].detach().copy_(in_list[dst])


# ------------------------------------------------------------------ #
# Collective synchronization — blocks until all ranks have contributed
# their data, then the last rank to arrive resolves the operation and
# wakes everyone up.
# ------------------------------------------------------------------ #


class _CollSync:
    def __init__(self, world_size, op):
        self._world_size = world_size
        self._op = op
        self._cond = threading.Condition()
        self._data = [None] * world_size
        self._count = 0
        self._done = False

    def join(self, rank, data):
        with self._cond:
            self._data[rank] = data
            self._count += 1
            if self._count < self._world_size:
                self._cond.wait_for(lambda: self._done)
            else:
                self._op.work(self._data)
                self._done = True
                self._cond.notify_all()
        return _completed_work(data)


# ------------------------------------------------------------------ #
# VNCCLProcessGroup
# ------------------------------------------------------------------ #


class VNCCLProcessGroup(dist.ProcessGroup):
    """
    Process group that resolves collectives locally via thread sync.
    Multiple VNCCLProcessGroup instances (one per rank) coordinate
    through class-level shared state keyed by group name.
    """

    _lock = threading.Lock()
    _active = {}

    @classmethod
    def _enter(cls, op, pg):
        with cls._lock:
            key = pg.pg_name
            if key not in cls._active:
                cls._active[key] = _CollSync(pg.size(), op)
            return cls._active[key]

    @classmethod
    def _leave(cls, sync, pg):
        with cls._lock:
            key = pg.pg_name
            if cls._active.get(key) is sync:
                del cls._active[key]

    def __init__(self, rank, world_size):
        super().__init__(rank, world_size)
        self._rank = rank
        self._world_size = world_size

        world = dist.distributed_c10d._world
        from torch.testing._internal.distributed.multi_threaded_pg import (
            ThreadLocalWorld,
        )

        if isinstance(world, ThreadLocalWorld):
            world = world._get_world()
        self._world = weakref.ref(world)
        self._ctx = torch.autograd.set_multithreading_enabled(False)

    def _do(self, op, data):
        sync = VNCCLProcessGroup._enter(op, self)
        result = sync.join(self._rank, data)
        VNCCLProcessGroup._leave(sync, self)
        return result

    # -- metadata --

    def size(self):
        return self._world_size

    @property
    def pg_name(self):
        return self._world().pg_names[self]

    @property
    def group_name(self):
        return self.pg_name

    def getBackendName(self):
        return "vnccl"

    def __repr__(self):
        return f"VNCCL(rank={self._rank}, world_size={self._world_size})"

    # -- collectives --

    def allreduce(self, tensor_list, opts=AllreduceOptions()):
        return self._do(_AllReduce(opts.reduceOp), tensor_list)

    def allreduce_coalesced(self, tensor_list, opts=AllreduceOptions()):
        return self._do(_AllReduce(opts.reduceOp), tensor_list)

    def broadcast(self, tensor_list, opts=BroadcastOptions()):
        return self._do(_Broadcast(opts.rootRank), tensor_list)

    def allgather(self, output_tensors, input_tensor, opts=AllgatherOptions()):
        return self._do(_AllGather(), (output_tensors, input_tensor))

    def _allgather_base(self, output, input, opts=AllgatherOptions()):
        chunks = list(torch.chunk(output, self._world_size))
        return self.allgather([chunks], [input], opts)

    def allgather_into_tensor_coalesced(self, outputs, inputs, opts=AllgatherOptions()):
        res = None
        for o, i in zip(outputs, inputs):
            res = self._allgather_base(o, i)
        return res

    def scatter(self, output_tensors, input_tensors, opts=ScatterOptions()):
        return self._do(_Scatter(opts.rootRank), (output_tensors, input_tensors))

    def gather(self, output_tensors, input_tensors, opts=ScatterOptions()):
        return self._do(_Gather(opts.rootRank), (output_tensors, input_tensors))

    def reduce_scatter(self, output_tensor, scatter_list, opts=ReduceScatterOptions()):
        return self._do(_ReduceScatter(opts.reduceOp), (output_tensor, scatter_list))

    def _reduce_scatter_base(self, output, input, opts=ReduceScatterOptions()):
        chunks = list(torch.chunk(input, self._world_size))
        return self.reduce_scatter([output], [chunks], opts)

    def reduce_scatter_tensor_coalesced(self, outputs, inputs, opts=ReduceScatterOptions()):
        works = [self._reduce_scatter_base(o, i, opts) for o, i in zip(outputs, inputs)]
        for w in works[:-1]:
            w.wait()
        return works[-1]

    def alltoall(self, output_list, input_list, opts=AllToAllOptions()):
        return self._do(_AllToAll(), (output_list, input_list))

    def barrier(self, opts=BarrierOptions()):
        return self.allreduce(tensor_list=[torch.ones(1)])


def _create_vnccl(prefix_store, rank, world_size, timeout):
    pg = VNCCLProcessGroup(rank, world_size)
    _store_based_barrier(rank, prefix_store, "", world_size, timeout)
    return pg


dist.Backend.register_backend("vnccl", _create_vnccl, devices=["cpu", "cuda"])
