from __future__ import annotations

import sympy
from sympy import S

from torch._prims_common import BoolLike, FloatLike, IntLike


"""
``torch.fx.experimental.symbolic_shapes`` provides interfaces for interacting with
our symbolic shapes reasoning system that is used heavily in torch.compile.  Although
this is not generally considered public API, when writing framework code in PyTorch
as well as extensions to PyTorch (e.g., in custom operator implementations), you may
need to make use of these APIs to setup dynamic shapes support appropriately.
"""

import abc
import atexit
import collections
import dis
import functools
import glob
import hashlib
import inspect
import itertools
import logging
import math
import operator
import os
import re
import sys
import threading
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import _GeneratorContextManager, contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import (
    Any,
    cast,
    Generic,
    NamedTuple,
    NoReturn,
    TYPE_CHECKING,
    TypeAlias,
    TypeGuard,
    TypeVar,
)
from typing_extensions import deprecated, ParamSpec

import torch
import torch.fx
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree

# NB: The sym_* functions are used via getattr() and must be imported here.
from torch import SymBool, SymFloat, SymInt
from torch._C._functorch import get_unwrapped, is_batchedtensor
from torch._guards import ShapeGuard, SLoc, Source, TracingContext
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_opaque_value
from torch._logging import dtrace_structured, LazyString, structured, trace_structured
from torch._opaque_base import OpaqueBase
from torch._subclasses.meta_utils import is_sparse_any
from torch._utils_internal import signpost_event
from torch.fx.experimental import _config as config
from torch.fx.experimental._config import AggressiveGuardFreeMode
from torch.fx.experimental.recording import (
    FakeTensorMeta,
    record_shapeenv_event,
    replay_shape_env_events,
    shape_env_check_state_equal,
    ShapeEnvEvent,
)
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch.types import py_sym_types
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import (
    Application,
    CeilToInt,
    CleanDiv,
    FloorDiv,
    FloorToInt,
    IntTrueDiv,
    IsNonOverlappingAndDenseIndicator,
    Max,
    Mod,
    PythonMod,
    TruncToInt,
)
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.printers import CppPrinter, PythonPrinter
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.symbol import make_symbol, symbol_is_type, SymT
from torch.utils._sympy.value_ranges import (
    bound_sympy,
    SymPyValueRangeAnalysis,
    ValueRangeError,
    ValueRanges,
)
from torch.utils._traceback import CapturedTraceback, format_frame


if TYPE_CHECKING:
    import types

    from torch import Tensor
    from torch._dynamo.source import TensorPropertySource
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.types import BoolLikeType, FloatLikeType, IntLikeType


InputList = list
DimList = list

log = logging.getLogger(__name__)


from torch.fx.experimental._size_hinting import (
    _guarding_hint_or_throw_base,
    _optimization_hint_base,
)



from ._symbolic_shapes_constraints import *  # noqa: F403
from ._symbolic_shapes_shape_env import *  # noqa: F403
from ._symbolic_shapes_utils import *  # noqa: F403

aten = torch._ops.ops.aten  # type: ignore[has-type]

__all__ = [
    "optimization_hint",
    "guarding_hint_or_throw",
    "guard_or_false",
    "guard_or_true",
    "has_symbolic_sizes_strides",
    "create_contiguous",
    "ShapeEnv",
    "is_concrete_int",
    "is_concrete_float",
    "is_concrete_bool",
    "has_static_value",
    "guard_int",
    "guard_float",
    "guard_scalar",
    "canonicalize_bool_expr",
    "SYMPY_INTERP",
    "free_symbols",
    "is_symbol_binding_fx_node",
    "is_nested_int",
    "SHAPEENV_EVENT_KEY",
    "CURRENT_NODE_KEY",
    "has_free_symbols",
    "has_free_unbacked_symbols",
    "sym_and",
    "sym_eq",
    "sym_or",
    "SymbolicContext",
    "StatelessSymbolicContext",
    "StatefulSymbolicContext",
    "SubclassSymbolicContext",
    "SymIntSymbolicContext",
    "TrackedFake",
    "statically_known_true",
    "statically_known_false",
    "guard_size_oblivious",
    "check_consistent",
    "compute_unbacked_bindings",
    "ConvertIntKey",
    "rebind_unbacked",
    "resolve_unbacked_bindings",
    "is_accessor_node",
    "ValueRangesSLoc",
    "SymIntEqByExpr",
    "Specialization",
]

