"""Utilities shared by the symbolic shapes compatibility wrapper and ShapeEnv."""

from __future__ import annotations

import sympy
from sympy import S

from torch._prims_common import BoolLike, FloatLike, IntLike

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

log = logging.getLogger("torch.fx.experimental.symbolic_shapes")


from torch.fx.experimental._size_hinting import (
    _guarding_hint_or_throw_base,
    _optimization_hint_base,
)
from ._symbolic_shapes_constraints import is_symbolic


if TYPE_CHECKING:
    from ._symbolic_shapes_shape_env import ShapeEnv


def guarding_hint_or_throw(
    a: torch.SymInt | torch.SymBool | int | bool | SymNode,
) -> int | bool:
    """
    Return a concrete hint for a symbolic value, for use in guarding decisions.

    Returns Python bool (True/False) for boolean inputs (SymBool, bool),
    and Python int for integer inputs (SymInt, int).
    """
    if isinstance(a, SymNode):
        if a._hint is not None:
            return a._hint
        hint = a.shape_env.guarding_hint_or_throw(a.expr)
        a._hint = hint
        return hint
    if isinstance(a, (torch.SymInt, torch.SymBool)):
        return guarding_hint_or_throw(a.node)
    if isinstance(a, bool):
        return a
    if type(a) is not int:
        raise AssertionError(f"Expected int, got {type(a)}")
    return a


def optimization_hint(a: torch.SymInt | int, fallback: int | None = None) -> int:
    """
    Return a concrete hint for a symbolic integer, for use in optimization decisions.

    Unlike guarding_hint_or_throw, this function does not add guards and is intended
    for optimization purposes only (e.g., memory estimation).
    """
    if isinstance(a, torch.SymInt):
        if a.node._hint is not None:
            return a.node._hint
        return a.node.shape_env.optimization_hint(a.node.expr, fallback=fallback)
    if type(a) is not int:
        raise AssertionError(f"Expected int, got {type(a)}")
    return a


class GuardOnDataDependentSymNode(RuntimeError):
    cond: sympy.Basic

    def __init__(self, cond: sympy.Basic, *args: Any) -> None:
        super().__init__(*args)
        self.cond = cond


class PendingUnbackedSymbolNotFound(RuntimeError):
    pass


class _ShapeEnvGuardError(RuntimeError):
    """Raised when a guard is attempted while the ShapeEnv is in error-on-guard mode."""


aten = torch._ops.ops.aten  # type: ignore[has-type]

# FX node metadata keys for symbolic shape FX graph.
SHAPEENV_EVENT_KEY = "shapeenv_event"
CURRENT_NODE_KEY = "current_node"


def log_lru_cache_stats(wrapped_f: functools._lru_cache_wrapper[object]) -> None:
    log.debug(
        "lru_cache_stats %s: %s",
        wrapped_f.__name__,  # type: ignore[attr-defined]
        wrapped_f.cumulative_cache_info(),  # type: ignore[attr-defined]
    )


# Note about Sympy Expr/SympyBoolean/Basic typing: the Sympy hierarchy is
#
#   Basic
#       Expr
#       SympyBoolean
#           Relational
#
# Notably, Expr and SympyBoolean are not related.  So use Basic when the
# expression could denote int, float OR bool, and otherwise use the more
# specific Expr for int/float and SympyBoolean for bool.
#
# In obscure Meta only situations, sympy.logic.boolalg doesn't exist at runtime.
# So make sure only type checker evaluates this alias.
# Xref: https://www.internalfb.com/diff/D53324783
SympyBoolean: TypeAlias = "sympy.logic.boolalg.Boolean"


_T = TypeVar("_T")
_SympyT = TypeVar("_SympyT", sympy.Expr, SympyBoolean, sympy.Basic)


class SymIntEqByExpr:
    """
    This is a wrapper around SymInt which has alternative semantics for
    equality and pickling.  Specifically, instead of erroring or guarding, we
    instead will hash/compare equality based on the underlying sympy
    expression; e.g., s0 and s1 will always compare as False.

    NB: This does NOT do fancy analysis that maybe_evaluate_static does;
    we can only reason through equalities that occur because to expressions
    canonicalize to the same expression via regular simplification.
    """

    @staticmethod
    def _extract(val: torch.SymInt | int) -> sympy.Expr:
        if isinstance(val, torch.SymInt):
            return val.node.expr
        else:
            return sympy.Integer(val)

    def __init__(self, val: torch.SymInt | int) -> None:
        self.val: sympy.Expr = SymIntEqByExpr._extract(val)

    def __repr__(self) -> str:
        return repr(self.val)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SymIntEqByExpr):
            raise AssertionError(f"Expected SymIntEqByExpr, got {type(other)}")
        return self.val == other.val

    def __hash__(self) -> int:
        return hash(self.val)


def _nested_int_aware_sort(
    tup: tuple[IntLikeType, int],
) -> tuple[int, IntLikeType, int]:
    return (
        # Order nested ints by their coefficients.
        # 1 here to order nested ints after non-nested-ints.
        (1, tup[0].node.nested_int_coeff(), tup[1])
        if is_nested_int(tup[0])
        else (0, *tup)
    )


# Wrapper on lru_cache that reports statistics at process end
def lru_cache(
    maxsize: int | None,
) -> Callable[[Callable[..., _T]], functools._lru_cache_wrapper[_T]]:
    def inner(f: Callable[..., _T]) -> functools._lru_cache_wrapper[_T]:
        wrapped_f = functools.lru_cache(maxsize)(f)
        old_cache_clear = wrapped_f.cache_clear
        prev_hits = 0
        prev_misses = 0

        # TODO: There's a ref-cycle here (wrapped_f -> cumulative_cache_info
        # -> wrapped_f) but cannot be solved with weakref as wrapped_f is not
        # weakref'able on some versions of Python

        def cumulative_cache_info() -> functools._CacheInfo:
            cur = wrapped_f.cache_info()
            return functools._CacheInfo(
                prev_hits + cur.hits,
                prev_misses + cur.misses,
                cur.maxsize,
                cur.currsize,
            )

        def new_cache_clear() -> None:
            nonlocal prev_hits, prev_misses
            cur = wrapped_f.cache_info()
            prev_hits += cur.hits
            prev_misses += cur.misses
            old_cache_clear()

        wrapped_f.cache_clear = new_cache_clear  # type: ignore[attr-defined, method-assign]
        wrapped_f.cumulative_cache_info = cumulative_cache_info  # type: ignore[attr-defined, method-assign]
        if log.isEnabledFor(logging.DEBUG):
            atexit.register(log_lru_cache_stats, wrapped_f)  # type: ignore[arg-type]
        return wrapped_f

    return inner


# These are modules that contain generic code for interacting with ShapeEnv
# which are unlikely to identify a particular interesting guard statement
@lru_cache(None)
def uninteresting_files() -> set[str]:
    import torch._compile
    import torch._dynamo.eval_frame
    import torch._higher_order_ops
    import torch._inductor.sizevars
    import torch._library.custom_ops
    import torch._library.fake_impl
    import torch._logging
    import torch._subclasses.fake_tensor
    import torch._subclasses.meta_utils
    import torch.export._trace

    split_module_files = {
        __file__,
        os.path.join(os.path.dirname(__file__), "_symbolic_shapes_constraints.py"),
        os.path.join(os.path.dirname(__file__), "_symbolic_shapes_shape_env.py"),
    }
    mods = [
        torch.export._trace,
        torch.fx.experimental.recording,
        torch.fx.experimental.sym_node,
        torch.fx.interpreter,
        torch.fx._symbolic_trace,
        torch,
        torch._compile,
        torch._dynamo.eval_frame,
        torch._inductor.sizevars,
        torch._library.custom_ops,
        torch._library.fake_impl,
        torch._subclasses.meta_utils,
        torch._subclasses.fake_tensor,
        torch._logging._internal,
        torch._logging.structured,
    ]
    import torch._dynamo.guards

    files = split_module_files | {inspect.getfile(m) for m in mods}

    # Add all Python files in torch._higher_order_ops directory
    higher_order_ops_dir = os.path.dirname(torch._higher_order_ops.__file__)
    hop_files = glob.glob(os.path.join(higher_order_ops_dir, "*.py"))

    return (
        files
        | set(hop_files)
        | torch._dynamo.guards.uninteresting_files()
        | {"<string>"}
    )


class ConstraintViolationError(RuntimeError):
    pass


def has_symbolic_sizes_strides(elem: torch.Tensor) -> bool:
    return elem._has_symbolic_sizes_strides


Int: TypeAlias = torch.SymInt | int


def create_contiguous(shape: Sequence[Int]) -> list[Int]:
    strides: list[Int] = [1]
    for dim in reversed(shape[:-1]):
        strides.append(dim * strides[-1])  # type: ignore[operator]
    return list(reversed(strides))


Scalar: TypeAlias = torch.SymInt | torch.SymFloat | torch.SymBool | int | float | bool


def has_guarding_hint(a: Scalar) -> bool:
    """
    Check if a symbolic value has a hint available for guarding.

    Returns True if the value is concrete or if the symbolic node has a hint,
    False otherwise.
    """
    if isinstance(a, SymTypes):
        return a.node.has_hint()
    return True


def is_concrete_int(a: IntLikeType) -> bool:
    """
    Utility to check if underlying object
    in SymInt is concrete value. Also returns
    true if integer is passed in.

    Args:
        a (SymInt or int): Object to test if it int
    """
    if not isinstance(a, (SymInt, int)):
        raise AssertionError(f"Expected SymInt or int, got {type(a)}")

    if isinstance(a, int):
        return True

    if isinstance(a.node.expr, sympy.core.numbers.Integer):
        return True

    return False


def is_concrete_float(a: FloatLikeType) -> bool:
    r"""Utility to check if underlying object
    in SymInt is concrete value. Also returns
    true if integer is passed in.

    Args:
        a (SymInt or float): Object to test if it float
    """
    if not isinstance(a, (SymFloat, float)):
        raise AssertionError(f"Expected SymFloat or float, got {type(a)}")

    if isinstance(a, float):
        return True

    if isinstance(a.node.expr, sympy.core.numbers.Float):
        return True

    return False


def is_concrete_bool(a: BoolLikeType) -> bool:
    """
    Utility to check if underlying object
    in SymBool is concrete value. Also returns
    true if integer is passed in.

    Args:
        a (SymBool or bool): Object to test if it bool
    """
    if not isinstance(a, (SymBool, bool)):
        raise AssertionError(f"Expected SymBool or bool, got {type(a)}")

    if isinstance(a, bool):
        return True

    if isinstance(
        a.node.expr, (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse)
    ):
        return True

    return False


def has_static_value(a: SymBool | SymFloat | SymInt | bool | float | int) -> bool:
    """
    User-code friendly utility to check if a value is static or dynamic.
    Returns true if given a constant, or a symbolic expression with a fixed value.

    Args:
        a (Union[SymBool, SymFloat, SymInt, bool, float, int]): Object to test
    """
    if not isinstance(a, BoolLike + FloatLike + IntLike):
        raise AssertionError(f"Expected BoolLike/FloatLike/IntLike, got {type(a)}")
    if (
        isinstance(a, BoolLike)
        and is_concrete_bool(a)  # type: ignore[arg-type]
        or isinstance(a, FloatLike)
        and is_concrete_float(a)  # type: ignore[arg-type]
        or isinstance(a, IntLike)
        and is_concrete_int(a)  # type: ignore[arg-type]
    ):
        return True

    if not isinstance(a, py_sym_types):
        raise AssertionError(f"Expected py_sym_types, got {type(a)}")
    return a.node.shape_env.bound_sympy(a.node.expr).is_singleton()  # type: ignore[union-attr]


@deprecated(
    "guard_size_oblivious will be removed. Consider using explicit unbacked handling \
    potentially utilizing guard_or_false, guard_or_true, or statically_known_true",
    category=FutureWarning,
)
def guard_size_oblivious(expr: torch.SymBool | bool) -> bool:
    """
    Perform a guard on a symbolic boolean expression in a size oblivious way.
    This is typically used when a non-oblivious test would result in a guard
    on a data dependent value of which we don't know the value of at compile time.
    When a guard is tested this way, we may diverge in behavior from how regular
    PyTorch semantics would treat it.  For more information, see
    https://github.com/pytorch/pytorch/pull/118579
    """
    if isinstance(expr, torch.SymBool):
        return expr.node.guard_size_oblivious("", 0)
    else:
        if not isinstance(expr, bool):
            raise AssertionError(f"Expected bool, got {type(expr)}")
        return expr


def check_consistent(new: _T, old: _T) -> None:
    """
    Test that two "meta" values (typically either Tensor or SymInt) have
    the same values, e.g., after retracing.  If we don't understand the
    quantities in question, we'll just skip the consistency check.
    """
    # TODO: do boolean equality test too, see
    # https://github.com/pytorch/pytorch/issues/124110
    scalar_types = (torch.SymInt, torch.SymFloat, int, float)

    if isinstance(new, torch.Tensor):
        if not isinstance(old, torch.Tensor):
            raise AssertionError(f"Expected Tensor, got {type(old)}")
        torch._check(
            old.dim() == new.dim(), lambda: f"{old.shape} != {new.shape} (old != new)"
        )
        # Do this manually so that each individual test is irrefutable
        # (TODO: should be a helper for this, maybe sym_eq?  That
        # gives us a compound expression and I'm not sure it
        # simplifies right now)
        for i, j in zip(old.shape, new.shape):
            torch._check(i == j, lambda: f"{old.shape} != {new.shape} (old != new)")
    # NB: bool is subclass of int
    elif isinstance(new, scalar_types) and not isinstance(new, bool):
        if not (isinstance(old, scalar_types) and not isinstance(old, bool)):
            raise AssertionError(f"{old} != {new}")
        torch._check(old == new, lambda: f"{old} != {new} (old != new)")


def resolve_unbacked_bindings(
    shape_env: ShapeEnv | None,
    bindings: dict[sympy.Symbol, pytree.KeyPath] | None,
) -> dict[sympy.Symbol, pytree.KeyPath] | None:
    """
    When we do fake tensor prop, we oftentimes will allocate new unbacked symints.
    We then run proxy tensor mode, which populates node.meta["unbacked_bindings"]
    with these new symints. To ensure consistency we use PropagateUnbackedSymInts
    to rename unbacked bindings to their old ones. But all of the node metas are
    still using the old bindings from before the renaming. This function helps to
    post facto apply any renamings discovered in the PropagateUnbackedSymInts pass.
    """
    if bindings is None:
        return None
    if shape_env is None:
        raise AssertionError("shape_env should not be None")
    return {shape_env.unbacked_renamings.get(k, k): v for k, v in bindings.items()}


Result: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]


def rebind_unbacked(
    shape_env: ShapeEnv | None, n: torch.fx.Node, result: Result
) -> None:
    """
    Suppose we are retracing a pre-existing FX graph that previously had
    fake tensor propagation (and therefore unbacked SymInts).  When we retrace,
    we re-propagate fake tensors, which results in new unbacked SymInts.
    When this happens, we need to tell the shape environment about the equivalence
    of the old and new unbacked SymInts.  Pass us the old torch.fx.Node (which
    has the old binding information) and the new result (which we can extract the
    new unbacked SymInts out from).
    """

    # Inputs never need rebinding
    if n.op == "placeholder":
        return

    if bindings := resolve_unbacked_bindings(
        shape_env, n.meta.get("unbacked_bindings")
    ):
        if shape_env is None:
            raise AssertionError("shape_env should not be None")
        for raw_u0, path in bindings.items():
            u1 = pytree.key_get(result, path)

            # Sometimes, things were previously unbacked bindings become constants.
            # There are two situations this can happen.
            #
            # First, you might have a runtime assert that causes the
            # constant-ification.  In this case, the /binding/ itself will
            # still be an unbacked symbol (because we will only force it
            # to be a constant later in fake tensor propagation).  In this
            # case, u1 is a SymInt and we still do all our work as normal.
            #
            # But second, it might be that fake tensor propagation DIRECTLY
            # converted the unbacked SymInt into a constant.  This happens
            # more rarely, but we have identified two situations it can
            # validly occur:
            #
            # - If you have a tensor_version operator, these are initially
            #   allocated as unbacked SymInts, but after AOTAutograd they
            #   get forced specialized to specific values.  In this case,
            #   there is no reason to do runtime asserts on them, this is
            #   just a hack to properly keep track of them to start.
            #
            # - If you have an item() call on a constant tensor, the result
            #   of the item() call is constant and we do not need runtime
            #   asserts on this symbol.  In
            #   https://github.com/pytorch/pytorch/issues/140625 we have a
            #   case where in the initial trace of the program we are unable
            #   to determine that torch.tensor is constant, but then
            #   subsequent passes cause torch.tensor to become a constant and
            #   then the unbacked symbol goes poof.
            #
            # In all of these cases, it is no longer necessary to generate
            # deferred runtime asserts, since other subsystems (e.g., the
            # constant-ification pass) ensure that the quantity is now truly
            # static and cannot change at runtime.  So it's OK to discard
            # in these situations.
            #
            # There is one more hazard (re
            # https://github.com/pytorch/pytorch/issues/141248), the problem
            # is that you can end up with "dangling" unbacked symbols that
            # exist in the ShapeEnv but are never bound anywhere.  You might
            # like an invariant that unbacked symbols never get lost.  But
            # we do not have this invariant, so do not try to enforce it.
            if isinstance(u1, (int, float)):
                log.info(
                    "rebind_unbacked: discard %s %s %s -> %s",
                    n.target,
                    raw_u0,
                    path,
                    u1,
                )
                continue

            # We only care about rebinding unbacked things
            if u1.node.hint is not None:
                continue

            # unbacked symbols bindings might be replaced to other backed or
            # unbacked replacements.
            #
            # Example:
            #   u = x.item()
            #   torch._check(u == 5)
            #
            # The safest approach is to retrieve raw_u1 from u1.node._expr
            # and perform the rebinding on the original unbacked symbol,
            # even if it’s no longer directly referenced.
            #
            # In other words, we should always rebind the original symbol
            # before any replacements are applied.
            #   u0 -> u0 == s1
            raw_u1 = u1.node._expr

            # TODO Do we still need this logic below?
            # Simplify SymBool binding
            if (
                isinstance(raw_u1, sympy.Piecewise)
                and len(raw_u1.args) == 2
                and (
                    raw_u1_args0 := cast(
                        tuple[sympy.Basic, sympy.Basic], raw_u1.args[0]
                    )
                )
                and raw_u1_args0[0] == 1
                and isinstance(eq := raw_u1_args0[1], sympy.Eq)
                and isinstance(new_raw_u1 := eq.lhs, sympy.Symbol)
                and shape_env.var_to_range[new_raw_u1].issubset(ValueRanges(0, 1))
                and eq.rhs == 1
                and cast(tuple[sympy.Basic, sympy.Basic], raw_u1.args[1]) == (0, True)
            ):
                # This is what the pattern match above is testing
                repacked = _sympy_cast_symbool_to_symint_guardless(
                    sympy.Eq(new_raw_u1, 1)
                )
                if repacked != raw_u1:
                    raise AssertionError(f"{repacked} != {raw_u1}")
                # Cancel the to_int(to_bool(x)). This is sound because x in
                # [0, 1]

                raw_u1 = new_raw_u1

            if not isinstance(raw_u1, sympy.Symbol):
                if raw_u1.free_symbols:
                    raise AssertionError(f"should have been constant, but got {raw_u1}")
                continue

            # The old and new could be the same if you improperly hit the memo
            # while retracing.  Make sure you updated FakeTensorMode.epoch
            if raw_u0 == raw_u1:
                raise AssertionError(f"{raw_u0} possible memo disaster")
            # Reuse the OLD symbol name
            shape_env._rename_unbacked_to(raw_u1, raw_u0)


# NB: You could try to expand this to cover more cases by simply
# detecting whenever you have an int output, but this is a bit
# dangerous in case someone adds a function that returns an int but is
# mutating.  So manually whitelist for now.
def is_accessor_node(node: torch.fx.Node) -> bool:
    """
    Helper function to determine if a node is trying to access
    a symbolic integer such as size, stride, offset or item. Currently
    primarily only used in a DCE pass to figure out purity.
    """

    # Dynamo only exercised condition
    if (
        node.op == "call_method"
        and isinstance(node.args[0], torch.fx.Node)
        and isinstance(node.args[0].meta.get("example_value"), torch.Tensor)
        and node.target in ["size", "stride", "storage_offset", "item"]
    ):
        return True

    if node.op == "call_function" and node.target in [
        torch.ops.aten.sym_size,
        torch.ops.aten.sym_size.default,
        torch.ops.aten.sym_size.int,
        torch.ops.aten.sym_stride,
        torch.ops.aten.sym_stride.default,
        torch.ops.aten.sym_stride.int,
        torch.ops.aten.sym_storage_offset,
        torch.ops.aten.sym_storage_offset.default,
        torch.ops.aten.sym_numel.default,
    ]:
        return True

    return False


def canonicalize_bool_expr(expr: _T) -> _T:
    """
    Canonicalize a boolean expression by transforming it into a lt / le
    inequality and moving all the non-constant terms to the rhs.
    We canonicalize And / Ors / Not via cnf and then canonicalize their subexpr
    recursively
    nb. sympy.Rel.canonical is not good enough https://github.com/sympy/sympy/issues/25924

    Args:
        expr (sympy.Expr): Expression to canonicalize
    """
    # Canonicalise an inequality by transforming it into a lt / le
    # inequality and moving all the non-constant terms to the rhs
    # We canonicalise And / Ors / Not via cnf
    # nb. Relational.canonical in sympy is broken
    # https://github.com/sympy/sympy/issues/25924

    if not isinstance(
        expr, (sympy.Rel, sympy.And, sympy.Or, sympy.Not, sympy.Eq, sympy.Ne)
    ):
        return expr

    if isinstance(expr, (sympy.And, sympy.Or, sympy.Not)):
        expr = sympy.logic.boolalg.to_cnf(expr)
    return _canonicalize_bool_expr_impl(expr)  # type: ignore[arg-type, return-value]


def _sympy_from_args(
    cls: type[sympy.Add | sympy.Mul],
    args: list[sympy.Expr],
    sort: bool = True,
    is_commutative: bool | None = None,
) -> sympy.Expr:
    """
    Create a sympy expression from a list of arguments, optimizing for performance.

    This function creates a sympy Add or Mul expression from a list of arguments
    while avoiding expensive operations like flattening. It handles sorting the
    arguments appropriately based on the expression type.

    Args:
        cls: The sympy class to create (Add or Mul)
        args: List of sympy expressions to combine
        sort: Whether to sort the arguments (default: True)
        is_commutative: Whether the operation is commutative (default: None)

    Returns:
        A sympy expression of type cls combining all arguments

    Raises:
        ValueError: If cls is not sympy.Add or sympy.Mul
    """

    if not args:
        return cls.identity  # type: ignore[union-attr]

    # These args are already in canonical form, so we avoid calling
    # Add(*args) to avoid expensive Add.flatten operation
    if sort:
        if cls is sympy.Add:
            sort_fn = sympy.core.add._addsort
        elif cls is sympy.Mul:
            sort_fn = sympy.core.mul._mulsort
        else:
            raise ValueError(f"Unknown cls: {cls}")

        # we don't support non commutative with sort
        if is_commutative is not True:
            raise AssertionError("is_commutative must be True")
        if args[0].is_Number:
            rest = args[1:]
            sort_fn(rest)
            return cls._from_args([args[0]] + rest, is_commutative=is_commutative)  # type: ignore[attr-defined]
        else:
            args = args.copy()
            sort_fn(args)
            return cls._from_args(args, is_commutative=is_commutative)  # type: ignore[attr-defined]
    else:
        # if the args are already sorted, we create directly
        return cls._from_args(args, is_commutative=is_commutative)  # type: ignore[attr-defined]


def _canonicalize_bool_expr_impl(expr: SympyBoolean) -> SympyBoolean:
    """
    After canonicalization, we are guaranteed to have eliminated Ge/Gt relations
    (rewriting them to Le/Lt, respectively).
    """
    if isinstance(expr, (sympy.And, sympy.Or)):
        return type(expr)(*map(canonicalize_bool_expr, expr.args))

    opposite = {sympy.Gt: sympy.Lt, sympy.Ge: sympy.Le}
    t: type[Any]
    if isinstance(expr, tuple(opposite.keys())):
        rhs = expr.lhs - expr.rhs  # type: ignore[attr-defined]
        t = opposite[type(expr)]  # type: ignore[index]
    else:
        if not isinstance(expr, (sympy.Lt, sympy.Le, sympy.Eq, sympy.Ne)):
            raise AssertionError(f"Expected Lt/Le/Eq/Ne, got {type(expr)}")
        rhs = expr.rhs - expr.lhs
        t = type(expr)

    def is_neg(t: sympy.Expr) -> bool:
        return (t.is_Number and t.is_negative) or (
            isinstance(t, sympy.Mul) and t.args[0].is_Number and t.args[0].is_negative
        )

    lhs = S.Zero
    rhs = _reduce_to_lowest_terms(rhs)
    if isinstance(rhs, sympy.Add):
        pos = []
        neg = []
        for term in rhs.args:
            if is_neg(term):
                neg.append(-term)
            else:
                pos.append(term)
        # these are already sorted
        rhs = _sympy_from_args(sympy.Add, pos, sort=False, is_commutative=True)
        # the terms were changed, so needs a sorting
        lhs = _sympy_from_args(sympy.Add, neg, sort=True, is_commutative=True)
    elif is_neg(rhs):
        # lhs == 0
        lhs, rhs = -rhs, S.Zero
    # We don't have to evaluate here because lhs, rhs came from a Boolean
    # and it was already simplified
    return t(lhs, rhs, evaluate=False)


def _reduce_to_lowest_terms(expr: sympy.Expr) -> sympy.Expr:
    """
    Eliminates any integer factor from a given expression.
    E.g., 6x + 4y reduces to 3x + 2y.

    Useful when an expression is == or != to 0.
    """

    def integer_coefficient(x: sympy.Expr) -> int:
        if x.is_Integer:
            return abs(int(x))
        elif x.is_Mul:
            # If one of the args of a Mul is an Integer, it is the
            # first arg. eg: args(2*x*3*y) == (6, x, y)
            return abs(int(x.args[0])) if x.args[0].is_Integer else 1  # type: ignore[call-overload]
        else:
            return 1

    def div_by_factor(x: sympy.Expr, factor: int) -> sympy.Expr:
        if x.is_Integer:
            return x / factor
        elif x.is_Mul:
            if x.args[0] != factor:
                args = [x.args[0] / sympy.Integer(factor), *x.args[1:]]
            else:
                # Mul._from_args require a canonical list of args
                # so we remove the first arg (x.args[0] / factor) if it was 1
                args = list(x.args[1:])
            return _sympy_from_args(sympy.Mul, args, is_commutative=x.is_commutative)
        else:
            raise AssertionError(f"illegal arg to div_by_factor: {x}")

    if expr.is_Add:
        atoms = cast(Sequence[sympy.Expr], expr.args)
        factor = functools.reduce(math.gcd, map(integer_coefficient, atoms))
        if factor == 1:
            return expr
        # pyrefly: ignore [bad-argument-type]
        atoms = [div_by_factor(x, factor) for x in atoms]
        return _sympy_from_args(
            sympy.Add, atoms, sort=True, is_commutative=expr.is_commutative
        )
    elif expr.is_Integer:
        return S.One
    elif expr.is_Mul:
        return div_by_factor(expr, integer_coefficient(expr))
    return expr


def is_nested_int(s: IntLikeType) -> TypeGuard[SymInt]:
    return isinstance(s, torch.SymInt) and s.node.is_nested_int()


IterateExprsAtom: TypeAlias = (
    SymInt | SymFloat | SymBool | int | float | bool | sympy.Basic | torch.Tensor
)
IterateExprs: TypeAlias = IterateExprsAtom | Sequence[IterateExprsAtom]


def _iterate_exprs(val: IterateExprs) -> Iterator[sympy.Basic]:
    """
    Recursively iterate through a value and yield all sympy expressions contained within it.

    This function traverses various data structures (tensors, lists, tuples, etc.) and extracts
    any symbolic expressions they contain. It's used for operations like finding free symbols
    in complex nested structures.

    Args:
        val: The value to extract sympy expressions from. Can be a symbolic type (SymInt, SymFloat, SymBool),
             a sympy expression, a primitive type (int, float, bool), a container (tuple, list),
             a sparse tensor, a regular tensor, None, or a torch.Generator.

    Yields:
        sympy.Basic: Each sympy expression found in the value.

    Raises:
        AssertionError: If the value is of an unsupported type.
    """
    # This is almost close enough to implement in terms of _iterate_nodes()
    # except that it needs to handle `list[sympy.Basic]` which _iterate_nodes()
    # can't handle.
    if isinstance(val, SymTypes):
        # This allow applies to the jagged layout NestedTensor case as
        # nested ints are not symbolic
        if is_symbolic(val):
            yield val.node.expr
    elif isinstance(val, SymNode):
        yield val.expr
    elif isinstance(val, sympy.Basic):
        yield val
    elif isinstance(val, (int, float, bool)):
        pass
    elif isinstance(val, (tuple, list)):
        for s in val:
            yield from _iterate_exprs(s)
    elif is_sparse_any(val):
        yield from _iterate_exprs(val.size())
    elif isinstance(val, torch.Tensor):
        yield from _iterate_exprs(val.size())
        yield from _iterate_exprs(val.stride())
        yield from _iterate_exprs(val.storage_offset())
    elif val is None:
        pass
    # see Note: [Generator arguments in AOTDispatcher]
    elif isinstance(val, torch.Generator) or is_opaque_value(val):
        pass
    elif isinstance(val, FakeScriptObject):
        pass
    else:
        raise AssertionError(f"cannot extract sympy expressions from {val} {type(val)}")


def _iterate_nodes(val: Any) -> Iterator[SymNode]:
    """
    Recursively iterate through a value and yield all SymNodes contained
    within it.
    """
    if isinstance(val, SymNode):
        yield val
    elif isinstance(val, py_sym_types):
        # This allow applies to the jagged layout NestedTensor case as
        # nested ints are not symbolic
        if is_symbolic(val):
            yield val.node
    elif isinstance(val, (tuple, list, torch.Size)):
        for s in val:
            yield from _iterate_nodes(s)
    elif isinstance(val, torch.Tensor):
        yield from _iterate_nodes(val.size())
        if not is_sparse_any(val):
            yield from _iterate_nodes(val.stride())
            yield from _iterate_nodes(val.storage_offset())


def free_symbols(val: IterateExprs) -> OrderedSet[sympy.Symbol]:
    """
    Recursively collect all free symbols from a value.

    This function traverses various data structures (tensors, lists, tuples, etc.) and extracts
    all sympy symbols contained within them. It's useful for finding all symbolic variables
    that a complex nested structure depends on.

    Args:
        val: The value to extract symbols from. Can be a symbolic type (SymInt, SymFloat, SymBool),
             a container (tuple, list), a tensor, or None.

    Returns:
        OrderedSet[sympy.Symbol]: An ordered set of all free symbols found in the value.
    """
    if val is None:
        return OrderedSet()

    itr = _iterate_exprs(val)

    # we need at least 1 to call union, so we hand code the identity
    try:
        first_expr = next(itr)
    except StopIteration:
        return OrderedSet()

    # TODO: Apparently, returning an OrderedSet here breaks
    # python test/distributed/tensor/test_dtensor_compile.py TestDTensorCompile.test_dtensor_dynamic
    return first_expr.free_symbols.union(*(e.free_symbols for e in itr))  # type: ignore[return-value]


def has_free_symbols(val: IterateExprs) -> bool:
    """Faster version of bool(free_symbols(val))"""
    return not all((e.is_number or e.is_Boolean) for e in _iterate_exprs(val))


def has_free_unbacked_symbols(x: IterateExprs) -> bool:
    """Faster version of bool(free_unbacked_symbols(val))"""
    from sympy.core.traversal import iterargs

    for s in _iterate_exprs(x):
        for arg in iterargs(s):
            if arg.is_Symbol and symbol_is_type(
                arg, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT)
            ):
                return True
    return False


def free_unbacked_symbols(x: IterateExprs) -> OrderedSet[sympy.Symbol]:
    """Like free_symbols, but filtered to only report unbacked symbols"""

    # NB: keep synced with is_unbacked_symint
    return OrderedSet(
        s
        for s in free_symbols(x)
        if symbol_is_type(s, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT))
    )


def _free_non_source_unbacked_symbols(
    x: IterateExprs, unbacked_inputs: OrderedSet[sympy.Symbol]
) -> OrderedSet[sympy.Symbol]:
    """Unbacked symbols that are not inputs to the graph. These are symbols that originated from
    data-dependent operations as opposed to mark_unbacked calls."""
    unbacked_symbols = free_unbacked_symbols(x)
    non_source_symbols = unbacked_symbols - unbacked_inputs
    return non_source_symbols


# WARNING: Don't use this on Dynamo produced graphs, they don't have meta
# setup!
def is_symbol_binding_fx_node(node: torch.fx.Node) -> sympy.Symbol | None:
    """
    Check if a given FX node is a symbol binding node.

    A symbol binding node is one that has a SymInt value in its meta that contains
    a sympy Symbol expression, and is either a placeholder node or contains unbacked symbols.

    Args:
        node (torch.fx.Node): The FX node to check

    Returns:
        Optional[sympy.Symbol]: The sympy Symbol if the node is a symbol binding node, None otherwise
    """
    if (
        "val" in node.meta
        and isinstance(node.meta["val"], torch.SymInt)
        and isinstance(node.meta["val"].node.expr, sympy.Symbol)
        and (
            node.op == "placeholder"
            or free_unbacked_symbols(node.meta["val"].node.expr)
        )
    ):
        return node.meta["val"].node.expr
    return None


def find_symbol_binding_fx_nodes(
    graph: torch.fx.Graph,
) -> dict[sympy.Symbol, torch.fx.Node]:
    """
    Find all nodes in an FX graph that bind sympy Symbols.

    This function scans through all nodes in the given FX graph and identifies
    nodes that bind sympy Symbols (typically placeholder nodes with SymInt values).
    When multiple nodes bind the same symbol, only the first occurrence is kept.

    Args:
        graph: The FX graph to search for symbol binding nodes

    Returns:
        A dictionary mapping from sympy Symbols to their binding FX nodes
    """
    r = {}
    # NB: Prefer first occurrence of symbol
    for node in graph.nodes:
        if (s := is_symbol_binding_fx_node(node)) is not None and s not in r:
            r[s] = node
    return r


@dataclass(frozen=True, slots=True)
class Specialization:
    """
    This class is used in multi-graph compilation contexts where we generate
    multiple specialized graphs and dispatch to the appropriate one at runtime.
    This allows us to optimize the trade-off between performance and generality
    by creating specialized versions for common patterns (e.g., x.shape[0] % 16 == 0)
    while maintaining a general fallback.
    """

    source: TensorPropertySource
    check_fn: Callable


# Analogous to ConvertIntSource
@dataclass(frozen=True, slots=True)
class ConvertIntKey:
    def __str__(self) -> str:
        return ".cast_symbool_to_symint_guardless()"

    def get(self, b: bool) -> IntLikeType:
        """Get the int value from bool"""
        return cast_symbool_to_symint_guardless(b)


@dataclass(frozen=True, slots=True)
class CallMethodKey:
    name: str

    def __str__(self) -> str:
        return f".{self.name}()"

    def get(self, o: Any) -> Any:
        """Call the method on object"""
        return getattr(o, self.name)()


@dataclass(frozen=True, slots=True)
class InnerTensorKey:
    inner_name: str

    def __str__(self) -> str:
        return f".{self.inner_name}"

    def get(self, o: Any) -> Any:
        """Get the inner tensor attribute"""
        return getattr(o, self.inner_name)


@dataclass(frozen=True, slots=True)
class DivideByKey:
    divisor: IntLikeType

    def __str__(self) -> str:
        return f".__floordiv__({self.divisor})"

    def get(self, o: int) -> int:
        """Divide object by divisor"""
        return o // self.divisor


def _free_unbacked_symbols_with_path(
    a: object,
    path: pytree.KeyPath,
    real: object | None = None,
    shape_env: ShapeEnv | None = None,
    pending: set[sympy.Symbol] | None = None,
    simplify: bool = False,
) -> dict[sympy.Symbol, pytree.KeyPath]:
    """
    Recursively traverses a structure to find unbacked symbols and their access paths.

    This function walks through tensors, lists, tuples, and symbolic values to locate
    unbacked symbols that are in the pending set, and returns a mapping from those
    symbols to their access paths in the structure.

    Args:
        a: The object to traverse (tensor, list, tuple, SymInt, etc.)
        path: The current path in the object tree
        real: Optional real tensor corresponding to the fake tensor being traversed
        shape_env: Optional ShapeEnv to register unbacked values with
        pending: Set of unbacked symbols to look for (will be modified in-place)
        simplify: Whether to use simplified expressions

    Returns:
        A dictionary mapping unbacked symbols to their access paths
    """
    go = functools.partial(
        _free_unbacked_symbols_with_path,
        shape_env=shape_env,
        pending=pending,
        simplify=simplify,
    )

    def expr(s: SymInt | SymFloat | SymBool) -> sympy.Expr:
        if simplify:
            return s.node.expr
        # (When called from compute_unbacked_bindings)
        # NB: Intentionally access _expr, not expr, do not want
        # simplification!
        return s.node._expr

    if pending is None:
        pending = set()
    r = {}

    def match_tensor(a: torch.Tensor, real_tensor: torch.Tensor | None = None):
        r.update(
            go(
                a.size(),
                path + (CallMethodKey("size"),),
                real=real_tensor.size() if real_tensor is not None else None,
            )
        )
        if a.layout not in [
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ]:
            r.update(
                go(
                    a.stride(),
                    path + (CallMethodKey("stride"),),
                    real=real_tensor.stride() if real_tensor is not None else None,
                )
            )
        r.update(
            go(
                a.storage_offset(),
                path + (CallMethodKey("storage_offset"),),
                real=(
                    real_tensor.storage_offset() if real_tensor is not None else None
                ),
            )
        )

    if isinstance(a, (tuple, list)):
        # NB: real is apparently not always a tuple/list here
        # python test/inductor/test_torchinductor.py CpuTests.test_index_propagation_nested_indirect_indexing_cpu
        for i in range(len(a)):
            r.update(
                go(
                    a[i],
                    path + (pytree.SequenceKey(i),),
                    real=real[i] if real is not None else None,  # type: ignore[index]
                )
            )
    elif is_traceable_wrapper_subclass(a):
        # TODO: Determine if this is correct
        attrs, _ = a.__tensor_flatten__()
        for attr in attrs:
            sub = getattr(a, attr)
            r.update(go(sub, path + (InnerTensorKey(attr),)))

        # match DTensor outer shapes
        if torch.distributed.is_available() and isinstance(
            a, torch.distributed.tensor.DTensor
        ):
            match_tensor(a)
    elif isinstance(a, torch.Tensor) and is_batchedtensor(a):
        unwrapped_tensor = get_unwrapped(a)
        r.update(go(unwrapped_tensor, path))
    elif isinstance(a, torch.Tensor) and not is_batchedtensor(a):
        from torch._subclasses.fake_tensor import FakeTensor

        if not isinstance(a, FakeTensor):
            raise AssertionError(f"Expected FakeTensor, got {type(a)}")
        match_tensor(a, a.real_tensor)
    elif (
        isinstance(a, (torch.SymInt, torch.SymFloat))
        and isinstance(s := expr(a), sympy.Symbol)
        and s in pending
    ):
        r[s] = path
        if shape_env and real is not None:
            if not isinstance(real, (int, float)):
                raise AssertionError(f"Expected int or float, got {type(real)}")

            shape_env.set_real_tensor_prop_unbacked_vals(s, real)

        pending.remove(s)
    # When an unbacked SymInt is perfectly divisible by an integer
    # constant, we replace it with the integer constant to improve
    # reasoning capabilities.  However, in synthetic examples, it is
    # then possible that the factor never is explicitly allocated.
    # Fortunately, we can compute it by division.
    elif (
        isinstance(a, torch.SymInt)
        and isinstance(s := expr(a), sympy.Mul)
        and len(s.args) == 2
        and isinstance(lhs := s.args[0], (sympy.Integer, sympy.Symbol))
        and isinstance(rhs := s.args[1], sympy.Symbol)
        # support exactly one unbacked for now
        and ((rhs in pending) ^ (lhs in pending))
        # support constant coefficient or backed symbolic coefficient
        and (
            isinstance(coeff := lhs if lhs not in pending else rhs, sympy.Integer)
            or shape_env
            and coeff in shape_env.backed_var_to_val
        )
    ):

        def _symint_wrap(s: sympy.Symbol) -> SymInt:
            return shape_env.create_symintnode(  # type: ignore[union-attr]
                s,
                hint=int(shape_env.backed_var_to_val[s]),  # type: ignore[union-attr]
                source=shape_env.var_to_sources.get(s, [None])[0],  # type: ignore[union-attr]
            )

        unbacked = lhs if lhs in pending else rhs
        divisor: IntLikeType = (
            int(coeff)
            if shape_env and isinstance(coeff, sympy.Integer)
            else _symint_wrap(coeff)
        )
        # TODO: DivideByKey needs to test divisibility at runtime!

        # pyrefly: ignore [unsupported-operation]
        r[unbacked] = path + (DivideByKey(divisor),)
        if real is not None:
            if not isinstance(real, int):
                raise AssertionError(f"Expected int, got {type(real)}")
            val = (
                real // int(coeff)
                if isinstance(coeff, sympy.Integer)
                else CleanDiv(real, coeff)
            )
            if shape_env:
                shape_env.set_real_tensor_prop_unbacked_vals(unbacked, val)
        pending.remove(unbacked)
    # The annoyance here arises from the fact that SymBool is
    # allocated by allocating a SymInt and then testing if it's equal
    # to one.  So you have a complicated binding site logic for this.
    elif (
        isinstance(a, torch.SymBool)
        and isinstance(s := expr(a), sympy.Eq)
        # This must match create_unbacked_symbool EXACTLY
        and isinstance(s.lhs, sympy.Symbol)
        and s.rhs == 1
        and s.lhs in pending
    ):
        # pyrefly: ignore [unsupported-operation]
        r[s.lhs] = path + (ConvertIntKey(),)
        if real is not None:
            if type(real) is not bool:
                raise AssertionError(f"Expected bool, got {type(real)}")
            if shape_env:
                shape_env.set_real_tensor_prop_unbacked_vals(s, int(real))

        pending.remove(s.lhs)

    return r


def compute_unbacked_bindings(
    shape_env: ShapeEnv | None,
    example_value: object,
    old_example_value: object | None = None,
    peek: bool = False,
) -> dict[sympy.Symbol, pytree.KeyPath] | None:
    """
    After having run fake tensor propagation and producing example_value
    result, traverse example_value looking for freshly bound unbacked
    symbols and record their paths for later.  It is an error if
    we have allocated an unbacked SymInt but it cannot be found in
    example_value.  (NB: this means if you have a multi-output
    function, you must call this on the tuple of tensor output, you
    cannot wait!)

    The peek parameter lets you check out what the bindings are without
    changing the affected list.  This is primarily useful for ensuring
    real_tensor_prop_unbacked_vals is promptly populated when propagate_real_tensors is on.
    """
    if shape_env is None:
        return None

    fresh_sym = shape_env.pending_fresh_unbacked_symbols
    ign_sym = shape_env.ignorable_fresh_unbacked_symbols

    pending = set(fresh_sym)
    ignorable = set(ign_sym)
    if not peek:
        if pending:
            log.info("compute_unbacked_bindings %s", fresh_sym)
        fresh_sym.clear()
        ign_sym.clear()

    if not pending:
        return None

    symbol_to_path = _free_unbacked_symbols_with_path(
        example_value, (), shape_env=shape_env, pending=pending, simplify=False
    )

    pending -= ignorable
    if not peek and pending:
        extra = (
            repr((example_value.stride(), example_value.storage_offset()))
            if isinstance(example_value, torch.Tensor)
            else ""
        )
        msg = (
            f"Pending unbacked symbols {pending} not in returned outputs {example_value} {extra}.\n"
            "Did you accidentally call new_dynamic_size() or item() more times "
            "than you needed to in your fake implementation?\n"
            "For more help, see https://docs.google.com/document/d/1RWrH-3wLEpzR9kCS6gGBNen_-Fs-8PVbWWFE5AcgeWE/edit"
        )
        if torch.fx.experimental._config.soft_pending_unbacked_not_found_error:
            log.warning(msg)
        else:
            raise PendingUnbackedSymbolNotFound(msg)

    # Why do we have to do some rebinding here?  If the original FX node
    # wasn't a binding site because you had a memo hit, but post
    # translation you aren't a memo hit anymore, there's now a new binding
    # site... but we know (because it's the same FX node) that the value
    # is actually the same, they're just not obviously equal anymore.
    #
    # The logic here is written carefully, because unlike the
    # bind_unbacked case, we are not guaranteed to have a symbol for
    # old_sym.  If we have a symbol, do regular rename unbacked to; but if
    # we don't, we need to specially eliminate the fresh unbacked symbol
    # (NB: we are /trusting/ that the memoization is correct, and that we
    # don't need to generate a new runtime assert.  This is load bearing,
    # as repropagation can happen after we've frozen runtime asserts.)
    if old_example_value is not None:
        for keypath in symbol_to_path.values():
            old_sym = pytree.key_get(old_example_value, keypath)
            new_sym = pytree.key_get(example_value, keypath)
            if isinstance(new_sym, SymTypes) and isinstance(
                new_s := new_sym.node.expr, sympy.Symbol
            ):
                if (
                    isinstance(old_sym, SymTypes)
                    and (old_s := old_sym.node.expr) != new_s
                ):
                    # If old_s is not an unbacked_symbol,
                    # we assume that the original unbacked symbol is replaced
                    # by a backed symbol (old_s). This can happen
                    # when this node reuses the original symbol (due to memoi)
                    # and the original symbol gets replaced by the backed symbol.
                    # When this happens we just replace new_s by the old_s
                    # because we know the value is the same.

                    if isinstance(old_s, sympy.Symbol) and free_unbacked_symbols(old_s):
                        shape_env._rename_unbacked_to(new_s, old_s)
                    else:
                        shape_env._eliminate_unbacked(new_s, old_s)
                elif not isinstance(old_sym, SymTypes):
                    shape_env._eliminate_unbacked(new_s, sympy.sympify(old_sym))

    return symbol_to_path


# Note [guard_or_]
# The following two functions are common utilities used while defining unbacked semantics
# of various framework code. Those would be used in situations you prefer to guard and know
# the result of the expression over not guarding, but in case you hit a data dependent error
# you are ok with just returning true or false.
#
# When to use this?
# (1) If you can use a higher level combinator prefer using those instead, they are definitely safe (modulo short-circuiting).
#
# (2) It can be used if the program would behave equivalently if _guard_or returned true or false.
# Many inductor optimizations fall in this bracket for example.
#
# (3) Finally, it's even be OK if the program wouldn't behave equivalently, so long as the
# change is semantics preserving.  It can be semantics preserving if the program errors in more
# cases than it did previously (but otherwise behaves identically), or if it changes some quantity
# in a way that doesn't matter (e.g., strides often fall in this bucket.)
#
# (4) Specialize for the general case and add a runtime assertion that would fail during
#     runtime if the conditions for the general case are not satisfied. Examples for this are;
#      assuming expand/reshape inputs are not -1. or assuming the non-broadcasting path.
#
def _guard_or(a: BoolLikeType, default: bool) -> bool:
    """
    Try to guard a, if data dependent error encountered just return default.
    """
    if not isinstance(a, SymBool):
        if not isinstance(a, bool):
            raise AssertionError(f"Expected bool, got {type(a)}")
        return a

    # if backed_size_oblivious is True we treat backed as unbacked here.
    if torch.fx.experimental._config.backed_size_oblivious:
        result = _static_eval_sym_bool(a)
        return result if result is not None else default

    shape_env = getattr(a.node, "shape_env", None)

    # xla symnode path.
    if shape_env is None:
        return guard_bool(a)

    sym_node = a.node
    r = sym_node.shape_env.evaluate_sym_node(
        sym_node, size_oblivious=False, fallback_value=default
    )
    return bool(r)


def guard_or_false(a: BoolLikeType) -> bool:
    """
    Try to guard a, if data dependent error encountered just return false.
    """
    return _guard_or(a, False)


def guard_or_true(a: BoolLikeType) -> bool:
    """
    Try to guard a, if data dependent error encountered just return true.
    """
    return _guard_or(a, True)


def _static_eval_sym_bool(x: SymBool) -> bool | None:
    if not isinstance(x, SymBool):
        raise AssertionError(f"Expected SymBool, got {type(x)}")
    expr = x.node.expr

    try:
        # Shape env access is inside the try on purpose. xla symnode does not
        # have it on its attributes.
        shape_env = x.node.shape_env
        simplified = shape_env._maybe_evaluate_static(expr)
        if simplified is not None:
            return bool(simplified)
        else:
            return None
    except Exception:
        log.debug("Could not simplify %s", expr)
        return None


def statically_known_false(x: BoolLikeType) -> bool:
    """
    Returns True if x can be simplified to a constant and is False.
    If x cannot be evaluated from static, we return False

    .. note::
        This function doesn't introduce new guards, so the expression may end
        up evaluating to False at runtime even if this function returns False.

    Args:
        x (bool, SymBool): The expression to try statically evaluating
    """
    if not isinstance(x, SymBool):
        if not isinstance(x, bool):
            raise AssertionError(f"Expected bool, got {type(x)}")
        return not x

    result = _static_eval_sym_bool(x)
    if result is None:
        return False

    return not result


def statically_known_true(x: BoolLikeType) -> bool:
    """
    Returns True if x can be simplified to a constant and is true.

    .. note::
        This function doesn't introduce new guards, so the expression may end
        up evaluating to true at runtime even if this function returns False.

    Args:
        x (bool, SymBool): The expression to try statically evaluating
    """
    if not isinstance(x, SymBool):
        if not isinstance(x, bool):
            raise AssertionError(f"Expected bool, got {type(x)}")
        return x
    result = _static_eval_sym_bool(x)
    if result is None:
        return False

    return result


def sym_and(x: BoolLikeType, *others: BoolLikeType) -> BoolLikeType:
    """
    and, but for symbolic expressions, without bool casting.
    """
    if len(others) == 0:
        return x
    for y in others:
        x = operator.and_(x, y)
    return x


def sym_eq(x: _T, y: _T) -> BoolLikeType:
    """
    Like ==, but when run on list/tuple, it will recursively test equality
    and use sym_and to join the results together, without guarding.
    """
    if isinstance(x, (tuple, list)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            return False
        return functools.reduce(operator.and_, map(sym_eq, x, y), True)
    elif isinstance(x, (int, torch.SymInt)) and isinstance(y, (int, torch.SymInt)):
        return x == y
    else:
        raise AssertionError(f"unexpected sym_eq between {type(x)} {type(y)}")


def sym_or(x: BoolLikeType, *others: BoolLikeType) -> BoolLikeType:
    """
    or, but for symbolic expressions, without bool casting.
    """
    if len(others) == 0:
        return x
    for y in others:
        x = operator.or_(x, y)
    return x


def guard_scalar(
    a: SymBool | SymInt | SymFloat | int | bool | float,
) -> bool | int | float:
    """
    Guard a scalar value, which can be a symbolic or concrete boolean, integer, or float.

    This function dispatches to the appropriate guard function based on the type of the input.

    Args:
        a: A symbolic or concrete scalar value (bool, int, or float)

    Returns:
        The concrete value after guarding

    Raises:
        AssertionError: If the input is not a recognized scalar type
    """
    if isinstance(a, (SymBool, bool)):
        return guard_bool(a)
    elif isinstance(a, (SymInt, int)):
        return guard_int(a)
    elif isinstance(a, (SymFloat, float)):
        return guard_float(a)
    else:
        raise AssertionError(f"unrecognized scalar {a}")


def _advise_is_size(a: SymInt) -> None:
    """
    Don't use this directly; use torch._check_is_size instead.

    This is a softer version of _constrain_range_for_size (with min=0,
    max=Inf).  Instead of forcibly constraining a variable (and erroring if we
    failed to constrain it), it will simply advise us that a size is
    constrained in some way.  We will always defer a runtime assert for this
    constraint if we cannot prove it at compile-time, but we we only
    *sometimes* learn useful extra information at compile-time with this
    information.  This is in contrast to constrain_range_for_size, where if
    you don't call that on a fresh unbacked symint, chances are we will choke.

    TODO: Make Dynamo handle this appropriately if this is seen in Dynamo-ed
    code.  Right now this is only really used in code with AOTAutograd trace
    through, so it is not a big problem that this isn't supported, but in
    principle all of this code should be Dynamo'able too.

    TODO: I didn't support min/max because I didn't have a use case where this
    actually helped.  In principle we can support it, it just makes the
    implementation below more complicated.
    """

    # This must always succeed, because the sole allowed caller _check_is_size
    # was responsible for expect_true'ing this
    # This assert triggers expensive sym compute, do not do it until its cheap.
    # assert a >= 0

    # NB: it's important not to constrain range for size for *hinted* SymInts,
    # because it is not only unsound, it will immediately trip our asserts
    # that hints have to be consistent with static analysis!  If you somehow
    # have an unbounded SymInt that later constrains to 1, this will be
    # inconsistent with the range
    if (
        isinstance(a, SymInt)
        and isinstance(a.node, SymNode)
        and isinstance(a.node.expr, sympy.Symbol)
        and a.node.shape_env.is_unbacked_symint(a.node.expr)
    ):
        _constrain_range_for_size(a)


def _advise_is_bounded(a: SymInt, upper_bound: IntLikeType) -> None:
    if (
        isinstance(a, SymInt)
        and isinstance(a.node, SymNode)
        and isinstance(a.node.expr, sympy.Symbol)
        and a.node.shape_env.is_unbacked_symint(a.node.expr)
        and isinstance(upper_bound, int)  # TODO: relax
    ):
        a.node.shape_env._constrain_is_bounded(a.node.expr, upper_bound)


def _constrain_range_for_size(
    a: SymInt, min: int | None = None, max: int | None = None
) -> None:
    """
    This function is NOT INTENDED to be used by itself.
    """

    if isinstance(a, (SymFloat, SymBool)):
        raise ValueError("Constraining SymFloat/SymBool is nyi")

    if not isinstance(a, SymInt):
        raise AssertionError("can only constrain range for SymInt")
    if not isinstance(a.node.expr, sympy.Symbol):
        raise AssertionError(f"constraining non-Symbols NYI: {a}")

    a.node.shape_env._constrain_range_for_size(a.node.expr, min, max)


# inclusive both ways
def constrain_range(a: SymInt, *, min: int | None, max: int | None = None) -> None:
    """
    Applies a constraint that the passed in SymInt must lie between min-max
    inclusive-inclusive, WITHOUT introducing a guard on the SymInt (meaning
    that it can be used on unbacked SymInts).  If min/max are None, we assume
    that the dimension is unbounded in that direction.  Repeated application
    of constrain_range intersects the ranges.  This is a fairly low level API
    that doesn't have a lot of safety guarantees (TODO: provide higher level
    APIs).

    Currently, we use this API in the following circumstance: when we allocate
    an unbacked SymInt, denoting an integer quantity which is data dependent,
    we ordinarily do not know anything about what values it may take.  This
    means that any sort of guard on it will immediately fail.  However, in
    many cases, we know something about the unbacked SymInt: for example, we
    know that nonzero(x).size(0) must be >= 0.  We use constrain_range to
    narrow the possible range, declaring that negative symbols are impossible.
    This permits to definitely answer True to queries like 'nnz >= 0', even if
    we don't know what the actual (hinted) value of 'nnz' is.  In fact, we
    actually use constrain_range to unsoundly discharge common guards: for an
    unbacked SymInt produced by nonzero, we will also assume that it is not
    equal to 0/1 (even though these are perfectly possible values at runtime),
    because we generally expect graphs that are valid for N=2 to also be valid
    for N=1.
    """
    if min is None:
        min = -int_oo
    if max is None:
        max = int_oo

    if max < min:
        raise ValueError(
            "Maximum value to constrain_as_size can't be less than the specified min value, "
            f"received min={min} and max={max}"
        )

    if isinstance(a, int):
        if not (min <= a <= max):
            raise ValueError(f"Invalid value {a} for range [{min}:{max}]")
        return

    a.node.shape_env._constrain_range(a.node.expr, min, max)


def constrain_unify(a: torch.SymInt, b: torch.SymInt) -> None:
    """
    Given two SymInts, constrain them so that they must be equal.  NB:
    this will not work with SymInts that represent nontrivial expressions
    (yet!)
    """
    if not isinstance(a, SymInt):
        if not isinstance(b, SymInt):
            if a != b:
                raise AssertionError(f"Expected {a} == {b}")
            return
        else:
            shape_env = b.node.shape_env
    else:
        shape_env = a.node.shape_env

    shape_env._constrain_unify(a, b)


# Assume that a boolean is true for the purposes of subsequent symbolic
# reasoning.  This will keep track of corresponding runtime checks to verify
# that the result is upheld: either as a regular guard, or as a special set
# of asserts which are triggered when an unbacked SymInt is allocated.
#
# DO NOT use this function for these cases:
#
#  - This is inappropriate for "branching" conditions (where both
#    true and false result in valid programs).  We will always assume
#    the condition evaluates true, and so it will never be possible
#    to trace the false condition when you use it.  For true branching
#    on unbacked SymInts, you must use torch.cond; if you incorrectly
#    use expect_true in this case, you will make the false branch
#    unreachable (as we will simply assume that only the true branch
#    is ever exercised).
#
#  - This is inappropriate for situations where you know some other system
#    invariant guarantees that this property holds, since you don't
#    really need to insert a runtime check in that case.  Use something
#    like constrain_range in that case.
#
# This API has a hitch.  To avoid having to reimplement error reporting
# capabilities, this function CAN return False.  The invariant is that
# the surrounding code must raise an error when this function returns
# False.  This is quite low level, so we recommend using other functions
# like check() which enforce this in a more intuitive way.
#
# By the way, this name is a nod to the __builtin_expect macro,
# which is used similarly (but unlike __builtin_expect, you MUST fail
# in the unlikely branch.)  (I think expect is a good name; in recent
# versions of C++, this is replaced with [[likely]], which is weaker
# and not accurate for this function!)
def expect_true(a: BoolLikeType, skip: int = 0) -> bool:
    if isinstance(a, SymBool):
        # TODO: check perf implications of this
        frame = inspect.currentframe()
        for _ in range(skip + 1):  # always run this loop at least once
            if frame is None:
                break
            frame = frame.f_back
        return a.node.expect_true(
            frame.f_code.co_filename if frame else "", frame.f_lineno if frame else 0
        )
    if type(a) is not bool:
        raise AssertionError(f"Expected bool, got {a}")
    return a


def guard_bool(a: BoolLikeType) -> bool:
    if isinstance(a, SymBool):
        return a.node.guard_bool("", 0)  # NB: uses Python backtrace
    if type(a) is not bool:
        raise AssertionError(f"Expected bool, got {a}")
    return a


def guard_int(a: IntLikeType) -> int:
    if isinstance(a, SymInt):
        return a.node.guard_int("", 0)  # NB: uses Python backtrace
    if type(a) is not int:
        raise AssertionError(f"Expected int, got {a}")
    return a


def guard_float(a: FloatLikeType) -> float:
    if isinstance(a, SymFloat):
        return a.node.guard_float("", 0)  # NB: uses Python backtrace
    if not isinstance(a, float):
        raise AssertionError(f"Expected float, got {a}")
    return a


# Given a GraphModule, return all the FakeTensors for all the placeholders
def fx_placeholder_vals(gm: torch.fx.GraphModule) -> list[object]:
    return [n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder"]


def fx_placeholder_targets(gm: torch.fx.GraphModule) -> list[str]:
    return [n.target for n in gm.graph.nodes if n.op == "placeholder"]


# Given a GraphModule and arguments to run it with, evaluate that the guards
# for its associated ShapeEnv are satisfied by the passed arguments.  This
# WILL check for duck sizing.
def eval_guards(
    gm: torch.fx.GraphModule, *args: Tensor, ignore_static: bool = True
) -> bool:
    if gm.shape_env is None:
        raise AssertionError("gm.shape_env must not be None")
    return gm.shape_env.evaluate_guards_for_args(  # type: ignore[operator, union-attr]
        fx_placeholder_vals(gm), args, ignore_static=ignore_static
    )


def bind_symbols(gm: torch.fx.GraphModule, *args: Tensor) -> dict[sympy.Symbol, int]:
    if gm.shape_env is None:
        raise AssertionError("gm.shape_env must not be None")
    return gm.shape_env.bind_symbols(fx_placeholder_vals(gm), args)  # type: ignore[operator, union-attr]



def error() -> NoReturn:
    raise AssertionError("shouldn't be hit")


# TODO: Deduplicate this with torch/_prims_common/__init__.py
def eval_is_non_overlapping_and_dense(
    sizes: Sequence[int], strides: Sequence[int]
) -> int:
    return int(guard_bool(_eval_is_non_overlapping_and_dense(sizes, strides)))


def _eval_is_non_overlapping_and_dense(
    sizes: Sequence[int], strides: Sequence[int]
) -> bool:
    """
    Evaluates whether a tensor with the given sizes and strides is non-overlapping and dense.

    A tensor is non-overlapping if there's no memory location that belongs to more than one element.
    A tensor is dense if all elements are stored in memory without gaps.

    Args:
        sizes: Sequence of dimension sizes for the tensor
        strides: Sequence of strides for the tensor

    Returns:
        True if the tensor is non-overlapping and dense, False otherwise
    """
    dim = len(sizes)

    # Short-circuits for tensors of rank one, which are
    # non-overlapping and "dense" if their stride is one
    # or it is a 0/1 element tensor
    if dim == 1:
        return strides[0] == 1 or sizes[0] < 2

    # Checks that there exists a permutation of the strides s.t. the tensor would be contiguous
    # Sorts (length, stride) pairs by stride
    lengths_and_strides = sorted(zip(sizes, strides), key=operator.itemgetter(1))

    # Unlike the C++ code, we don't move the 0/1 size dimensions to the
    # end.  So we have to keep going for this code.
    expected_stride = 1
    for length, stride in lengths_and_strides:
        if length == 1:
            continue

        if stride != expected_stride:
            return False

        expected_stride *= length

    return True


def _sympy_cast_symbool_to_symint_guardless(x: SympyBoolean) -> sympy.Expr:
    return sympy.Piecewise((1, x), (0, True))


def cast_symbool_to_symint_guardless(
    symbool: bool | torch.SymBool,
) -> int | torch.SymInt:
    """
    Converts a SymBool or bool to a SymInt or int without introducing guards.

    This function maps True to 1 and False to 0, preserving the symbolic nature
    of the input when it's a SymBool. Unlike regular casting which might introduce
    guards, this function performs the conversion without adding any guards.

    Args:
        symbool: A boolean value, either a concrete bool or symbolic SymBool

    Returns:
        The corresponding integer value (1 for True, 0 for False) as either
        a concrete int or symbolic SymInt
    """
    if isinstance(symbool, bool):
        return 1 if symbool else 0
    int_sym = _sympy_cast_symbool_to_symint_guardless(symbool.node.expr)
    return symbool.node.shape_env.create_symintnode(
        int_sym,
        hint=guarding_hint_or_throw(symbool) if has_guarding_hint(symbool) else None,
    )


def _eval_is_non_overlapping_and_dense_flat(*args: int) -> int:
    # Guard code strings print IsNonOverlappingAndDenseIndicator with flat args
    # (s0, s1, ..., stride0, stride1, ...) but eval_is_non_overlapping_and_dense
    # expects two sequences (sizes, strides). This wrapper bridges the gap.
    dim = len(args) // 2
    return eval_is_non_overlapping_and_dense(list(args[:dim]), list(args[dim:]))


SYMPY_INTERP = {
    "IsNonOverlappingAndDenseIndicator": _eval_is_non_overlapping_and_dense_flat,
    "cast_symbool_to_symint_guardless": cast_symbool_to_symint_guardless,
    "math": math,
    "torch": torch,
}


def _lru_cache(
    fn: Callable[..., _T], maxsize: int | None = None
) -> functools._lru_cache_wrapper[_T]:
    """
    Wrapper around lru_cache that clears when new info about shapes has been
    updated.

    Use lru_cache if the output is always the same, regardless of the
    constraints we know now (i.e. evaluate_expr)

    Use _lru_cache otherwise.

    Also note that this depends on _update_version_counter being called on the
    shape environment whenever the constraints are updated, otherwise the cache
    will not be cleared.
    """
    fn_cache = lru_cache(maxsize)(fn)
    prior_version = 0

    if config.validate_shape_env_version_key:
        prior_key = None

        @functools.wraps(fn)
        def wrapper(self: ShapeEnv, *args: Any, **kwargs: Any) -> _T:
            nonlocal prior_version, prior_key
            if prior_key is None:
                prior_key = self._get_key()

            if prior_version != self._version_counter:
                fn_cache.cache_clear()
                prior_version = self._version_counter
                prior_key = self._get_key()
            else:
                if prior_key != self._get_key():
                    raise AssertionError(
                        "ShapeEnv cache key changed without version being updated!"
                    )

            return fn_cache(self, *args, **kwargs)

    else:

        @functools.wraps(fn)
        def wrapper(self: ShapeEnv, *args: Any, **kwargs: Any) -> _T:  # type: ignore[misc]
            nonlocal prior_version
            if prior_version != self._version_counter:
                fn_cache.cache_clear()
                prior_version = self._version_counter

            return fn_cache(self, *args, **kwargs)

    wrapper.cache_clear = fn_cache.cache_clear  # type: ignore[attr-defined]
    wrapper.cache_info = fn_cache.cache_info  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]

def _is_int(expr: object) -> TypeGuard[SymInt]:
    return isinstance(expr, SymInt) and expr.node.expr.is_number


# WARNING: This is legacy, DO NOT USE
def _is_dim_dynamic(t: torch.Tensor, d: int) -> bool:
    return hasattr(t, "_dynamo_dynamic_indices") and d in t._dynamo_dynamic_indices


class PropagateUnbackedSymInts(torch.fx.Interpreter):
    def run_node(self, n: torch.fx.Node) -> Result:
        """
        Run an FX node, propagating unbacked Symbol bindings to the new fake tensor
        """
        from torch._guards import detect_fake_mode

        result = super().run_node(n)
        fake_mode = detect_fake_mode()
        if fake_mode is None:
            raise AssertionError("fake_mode must not be None")
        rebind_unbacked(fake_mode.shape_env, n, result)
        return result


def _find_user_code_frame() -> types.FrameType | None:
    frame = inspect.currentframe()
    while frame is not None:
        if not frame.f_code.co_filename.startswith(
            os.path.dirname(inspect.getfile(torch)) + os.path.sep
        ):
            break
        frame = frame.f_back
    return frame


def _blame_user_code(e: Exception, frame: types.FrameType) -> None:
    frame_summary = traceback.FrameSummary(
        frame.f_code.co_filename,
        frame.f_lineno,
        frame.f_code.co_name,
    )
    msg = e.args[0]
    msg += "\n\nThe following call raised this error:\n" + "".join(
        traceback.StackSummary.from_list([frame_summary]).format()
    )
    e.args = (msg,)


class _PythonMsgPrinter(PythonPrinter):
    """
    Util printer that replaces sympy symbols with their source-level names
    and renders sympy relational operators (e.g., Eq, Ne, Ge, Le) inline
    (i.e., as ==, !=, >, <).
    """

    def __init__(self, src_map: dict[str, list[str]]) -> None:
        super().__init__()
        self.src_map = src_map

    def _print_Symbol(self, sym: sympy.Symbol) -> str:
        return self.src_map[sym.name][0]


def _suggest_torch_checks(
    e: GuardOnDataDependentSymNode, src_map: defaultdict[str, list[str]]
) -> None:
    """
    Enhances a GuardOnDataDependentSymNode error with suggested fixes using torch._check.

    This function analyzes the condition that caused the data-dependent error and generates
    user-friendly suggestions for fixing it by adding appropriate torch._check calls.
    It handles special cases like non-negative checks with specific recommendations.

    Args:
        e: The GuardOnDataDependentSymNode error to enhance with suggestions
        src_map: A mapping from symbol names to their corresponding source-level variable names

    Returns:
        None. Modifies the error message in-place by updating e.args[0].
    """
    # extract the unresolved condition on unbacked symints in the error
    cond = e.cond
    diff = ", ".join(s.name for s in cond.free_symbols if s.name not in src_map)
    if diff:
        log.warning("Unable to find user code corresponding to {%s}", diff)
        return
    printer = _PythonMsgPrinter(src_map)
    msg = e.args[0]
    msg += "\nTo fix the error, insert one of the following checks before this call:"

    not_cond_str = printer.doprint(sympy.Not(cond))

    # suggested fixes to resolve `cond` are to tell the compiler to assume
    # either `cond` or its negation (the user will need to select which)
    suggested_fixes = [
        f"torch._check({printer.doprint(cond)})",
        f"torch._check({not_cond_str})",
    ]

    for i, fix in enumerate(suggested_fixes):
        msg += f"\n  {i + 1}. {fix}"
    src_mapped = ", ".join(
        f"`{s}` with {' or '.join(src_map[s])}"
        for s in sorted(s.name for s in cond.free_symbols)
    )
    msg += f"\n\n(These suggested fixes were derived by replacing {src_mapped} in {cond} and its negation.)"
    e.args = (msg,)


def _suggest_fixes_for_data_dependent_error_non_strict(
    e: GuardOnDataDependentSymNode,
) -> None:
    """
    Given a raised data-dependent error, add the following to the error message:
    1. the closest user code location that raised the error;
    2. suggested fixes for the error in terms of live variables at that location.
    """

    # walk the stack up from the data-dependent error until a non-torch frame is found
    frame = _find_user_code_frame()
    if frame is not None:
        # add frame info to error message
        _blame_user_code(e, frame)

        # map symbol names reachable via frame locals to their source-level names
        src_map = defaultdict(list)
        for var, val in frame.f_locals.items():
            try:
                tree_leaves_with_path = pytree.tree_leaves_with_path(val)
            except ValueError:
                log.warning(
                    "pytree.tree_leaves_with_path failed for value of type {%s} in local variable {%s}",
                    type(val),
                    var,
                )
                continue
            # figure out how to access any symbol inside `val` through `var`
            for path, leaf in tree_leaves_with_path:
                name = var + pytree.keystr(path)
                if isinstance(leaf, torch.SymInt):
                    src_map[str(leaf.node.expr)].append(name)
                elif isinstance(leaf, torch.Tensor):
                    for i, dim in enumerate(leaf.shape):
                        if isinstance(dim, torch.SymInt):
                            src_map[str(dim.node.expr)].append(f"{name}.shape[{i}]")

        # add suggested torch.check()s based on `src_map` to the error message
        # replacing unbacked symints in the unresolved condition in the error
        if isinstance(e.cond, sympy.logic.boolalg.Boolean):
            _suggest_torch_checks(e, src_map)


@contextmanager
def _remove_effect_token_unbacked_bindings(
    node: torch.fx.Node,
) -> Generator[None, None, None]:
    """
    Temporarily modifies unbacked_bindings in a node's metadata by removing the first element
    of each path, which corresponds to an effect token.

    This is used when processing nodes that have effect tokens as the first element in their
    unbacked_bindings paths. The context manager ensures that the original bindings are
    restored after the operation is complete.

    Args:
        node: The FX node whose unbacked_bindings will be temporarily modified

    Yields:
        None
    """
    old_bindings = node.meta.get("unbacked_bindings", {})

    # Remove the extra layer for effect token
    new_bindings = {k: path[1:] if path else path for k, path in old_bindings.items()}

    node.meta["unbacked_bindings"] = new_bindings

    try:
        yield
    finally:
        node.meta["unbacked_bindings"] = old_bindings


# This helper function is used in passes that insert runtime assertions in the graph.
# When accessing expressions representing input placeholders, we do not apply replacements
# since those inputs should be seen by assertions that use them to be inserted. The only replacement
# that we apply is unbacked renaming.
def _get_placeholder_expr(sym_node: SymNode) -> sympy.Expr:
    shape_env = sym_node.shape_env
    result = sym_node._expr
    if result in shape_env.unbacked_renamings:
        return shape_env.unbacked_renamings[result]
    return result

__all__ = [
    "guarding_hint_or_throw",
    "optimization_hint",
    "GuardOnDataDependentSymNode",
    "PendingUnbackedSymbolNotFound",
    "_ShapeEnvGuardError",
    "aten",
    "SHAPEENV_EVENT_KEY",
    "CURRENT_NODE_KEY",
    "log_lru_cache_stats",
    "SympyBoolean",
    "_T",
    "_SympyT",
    "SymIntEqByExpr",
    "_nested_int_aware_sort",
    "lru_cache",
    "uninteresting_files",
    "ConstraintViolationError",
    "has_symbolic_sizes_strides",
    "Int",
    "create_contiguous",
    "Scalar",
    "has_guarding_hint",
    "is_concrete_int",
    "is_concrete_float",
    "is_concrete_bool",
    "has_static_value",
    "guard_size_oblivious",
    "check_consistent",
    "resolve_unbacked_bindings",
    "Result",
    "rebind_unbacked",
    "is_accessor_node",
    "canonicalize_bool_expr",
    "_sympy_from_args",
    "_canonicalize_bool_expr_impl",
    "_reduce_to_lowest_terms",
    "is_nested_int",
    "IterateExprsAtom",
    "IterateExprs",
    "_iterate_exprs",
    "_iterate_nodes",
    "free_symbols",
    "has_free_symbols",
    "has_free_unbacked_symbols",
    "free_unbacked_symbols",
    "_free_non_source_unbacked_symbols",
    "is_symbol_binding_fx_node",
    "find_symbol_binding_fx_nodes",
    "Specialization",
    "ConvertIntKey",
    "CallMethodKey",
    "InnerTensorKey",
    "DivideByKey",
    "_free_unbacked_symbols_with_path",
    "compute_unbacked_bindings",
    "_guard_or",
    "guard_or_false",
    "guard_or_true",
    "_static_eval_sym_bool",
    "statically_known_false",
    "statically_known_true",
    "sym_and",
    "sym_eq",
    "sym_or",
    "guard_scalar",
    "_advise_is_size",
    "_advise_is_bounded",
    "_constrain_range_for_size",
    "constrain_range",
    "constrain_unify",
    "expect_true",
    "guard_bool",
    "guard_int",
    "guard_float",
    "fx_placeholder_vals",
    "fx_placeholder_targets",
    "eval_guards",
    "bind_symbols",
    "error",
    "eval_is_non_overlapping_and_dense",
    "_eval_is_non_overlapping_and_dense",
    "_sympy_cast_symbool_to_symint_guardless",
    "cast_symbool_to_symint_guardless",
    "_eval_is_non_overlapping_and_dense_flat",
    "SYMPY_INTERP",
    "_lru_cache",
    "_is_int",
    "_is_dim_dynamic",
    "PropagateUnbackedSymInts",
    "_find_user_code_frame",
    "_blame_user_code",
    "_PythonMsgPrinter",
    "_suggest_torch_checks",
    "_suggest_fixes_for_data_dependent_error_non_strict",
    "_remove_effect_token_unbacked_bindings",
    "_get_placeholder_expr",
]
