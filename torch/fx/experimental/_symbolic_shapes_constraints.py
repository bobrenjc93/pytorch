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



class DimDynamic(Enum):
    """
    Controls how to perform symbol allocation for a dimension.  It is always
    sound to default this to DYNAMIC, but the policies DUCK and STATIC can
    result in better trace-time and compile-time performance, as they reduce
    the number of allocated symbols and generally make your graph more static.

    NB: If we notice you've applied a constraint to the dimension, we will
    force it to DYNAMIC for simplicity.

    DimDynamic is controlled by a variety of higher level UX features.
    Currently:

    - In eager mode, the default policy is DUCK.
        - The default is changed to STATIC with assume_static_by_default.
        - An individual dim is marked DYNAMIC if you mark_dynamic_dim.
    - In export mode, the default policy is STATIC.
        - An individual dim is marked DYNAMIC if you specify it in
          dynamic_shapes passed to export.
    """

    # Treat the dimension symbolically
    DYNAMIC = 0
    # Treat the dimension symbolically, but if its hint matches another
    # dynamic dimension, unify the two symbols ("duck sizing")
    DUCK = 1
    # Treat the dimension statically based on its hint
    STATIC = 2
    # Treat the dimension as unbacked
    UNBACKED = 3
    # Infer the strides from stride. If size is static, strides will be static as well.
    INFER_STRIDE = 4


# NB: These constraints affect both clients and backends: given some
# constraint C, the client must pass inputs that satisfy the constraint,
# while a backend must not introduce guards BEYOND this constraint.
# For clarity, we document the implications on both sides for both the client
# and the backend.
#
# NB: These constraints are on a *single* dimension.  In principle, we could
# also have multi-dimension constraints, but our guess is that this is not
# actually useful and so we are not supporting it right now.
#
# NB: Strict constraints are typically only suitable for export, as in eager
# a backend like inductor may validly introduce extra, discretionary guards
# to improve performance of code.  A StrictMinMaxConstraint would be brittle
# under future optimizations performed by inductor; we don't guarantee
# eager code with StrictMinMaxConstraint will keep working in the future!


@dataclass(frozen=True, slots=True)
class Constraint:
    warn_only: bool


@dataclass(frozen=True, slots=True)
class StrictMinMaxConstraint(Constraint):
    """
    For clients: the size at this dimension must be within 'vr' (which
    specifies a lower and upper bound, inclusive-inclusive) AND it
    must be non-negative and should not be 0 or 1 (but see NB below).

    For backends: there must not be any guards on this dimension which
    are not implied by the given lower and upper bound.  Regardless of
    the lower bound, the backend can assume the size is non-negative
    and that it is not 0 or 1.

    An unbounded StrictMinMaxConstraint can be thought of as a strict version
    of "RelaxedUnspecConstraint".

    NB: Export will often unsoundly assume that a graph works for 0/1, even
    though at trace time we assumed size is not 0 or 1.  The idea is that
    if we produce a graph that works for a range of values, it will be OK
    for N=0/1 too.
    """

    vr: ValueRanges

    def render(self, source: Source) -> str:
        """Format the constrain equation"""
        # TODO: better printing for -oo and oo
        return f"{self.vr.lower} <= {source.name} <= {self.vr.upper}"


@dataclass(frozen=True, slots=True)
class RelaxedUnspecConstraint(Constraint):
    """
    For clients: no explicit constraint; constraint is whatever is implicitly
    inferred by guards from tracing.

    For backends: there must exist at least TWO possible values for the
    size at this dimension which satisfy the guards for this dimension.

    In other words, this constraint helps us distinguish between "we don't
    care if this dimension specializes or not" versus "this dimension must be
    unspecialized."  However, this constraint doesn't say very much about what
    specialization is permitted; for example, if we guard on a size being
    even, this would still be acceptable under an unspec constraint.  This
    makes RelaxedUnspecConstraint useful for eager mode, where your backend compiler
    may add constraints to otherwise dynamic dimensions; we can't assert that
    there are NO guards as this is brittle because compilers should be able to
    add extra constraints.  If you want to assert that there are no guards,
    use StrictMinMaxConstraint with an unbounded ValueRanges.
    """

    def render(self, source: Source) -> str:
        return f"RelaxedUnspecConstraint({source.name})"


# NB: None here indicates the client constraint is whatever is implicitly
# inferred by guards from tracing, and that a backend can add whatever guards
# it wants (including fully specializing the value).
DimConstraint = StrictMinMaxConstraint | RelaxedUnspecConstraint | None


@dataclass(frozen=True, slots=True)
class EqualityConstraint(Constraint):
    """
    Represent and decide various kinds of equality constraints between input sources.

    A "source pair" is a pair of input sources for dynamic dimensions that
    are specified equal. We represent `source_pairs` in a union-find forest
    so that we can efficiently check whether two such sources are transitively equal.

    A "derived equality" relates an input source to an expression over a root.
    The root can be another input source, corresponding to some dynamic dimension,
    or a phantom symbol that does not directly represent any dynamic dimension. We
    represent `derived_equalities` involving input sources in a transitively-closed map
    so that we can efficiently check whether an input source is transitively equal to
    a given expression over another input source.
    (NOTE: In contrast, it is easy to decide whether an input source is transitively equal
    to a given expression over a phantom symbol; such expressions are already in canonical
    form and so the problem reduces to symbolic expression equality.)
    """

    source_pairs: list[tuple[Source, Source]]
    derived_equalities: list[
        tuple[Source, Source | sympy.Symbol, Callable[[sympy.Expr], sympy.Expr]]
    ]
    phantom_symbols: list[sympy.Symbol]
    relaxed_sources: set[Source]

    _parents: dict[Source, Source] = field(init=False)
    _defs: dict[Source, sympy.Expr] = field(init=False)

    def __post_init__(self) -> None:
        """
        Pre-processing to answer queries `is_equal` and `is_derived` below.

        Example: Suppose we are given:
          source_pairs [a = b, b = c]
          derived_equalities [d = c + 1, e = d - 1]
        We first construct a union find with source_pairs:
          _parents = {a: a, b: a, c: a}
        Then we compute canonical symbolic expressions, recursively applying derived_equalities
        until we bottom out:
          _defs = {d: c + 1, e: (c + 1) - 1 aka c}
        """

        # self._parents is a map from input sources to input sources where, conceptually,
        # these are directed edges in a union-find forest
        _parents: dict[Source, Source] = {}
        object.__setattr__(self, "_parents", _parents)
        # self._defs is a map from input sources to "canonical" symbolic expressions,
        # i.e., unary expressions with symbols that corresponds to regular Dims (i.e.,
        # not derived Dims)
        _defs: dict[Source, sympy.Expr] = {}
        object.__setattr__(self, "_defs", _defs)

        for source1, source2 in self.source_pairs:
            # preprocess into a union-find forest
            self._union(self._find(source1), self._find(source2))
        for source, root, fn in self.derived_equalities:
            # preprocess into a transitively-closed map
            # NOTE(avik): we reuse the union-find forest for canonicalizing input sources
            if isinstance(root, (sympy.Symbol, sympy.Integer)):
                self._defs[self._find(source)] = fn(root)
            else:
                self._defs[self._find(source)] = fn(self._rewrite(root))

    def _find(self, source: Source) -> Source:
        # chase edges to find the root of this equivalence class
        if source in self._parents:
            return self._find(self._parents[source])
        else:
            return source

    def _union(self, root1: Source, root2: Source) -> None:
        # merge two equivalence classes by adding an edge from one root to the other
        if root1 != root2:
            self._parents[root1] = root2

    def _rewrite(self, src: Source) -> sympy.Expr:
        # always represent the given source by the root of its equivalence class
        src = self._find(src)
        if src in self._defs:
            # simply look up the definition if it exists
            # NOTE(avik): This works because definitions are always transitively-closed;
            # otherwise we would have to do recursive rewriting.
            return self._defs[src]
        else:
            # otherwise, create a symbol representing the source
            return sympy.Symbol(src.name)

    def is_equal(self, source1: Source, source2: Source) -> bool:
        return (
            # check whether source1 and source2 have the same root
            # or are relaxed
            (src1 := self._find(source1)) in self.relaxed_sources
            or (src2 := self._find(source2)) in self.relaxed_sources
            or src1 == src2
            # check whether source1 is derived equal to source2
            or self.is_derived(source1, source2, lambda x: x)
        )

    def is_derived(
        self, src: Source, symbol_src: Source, fn: Callable[[sympy.Expr], sympy.Expr]
    ) -> bool:
        # check whether both src and symbol_src have the same definition
        return self._rewrite(src) == fn(self._rewrite(symbol_src))


def _assert_symbol_context(symbolic_context: object) -> TypeGuard[SymbolicContext]:
    if not isinstance(symbolic_context, SymbolicContext):
        raise AssertionError("Invalid symbolic_context object")
    if type(symbolic_context) is SymbolicContext:
        raise AssertionError("Illegal usage of symbolic_context ABC")
    return True


def _is_supported_equivalence(
    expr: sympy.Expr,
) -> TypeGuard[sympy.Add | sympy.Mul | sympy.Symbol]:
    # Currently supported Dim ops are linear expressions with integer coefficients.
    # So check that expr only contains +, *, ints, and a single occurrence of a symbol.
    # (See also documentation of dynamic_shapes._DerivedDim.)
    if isinstance(expr, (sympy.Add, sympy.Mul)):
        if len(expr.args) > 2:
            return False
        lhs, rhs = expr.args
        return (_is_supported_equivalence(lhs) and isinstance(rhs, sympy.Integer)) or (
            isinstance(lhs, sympy.Integer) and _is_supported_equivalence(rhs)
        )
    return isinstance(expr, sympy.Symbol)


def _has_uninterpretable_sympy_function(expr: sympy.Basic) -> bool:
    """
    Add functions that our sympy interpreter can't reify into FX nodes
    """
    return expr.has(
        torch.utils._sympy.functions.ToFloat,
        torch.utils._sympy.functions.TruncToInt,
        torch.utils._sympy.functions.CeilToInt,
    )


@dataclass(frozen=True, slots=True)
class SymbolicContext:
    """
    Data structure specifying how we should create symbols in
    ``create_symbolic_sizes_strides_storage_offset``; e.g., should
    they be static or dynamic.

    This is an abstract base class because we are probably going to add
    another version of this that says "use exactly these SymInts, don't
    allocate fresh symbols."
    """


@dataclass(frozen=True, slots=True)
class SymIntSymbolicContext(SymbolicContext):
    """
    Data structure specifying any constraints on a SymInt input
    """

    constraint: DimConstraint


_P1 = ParamSpec("_P1")
_T1 = TypeVar("_T1")


@dataclass(frozen=True, slots=True)
class StatelessSymbolicContext(SymbolicContext, Generic[_P1, _T1]):
    """
    Create symbols in ``create_symbolic_sizes_strides_storage_offset`` via
    a symbolic_context determination as given by ``DimDynamic`` and ``DimConstraint``.
    This will cause fresh symbols to be allocated
    """

    dynamic_sizes: DimList[DimDynamic]
    dynamic_strides: DimList[DimDynamic] = None  # type: ignore[assignment]
    constraint_sizes: DimList[DimConstraint] = None  # type: ignore[assignment]
    constraint_strides: DimList[DimConstraint] = None  # type: ignore[assignment]
    specialize_on: list[list[Callable[_P1, _T1]]] | None = None
    # If the tensor is a view, this should be populated for the base. It contains
    # information on how to allocate symbols when recursively fakeifying the base
    # during view fake-ification.
    view_base_context: SymbolicContext | None = None
    # Maps dimension index to shape_id.
    shape_ids: dict[int, str | None] | None = None
    # Maps dimension index to (min, max) bounds for unbacked dimensions.
    unbacked_bounds: dict[int, tuple[int | None, int | None]] | None = None
    # TODO: add storage offset and stride symbolic_context

    def __post_init__(self) -> None:
        if self.specialize_on is None:
            object.__setattr__(
                self,
                "specialize_on",
                [[]] * len(self.dynamic_sizes),
            )
        if self.dynamic_strides is None:
            object.__setattr__(
                self,
                "dynamic_strides",
                [DimDynamic.INFER_STRIDE] * len(self.dynamic_sizes),
            )
        if self.constraint_sizes is None:
            object.__setattr__(
                self, "constraint_sizes", [None] * len(self.dynamic_sizes)
            )
        if self.constraint_strides is None:
            object.__setattr__(
                self, "constraint_strides", [None] * len(self.dynamic_sizes)
            )
        if not all(
            stride in (DimDynamic.INFER_STRIDE, DimDynamic.DYNAMIC, DimDynamic.DUCK)
            for stride in self.dynamic_strides
        ):
            raise AssertionError(
                "dynamic_strides must only contain INFER_STRIDE, DYNAMIC, or DUCK"
            )


# note [Tensor Fakification and Symbol Caching]
#
# As of the time of this note, dynamo creates a fresh fake tensor mode for backends.
# The reason we do this is because there are certain classes of operations, namely,
# metadata mutations, that change tensor size, stride, etc. This means that the fake tensor
# state at the end of a dynamo trace is different than the fake tensor state at the beginning
# of a trace. Backends like aot_autograd need a fresh fake tensor to correctly track metadata mutation,
# view relationships, etc.
#
# As we create a new fake mode, we also lose the memoization that comes with it. Rather than
# transfer the memoization cache, we instead transfer the shape env. However, with this
# comes nuance - as dynamo is selective in how it makes symbolic shapes. Due to strategies in
# automatic dynamic and constraints, the policy for which dims are dynamic is nuanced and varies across
# recompilations.
#
# In order to preserve the symbolic decisions made during dynamo tensor fakification, we pass
# a StatefulSymbolicContext at creation time. This object is tracked, per tensor, on the TracingContext.
# The lifecycle of this object should match the lifecycle of the original dynamo tracked tensor, and it is
# safe to reuse this object as many times as necessary to create a fake tensor. Fake tensors
# created with new fake modes should produce the same exact symbols as the original, providing the same shape_env
# is used.
# TODO(voz): Shape env validation
@dataclass(frozen=True, slots=True, kw_only=True)
class StatefulSymbolicContext(StatelessSymbolicContext):
    """
    Create symbols in ``create_symbolic_sizes_strides_storage_offset`` via
    a symbolic_context determination as given by a cache of Source:Symbol. A cache hit
    will reuse a stored symbol, and a cache miss will write to this cache.

    This behaves like StatelessSymbolicContext, except the cache supersedes the
    other values - dynamic_sizes and constraint_sizes will not be read if we cache
    hit.

    It is the cache owner's responsibility to maintain the lifecycle of the cache
    with respect to different shape_envs, clearing, etc.
    """

    tensor_source: Source
    # Why is this keyed on int first?
    # That integer is actually the id of the shape_env. This cache short-circuits symbol
    # creation, and we must store it per shape env. Now, while tracing invariants are a single
    # shape env per tracing context, and every new frame gets a new shape_env. So where would we have
    # multiple shape envs? The answer lies in recording. When we are replaying, replay_shape_env_events
    # is invoked, and creates a new shape_env. Replaying events against this new shape_env will
    # cause it to fail with unknown symbols, as the symbols cached here will skip creation, and never
    # get recorded in backed_var_to_val, etc.
    # TODO(voz): consider a weakref to the shape_env here
    shape_env_to_source_to_symbol_cache: dict[int, dict[str, sympy.Expr]] = field(
        default_factory=dict
    )
    excluded_sizes: tuple[int | None, ...] | None = None


@dataclass(frozen=True, slots=True)
class SubclassSymbolicContext(StatefulSymbolicContext):
    """
    The correct symbolic context for a given inner tensor of a traceable tensor subclass
    may differ from that of the outer symbolic context. This structure allows for this
    flexibility, with inner symbolic contexts mapped via attr -> symbolic context.
    """

    inner_contexts: dict[str, SymbolicContext] = field(default_factory=dict)


@dataclass(slots=True)
class TrackedFake:
    """
    Tracks the sources of all fake tensors we wrap in Dynamo.
    Used by shape guard computation.
    """

    fake: FakeTensor | SymInt | SymFloat
    source: Source
    symbolic_context: SymbolicContext | None

    def __hash__(self) -> int:
        return hash((self.fake, self.source.name))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TrackedFake):
            return self.fake is other.fake and self.source.name == other.source.name
        return False


def is_symbolic(
    val: int | SymInt | float | SymFloat | bool | SymBool,
) -> TypeGuard[SymInt | SymFloat | SymBool]:
    if isinstance(val, (int, float, bool)):
        return False
    return val.node.is_symbolic()



class DynamicDimConstraintPrinter(PythonPrinter):
    """
    Printer for dynamic dim constraints.
    - Instead of symbol s_k it prints its source t.size()[i]
    - Instead of Eq(_, _), Mod(_, _), etc. it prints _ == _, _ % _, etc.

    We use this to suggest code for specifying dynamic dim constraints.
    """

    def __init__(
        self,
        symbol_to_source: dict[sympy.Symbol, list[Source]],
        source_name_to_debug_name: Mapping[str, str],
    ):
        super().__init__()
        self.symbol_to_source = symbol_to_source
        self.source_name_to_debug_name = source_name_to_debug_name

    def _print_Symbol(self, expr: sympy.Symbol) -> str:
        if not isinstance(expr, sympy.Symbol):
            raise AssertionError(f"Expected sympy.Symbol, got {type(expr)}")
        if not self.symbol_to_source.get(expr):
            raise AssertionError(f"Unknown symbol {expr} created by constraints solver")
        return self.symbol_to_source[expr][0].name


class DimConstraints:
    """
    Custom solver for a system of constraints on symbolic dimensions.
    Solutions are "static" values or simplified "dynamic" constraints.
    """

    def __init__(
        self,
        symbol_to_source: dict[sympy.Symbol, list[Source]],
        var_to_val: Mapping[sympy.Symbol, sympy.Integer],
        marked_dynamic: set[sympy.Symbol],
        source_name_to_debug_name: Mapping[str, str],
    ) -> None:
        # We try to solve systems of inequalities with 1 free variable.
        self._univariate_inequalities: dict[sympy.Symbol, set[SympyBoolean]] = (
            defaultdict(set)
        )
        # Among them, we prioritize solving for a free variable that has equalities.
        # NOTE: _symbols_with_equalities is always a subset of _univariate_inequalities.keys()
        # and removing a symbol from the former => removing it from the latter.
        self._symbols_with_equalities: set[sympy.Symbol] = set()
        # A solution of a free variable with equalities becomes a substitution.
        # We use these substitutions to simplify other constraints.
        # NOTE: removing a symbol from _symbols_with_equalities => adding it to _substitutions.
        self._substitutions: dict[sympy.Symbol, sympy.Integer] = {}

        # In general, constraints may have // and % operations.
        # Of course, // can be expressed in terms of / and %.
        # Our inequality solver can handle / but not %. So we need to transform them away.
        # We do so by using the values of variables as hints to evaluate %.
        # For soundness we record additional congruence guards and solve them separately.
        self._var_to_val: Mapping[sympy.Symbol, sympy.Integer] = var_to_val
        self._congruences: defaultdict[sympy.Symbol, set[sympy.Expr]] = defaultdict(set)

        # We do not try to (directly) solve inequalities with > 1 free variables.
        # NOTE: free variables in these inequalities cannot also be in _substitutions.
        self._multivariate_inequalities: set[SympyBoolean] = set()

        # We park external equalities between free variables here.
        self._symbolic_equivalences: list[tuple[Source, sympy.Expr]] = []

        # Solutions come in two forms:
        # - (static) specializations
        # - (dynamic) inequalities / congruences
        self._static_results: set[str] = set()
        self._dynamic_results: set[str] = set()

        # printer for solutions
        self._dcp = DynamicDimConstraintPrinter(
            symbol_to_source, source_name_to_debug_name
        )

        # inconsistencies found on substituting with concrete values / static solutions
        self._inconsistencies: list[str] = []

        # symbols that are marked dynamic
        self._marked_dynamic = marked_dynamic

        # track supported sympy functions and subtract from list of all sympy functions
        self._supported_sympy_functions: set[sympy.Function] = {
            Application,
            Mod,
            PythonMod,
            FloorDiv,
        }
        self._enumerate_sympy_functions()

    def rewrite_with_congruences(self, s: sympy.Symbol, expr: _SympyT) -> _SympyT:
        """
        Eliminate expressions of the form b // d and b % d while adding congruences of the form b % d == k.
        This leaves rational operators (in particular of the form b / d) that our inequality solver can handle.
        We solve the added congruences separately (using our congruence solver, see below).
        """

        def mod_handler(*args: sympy.Expr) -> sympy.Expr:
            # Suppose that we have an expression of the form b % d with free variable s.
            # Using the value of s as a "hint," we can evaluate b % d to a value k.
            # Then we can rewrite b % d to k while adding the guard b % d == k.

            # NOTE(avik): This abstraction is provably sound but, in general, incomplete. It is complete IFF
            # the original expression always evaluates to a constant value (i.e., it does not vary with s).
            # In other words,
            # - solutions of s with the rewritten expression are guaranteed to also be solutions of s with
            #   the original expression;
            # - while it may be possible to find solutions of s with the original expression that are not
            #   solutions with the rewritten expression, in that case the original expression cannot evaluate
            #   to the same value for all solutions of s.
            #
            # Should we be worried about this incompleteness? No, because of the following reasons:
            # 1. It unblocks dramatic simplification that would not be otherwise possible with current tech
            #    (i.e., "don't let perfect be the enemy of the good").
            # 2. We already have a tradition of using hints to add guards in the compiler for making progress.
            # 3. We have not yet seen a counterexample arise in practice! In particular, any congruence guards
            #    we generate (or simplify to) seem to be of the form b % d == k where k is a constant.
            #
            # Here's a theoretical counterexample: 3*s % (s + 1) == s - 2, that is satisfied by all s >= 2.
            # With any hint (say) s = k, we'd rewrite this to: 3*s % (s + 1) == k - 2. But, substituting, we
            # would then get k - 2 == s - 2, and thus s = k as the (only, constant) solution!
            base, divisor = args
            base, divisor = (
                self.rewrite_with_congruences(s, base),
                self.rewrite_with_congruences(s, divisor),
            )
            mod_reduced = base.xreplace(self._var_to_val) % divisor.xreplace(
                self._var_to_val
            )
            congruence = (base - mod_reduced) % divisor
            if congruence != 0:
                self._congruences[s].add(congruence)
            return mod_reduced

        def floor_div_handler(*args: sympy.Expr) -> sympy.Expr:
            # Suppose that we have an expression of the form b // d with free variable s.
            # Using the value of s, we can evaluate b % d to a value k.
            # Then we can rewrite b // d to (b - k) / d, while adding the guard b % d == k.

            # NOTE(avik): This is exactly equivalent to rewriting b // d as (b - (b % d)) / d
            # and eliminating b % d as above.
            base, divisor = args
            base, divisor = (
                self.rewrite_with_congruences(s, base),
                self.rewrite_with_congruences(s, divisor),
            )
            mod_reduced = base.xreplace(self._var_to_val) % divisor.xreplace(
                self._var_to_val
            )
            congruence = (base - mod_reduced) % divisor
            if congruence != 0:
                self._congruences[s].add(congruence)
            # NB: Must not be CleanDiv, it needs to be regular sympy division
            # so inequality solver works.  This is sort of problematic for
            # is_integer tests though haha
            return (base - mod_reduced) / divisor

        # pyrefly: ignore [missing-attribute]
        if expr.has(Mod):
            # pyrefly: ignore [missing-attribute]
            expr = expr.replace(Mod, mod_handler)
        # 7 // -3 is -3, 7 % -3 is -2, and 7 - (-2) / -3 is -3.0 so negative
        # arguments should be OK.
        # pyrefly: ignore [missing-attribute]
        if expr.has(PythonMod):
            # pyrefly: ignore [missing-attribute]
            expr = expr.replace(PythonMod, mod_handler)
        # pyrefly: ignore [missing-attribute]
        if expr.has(FloorDiv):
            # pyrefly: ignore [missing-attribute]
            expr = expr.replace(FloorDiv, floor_div_handler)
        return expr

    def _enumerate_sympy_functions(self) -> None:
        module = torch.utils._sympy.functions
        all_functions = set()
        for attr in dir(module):
            if isinstance(func := getattr(module, attr), sympy.FunctionClass):
                all_functions.add(func)
        self._unsupported_sympy_functions = all_functions.difference(
            self._supported_sympy_functions
        )

    def _has_unsupported_sympy_function(self, expr: sympy.Basic) -> bool:
        """
        Tracks list of sympy.Functions the export solver doesn't know how to handle.
        """
        return expr.has(*self._unsupported_sympy_functions)

    def add(self, expr: SympyBoolean) -> bool:
        """Add an expression to the set of constraints.

        Return whether the expression is a trivial constraint (i.e., an obvious tautology).
        """
        if expr == sympy.true:
            return True
        orig_expr = expr
        orig_reduced = orig_expr.xreplace(self._var_to_val)
        # TODO(avik): https://github.com/pytorch/pytorch/issues/101093
        # It is possible that `expr` will fail the consistency check because of
        # precision errors. Specifically, on substituting its free symbols with
        # their concrete values, we might end up comparing floats. Until we have
        # a fix for this issue, we delay raising such failures. See solve().
        if orig_reduced == sympy.false:
            self._inconsistencies.append(f"{orig_expr} is inconsistent!")
        if isinstance(
            expr, (sympy.Ne, sympy.Or, sympy.And)
        ) or self._has_unsupported_sympy_function(expr):
            # we're not going to do anything useful with these, so drop them
            return False
        free_symbols = expr.free_symbols
        if not free_symbols:
            raise AssertionError(
                f"Did not expect constraint with no free variables: {expr}"
            )
        if len(free_symbols) > 1:
            # multivariate: record and move on
            self._multivariate_inequalities.add(expr)
        else:
            # univariate: can solve these immediately
            s = next(iter(free_symbols))
            # eliminate // and % (see documentation of `rewrite_with_congruences` above)
            old_n_congruences = len(self._congruences[s])
            expr = self.rewrite_with_congruences(s, expr)
            new_n_congruences = len(self._congruences[s])
            if expr == sympy.true:
                return old_n_congruences == new_n_congruences
            reduced = expr.xreplace(self._var_to_val)
            if reduced == sympy.false:
                self._inconsistencies.append(
                    f"{expr}, obtained by rewriting {orig_expr} with congruences, "
                    "is inconsistent!"
                )
            if isinstance(expr, sympy.Eq):
                # special status for symbols that have equalities (see `solve` below)
                self._symbols_with_equalities.add(s)
            self._univariate_inequalities[s].add(expr)
        return False

    def add_equality(self, source: Source, expr: sympy.Expr) -> None:
        """Add an equality constraint"""
        if expr.is_number:
            # specialization, right here
            self._static_results.add(f"{source.name} == {expr}")
        else:
            # these will resolve to either specializations or dynamic equality constraints
            self._symbolic_equivalences.append((source, expr))

    def _reduce_congruences(self) -> dict[sympy.Symbol, set[sympy.Expr]]:
        reduced_congruences: dict[sympy.Symbol, set[sympy.Expr]] = {}
        for s, congruences in self._congruences.items():
            remainder_modulus_pairs = []
            congruences_to_check = set()
            for congruence in congruences:
                base, divisor = congruence.args
                # We are given a congruence of the form base % divisor == 0 with a free variable s. So:
                # - we transform this into an equation of the form base = divisor * tmp;
                # - we solve this equation for s to get a linear solution with free variable tmp.
                tmp = sympy.Symbol("reduce_congruences_tmp", integer=True)
                symbol, solution = sympy.solve_linear(base - divisor * tmp, symbols=[s])
                # See https://docs.sympy.org/latest/modules/solvers/solvers.html#sympy.solvers.solvers.solve_linear
                # for how to interpret the results.
                if s == symbol:
                    # This means the solution is of the form s = modulus*tmp + remainder.
                    modulus, remainder = sympy.polys.polytools.div(solution, tmp)
                    if isinstance(modulus, sympy.Integer) and isinstance(
                        remainder, sympy.Integer
                    ):
                        # Make sure 0 <= remainder <= modulus.
                        remainder = remainder % modulus
                        remainder_modulus_pairs.append((remainder, modulus))
                        continue
                # This means that we did not get a unique solution to the equation.
                # No problem, we will check it.
                congruences_to_check.add(congruence)
            # Finally we solve for a congruence s such that s = r_i mod m_i for each (r_i, m_i).
            # The solution will be a congruence of the form s = r mod m.
            # NOTE(avik): Since the given m_i may not be pairwise coprime, we can't just use CRT.
            if remainder_modulus_pairs:
                remainder, modulus = sympy.ntheory.modular.solve_congruence(
                    *remainder_modulus_pairs
                )
                reduced_congruences[s] = {(s - remainder) % modulus}
                substitution = {
                    s: modulus * sympy.Symbol("tmp", integer=True) + remainder
                }
                reduced_congruences[s].update(
                    congruence
                    for congruence in congruences_to_check
                    if not sympy.checksol(congruence, substitution)
                )
            else:
                reduced_congruences[s] = congruences_to_check

        return reduced_congruences

    def _raise_inconsistencies(self) -> None:
        if self._inconsistencies:
            msg = "\n".join(self._inconsistencies)
            self._inconsistencies.clear()
            raise ValueError(f"The following inconsistencies were found:\n{msg}")

    def solve(self) -> None:
        """Solve the system of constraint equations to find simplified constraints"""
        self._raise_inconsistencies()
        # as long as there are symbols with equalities, solve for them
        # NOTE(avik): this is guaranteed to terminate (#iterations <= #symbols)
        while self._symbols_with_equalities:
            s = self._symbols_with_equalities.pop()
            exprs = self._univariate_inequalities.pop(s)
            solution = sympy.solvers.inequalities.reduce_inequalities(exprs, s)
            if isinstance(solution, sympy.And):
                solution = next(
                    (arg for arg in solution.args if isinstance(arg, sympy.Eq)),
                    solution,
                )
                if not isinstance(solution, sympy.Eq):
                    raise AssertionError(
                        f"Expected an equality constraint for {s}, got {solution}"
                    )
            symbol, val = solution.args
            if symbol != s:
                raise AssertionError(
                    f"Expected a constraint on {s} instead of on {symbol}"
                )
            # because this is univariate, the solution is a specialization
            self._static_results.add(
                f"{self._dcp.symbol_to_source[s][0].name} == {val}"
            )
            # add this as a substitution to simplify other constraints
            self._substitutions[s] = val  # type: ignore[assignment]

            # simplify multivariate inequalities: some of them will now become univariate!
            multivariate_inequalities = self._multivariate_inequalities
            self._multivariate_inequalities = set()
            for expr in multivariate_inequalities:
                self.add(expr.xreplace({s: self._substitutions[s]}))
            self._raise_inconsistencies()

        # solve linear congruences
        # NOTE(avik): We do not need to solve them for symbols that have already been specialized.
        reduced_congruences = self._reduce_congruences()
        for s, congruences in reduced_congruences.items():
            for congruence in congruences:
                # any congruence that cannot be checked becomes a dynamic constraint as well
                if s not in self._substitutions or not sympy.checksol(
                    congruence, {s: self._substitutions[s]}
                ):
                    if self._is_supported_congruence(congruence):
                        base, divisor = congruence.args
                        tmp_name = "_" + str(
                            self._dcp.source_name_to_debug_name.get(
                                self._dcp.symbol_to_source[s][0].name,
                                self._dcp.symbol_to_source[s][0].name,
                            )
                        )
                        tmp = sympy.Symbol(tmp_name, integer=True)
                        from torch._dynamo.source import ConstantSource

                        self._dcp.symbol_to_source[tmp] = [ConstantSource(tmp_name)]
                        r = try_solve(sympy.Eq(base, divisor * tmp), s)
                        if r is None:
                            raise AssertionError(
                                f"Failed to solve {base} = {divisor} * {tmp} for {s}"
                            )
                        self._dynamic_results.add(self._dcp.doprint(sympy.Eq(s, r[1])))

        # remaining symbols have only pure inequalities (no equalities)
        for s, exprs in self._univariate_inequalities.items():
            try:
                solution = sympy.solvers.inequalities.reduce_inequalities(exprs, s)
                # because this is univariate, the solution is a dynamic (range) constraint
                if isinstance(solution, sympy.Or):
                    solution = next(
                        iter(
                            arg
                            for arg in solution.args
                            if arg.xreplace(self._var_to_val)
                        )
                    )
                if isinstance(solution, sympy.And):
                    for arg in solution.args:
                        self._dynamic_results.add(self._dcp.doprint(arg))
                else:
                    self._dynamic_results.add(self._dcp.doprint(solution))
            except (NotImplementedError, AssertionError):
                log.warning("Failed to reduce inequalities", exc_info=True)
                for expr2 in exprs:
                    self._dynamic_results.add(self._dcp.doprint(expr2))

        # simplify symbolic equivalences: some of them will now become specializations!
        symbolic_equivalences = self._symbolic_equivalences
        self._symbolic_equivalences = []
        for source, expr3 in symbolic_equivalences:
            self.add_equality(source, expr3.xreplace(self._substitutions))

        # remaining symbolic equivalences become dynamic equality constraints
        for source, expr3 in self._symbolic_equivalences:
            self._dynamic_results.add(f"{source.name} == {self._dcp.doprint(expr3)}")

    @classmethod
    def _is_supported_congruence(cls, congruence: sympy.Expr) -> bool:
        base, divisor = congruence.args
        # Congruences that can be currently expressed with supported Dim ops are
        # of the form (x + a) % b == 0, where x is a Dim and a and b are constants.
        # This allows us to derive x as b*y - a for some Dim y.
        # (See also documentation of dynamic_shapes._DerivedDim.)
        if isinstance(base, sympy.Add):
            lhs, rhs = base.args
            cond = (
                isinstance(lhs, sympy.Symbol) and isinstance(rhs, sympy.Integer)
            ) or (isinstance(lhs, sympy.Integer) and isinstance(rhs, sympy.Symbol))
        else:
            cond = isinstance(base, sympy.Symbol)
        cond = cond and isinstance(divisor, sympy.Integer)
        return cond

    def forced_specializations(self) -> dict[str, sympy.Expr]:
        """Returns a dictionary of the names of symbols to their specialized value"""

        def debug_name(src: Source) -> str:
            name = src.name
            if self._dcp.source_name_to_debug_name:
                return f"{self._dcp.source_name_to_debug_name[name]} = {name}"
            else:
                return name

        return {
            debug_name(self._dcp.symbol_to_source[s][0]): val
            for s, val in self._substitutions.items()
            if s in self._marked_dynamic
        }

    def _is_derived_dim(
        self, dim: object
    ) -> TypeGuard[torch.export.dynamic_shapes._DerivedDim]:
        return isinstance(dim, torch.export.dynamic_shapes._DerivedDim)

    def _is_dim(self, dim: object) -> TypeGuard[torch.export.dynamic_shapes.Dim]:
        return isinstance(dim, torch.export.dynamic_shapes.Dim) and not isinstance(
            dim, torch.export.dynamic_shapes._DerivedDim
        )

    def _process_derived_dim_roots(
        self,
        results: dict[str, dict[str, Any]],
        name_to_dim: dict[str, Any],
    ) -> None:
        """
        Here we resolve 2 concerns with derived dims suggested fixes: 1) newly introduced roots,
        and 2) root swapping.

        1) Newly introduced roots appear with modulo guards, e.g. Mod(dx, 2) = 0 suggests
        dx is a derived dim equal to 2 * _dx, introducing a new root _dx. Currently the final
        suggested fixes handle this correctly, but we can get intermediate results that look like
        {"dy": {"eq": "dx + 1"}, "dx": {"eq": "2 * _dx + 1, "min": 3, "max": 15}}
        and this routine prettifies this by unifying to a single root, and making each suggestion
        either a derived dim or min/max range, not both.

        2) With suggested fixes for derived dims, roots can be swapped,
        e.g. dx, dx - 1 -> dy + 1, dy. Here we don't want to print out the attached name,
        since this leads to messages like "dx - 1 = Dim("dx - 1", ...)".
        Instead we evaluate the new root value, and remove results for its derivations.

        First we find all the original roots (specified in dynamic_shapes), that are found in the
        values of results (i.e. used for computing suggesting fix values). These original roots
        (suppose `dx`) are either specialized, unchanged, refined, or swapped
        (expressed as a derived dim). If any of the first 3 cases happen, we suggest `dx`'s value
        in results, and remove suggestions for derivations of `dx`, assuming the derived relation
        is valid. If swapped, we find the new root, and use the fix to evaluate `dx`'s new value,
        and then do the same with `dx`'s derivations.

        Assuming the originally specified derived relations are correct is valid, because:
            1) if the relations are plain wrong (e.g. input shape = (6, 4) with spec (dx, dx - 1))
               produce_guards() will catch this and crash before hand.
            2) if the relations are numerically correct but do not match the emitted guard,
               for example:

                    def forward(self, x, y):
                        return x.reshape([-1]) + y  # guard: s0 * 2 = s1
                    inputs = (torch.randn(6, 2), torch.randn(12))
                    dx = Dim("dx", min=2, max=32)
                    dynamic_shapes={"x": (dx, 2), "y": (dx + 6, )}  # this matches values but not op

               then this leads to 2 linear equations, and a) produce_guards() is able to solve for
               the unique solution of dx = 6 and specialize, and b) the export constraint solver will
               raise an issue due to range constraints (a unique solution means not all values in a
               range satisfy a guard) and also force specializations.
        """
        from torch.export.dynamic_shapes import Dim

        def _check_same_range(c: Mapping[str, int], dim: object) -> bool:
            # returns True if c & dim are both min/max ranges with same values
            return (
                self._is_dim(dim)
                and ("min" in c or "max" in c)
                and (
                    (dim.min < 2 and c.get("min", 2) == 2) or dim.min == c.get("min", 2)  # type: ignore[attr-defined]
                )  # let pass if analysis min = 2 and specified min = 0/1
                and dim.max == c.get("max", int_oo)  # type: ignore[attr-defined]
            )

        # 1) newly introduced roots
        # this part we handle adding newly introduced roots
        # these arise from guards like "x.shape[0] % 3 == 0"
        # leading to suggested fixes like "dx = 3*_dx"
        # extract _dx, and find appropriate min/max values
        #
        # before, we have something like:
        # {"dx": {"eq": 3*_dx+1, "min": 4, "max": 10}, "dy": dx+1, "dz": dx+2}
        # we want instead:
        # {"_dx": {"min": 1, "max": 4}, "dx": 3*_dx+1, "dy": 3*_dx+2, "dz": 3*_dx+3}
        introduced_roots: dict[str, str] = {}  # map new root -> old root
        for k, c in list(results.items()):
            if "eq" in c and isinstance(c["eq"], sympy.Expr):  # derived dim
                root = next(iter(c["eq"].free_symbols))
                if str(root) not in name_to_dim:
                    introduced_roots[str(root)] = k
                    # calculate necessary min & max
                    modulus, remainder = sympy.polys.polytools.div(c["eq"], root)
                    c_min = c.get("min", 2)
                    min_ = math.ceil((c_min - remainder) / modulus)
                    c_max = c.get("max", int_oo)
                    max_ = math.floor((c_max - remainder) / modulus)
                    # create result & dim
                    results[str(root)] = {"min": min_, "max": max_}
                    name_to_dim[str(root)] = Dim(str(root), min=min_, max=max_)
                    # remove old root min/max bounds
                    c.pop("min", None)
                    c.pop("max", None)

        # alter derivations that depend on old root, to unify to new root
        # e.g. dx=3*_dx+1, dy=dx+1 -> dy=3*_dx+2
        for old_root in introduced_roots.values():
            for c in results.values():
                if (
                    "eq" in c
                    and isinstance(c["eq"], sympy.Expr)
                    and str(symbol := next(iter(c["eq"].free_symbols))) == old_root
                ):  # derived dim with root = old_root
                    new_root_expr = results[str(old_root)]["eq"]  # dx=3*_dx+1

                    new_expr = c["eq"].subs({symbol: new_root_expr})  # dy=(3*_dx+1)+1
                    c["eq"] = new_expr

        # 2) root swapping
        # collect all the original roots that are used for calculating values of suggested fixes
        # this consists of:
        # 1) {"dx": {"min": ..., "max": ...}} -> dx: refined root dim
        # 2) {"dy": "dx + 1"} -> dx: root for suggested fix
        modified_roots: set[str] = set()
        for k, c in results.items():
            if k not in name_to_dim:  # _dynamo.export() may handle source directly
                continue
            if self._is_dim(name_to_dim[k]) and ("min" in c or "max" in c):  # case 1)
                modified_roots.add(k)
            elif "eq" in c and isinstance(c["eq"], sympy.Expr):  # case 2)
                root = next(iter(c["eq"].free_symbols))
                if root is None:
                    raise AssertionError("root must not be None")
                modified_roots.add(str(root))

        # exclude newly introduced roots, we've already processed these
        modified_roots = modified_roots.difference(introduced_roots)

        # evaluate the new value for each root
        # this is now either 1) unchanged, 2) refined with a new range,
        # or 3) specialized to a concrete value
        modified_root_values: dict[str, dict[str, Any]] = {}
        for mroot in modified_roots:
            swapped_root = True
            if mroot in results:
                c = results[mroot]
                if ("min" in c or "max" in c) or isinstance(  # range
                    c["eq"], int
                ):  # specialized
                    # here, the original root is a root Dim or concrete value in results.
                    # if it is a derived dim, it is swapped, and we handle that below.
                    if not _check_same_range(
                        c, name_to_dim[mroot]
                    ):  # ignore if unchanged
                        modified_root_values[mroot] = c
                    swapped_root = False

            if swapped_root:
                # if the original root has been swapped in results, that means the new root
                # is a range (if it had specialized, the original root would have too).
                # find this new root, and solve for the original root's range.
                for k, c in results.items():
                    if k not in name_to_dim:
                        continue
                    dim = name_to_dim[k]
                    if (
                        dim.__class__.__name__ == "_DerivedDim"
                        and dim.root.__name__ == mroot
                    ):
                        # only look for min/max root, otherwise root would have specialized
                        if "min" in c or "max" in c:
                            expr = sympy.sympify(k)
                            s = next(iter(expr.free_symbols))
                            result = {
                                "min": try_solve(sympy.Eq(expr, c["min"]), s)[1],  # type: ignore[arg-type, index]
                                "max": try_solve(sympy.Eq(expr, c["max"]), s)[1],  # type: ignore[arg-type, index]
                            }
                            if not _check_same_range(
                                result,
                                name_to_dim[mroot],  # type: ignore[index, arg-type]
                            ):  # ignore if unchanged
                                modified_root_values[mroot] = result  # type: ignore[index]
                                break

        # filter out results where the key is a derived dim (e.g. {"dx - 1" : 4})
        # we only want to suggest fixes for the root, to avoid derived names.
        # also, remove anything in modified_roots, since we either add new modified values after this,
        # or have decided they are unchanged.
        for k in list(results.keys()):
            if k not in name_to_dim:
                continue
            if self._is_derived_dim(name_to_dim[k]) or k in modified_roots:
                del results[k]

        # update results with modified root values
        # now results has the following properties:
        # - only contains original roots as keys
        # - each root is now either specialized, refined, or derived from another original root
        results.update(modified_root_values)

    def prettify_results(
        self,
        original_signature: inspect.Signature,
        dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any],
        constraint_violation_error: object,
        forced_specializations: dict[str, str],
    ) -> str:
        """Format a message for constraint violation errors"""
        from torch.export.dynamic_shapes import _get_dim_name_mapping

        if not self._dcp.source_name_to_debug_name:
            # nothing to do
            return ""

        def transform(s: str, inverse: bool = False) -> str:
            for k, v in self._dcp.source_name_to_debug_name.items():
                s = s.replace(k, v) if not inverse else s.replace(v, k)
            return s

        results: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        if dynamic_shapes is None:
            dynamic_shapes = {}

        def flip(op: str) -> str:
            if op == "<=":
                return ">="
            if op == ">=":
                return "<="
            if op == "<":
                return ">"
            if op == ">":
                return "<"
            if op != "==":
                raise AssertionError(f"Expected op to be '==', got {op}")
            return op

        def relation_with_digit(expr: str, op: str, digit: int) -> None:
            if op == "<=":
                results[expr]["max"] = digit
            elif op == "<":
                results[expr]["max"] = digit - 1
            elif op == ">=":
                results[expr]["min"] = digit
            elif op == ">":
                results[expr]["min"] = digit + 1
            else:
                if op != "==":
                    raise AssertionError(f"Expected op to be '==', got {op}")
                results[expr]["eq"] = digit

        # retrieve dynamic shapes
        name_to_dim = _get_dim_name_mapping(dynamic_shapes)

        for s in self._static_results.union(self._dynamic_results):
            t = transform(s)
            if t == s:
                continue
            left, op, right = re.split(r"( == | <= | >= | < | > )", t)
            op = op.strip()
            if op == "==" and left == right:
                continue
            if right.isdigit():
                relation_with_digit(left, op, int(right))
            elif left.isdigit():
                relation_with_digit(right, flip(op), int(left))
            else:
                if op != "==":
                    raise AssertionError(f"Expected op to be '==', got {op} for {t}")
                try:
                    results[left]["eq"] = sympy.sympify(right)
                except TypeError:  # rhs source is not linked to Dim name
                    pass

        # order forced specializations based on name
        forced_specializations = {
            k: forced_specializations[k]
            for k in sorted(
                forced_specializations.keys(),
                key=lambda x: x.split(" = ")[1],
            )
        }

        buf = ""
        if forced_specializations:
            debug_names = set()
            for k in forced_specializations:
                dim = name_to_dim[k.split(" = ")[0]]
                if self._is_derived_dim(dim):
                    debug_names.add(dim.root.__name__)  # type: ignore[attr-defined]
                else:
                    debug_names.add(dim.__name__)

            buf += (
                f"Specializations unexpectedly required ({', '.join(sorted(debug_names))})! "
                'For more information, run with TORCH_LOGS="+dynamic".\n'
            )
            for s, val in forced_specializations.items():
                buf += f"  - solving the guards generated for {s} resulted in a specialized value of {val}.\n"

        self._process_derived_dim_roots(results, name_to_dim)

        dims = []
        others = []

        # order results by source name
        results2 = {
            k: results[k]
            for k in sorted(
                results.keys(),
                key=lambda x: transform(x, inverse=True),
            )
        }
        for k, c in results2.items():
            if "eq" in c:
                other = c["eq"]
                if isinstance(other, int):
                    others.append(f"{k} = {other}")
                elif _is_supported_equivalence(other):
                    others.append(f"{k} = {other}")
            else:
                min_ = c.get("min", None)
                if min_ == 2:
                    min_ = None
                max_ = c.get("max", None)
                if min_ is not None and max_ is not None:
                    dims.append(f"{k} = Dim('{k}', min={min_}, max={max_})")
                elif min_ is not None:
                    dims.append(f"{k} = Dim('{k}', min={min_})")
                elif max_ is not None:
                    dims.append(f"{k} = Dim('{k}', max={max_})")
                else:
                    dims.append(f"{k} = Dim('{k}')")

        # results2 will get filtered out if no new suggestions,
        # this can happen if guards are too complex.
        # in that case don't suggest fix
        if dims or others:
            buf += "\nSuggested fixes:\n  "
            buf += "\n  ".join(dims + others)

        return buf



__all__ = [
    "DimDynamic",
    "Constraint",
    "StrictMinMaxConstraint",
    "RelaxedUnspecConstraint",
    "DimConstraint",
    "EqualityConstraint",
    "_assert_symbol_context",
    "_is_supported_equivalence",
    "_has_uninterpretable_sympy_function",
    "SymbolicContext",
    "SymIntSymbolicContext",
    "_P1",
    "_T1",
    "StatelessSymbolicContext",
    "StatefulSymbolicContext",
    "SubclassSymbolicContext",
    "TrackedFake",
    "is_symbolic",
    "DynamicDimConstraintPrinter",
    "DimConstraints",
]
