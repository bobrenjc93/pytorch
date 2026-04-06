# ruff: noqa: F403, F405
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



from ._symbolic_shapes_constraints import *
from ._symbolic_shapes_utils import *

IndicatorTypes = (IsNonOverlappingAndDenseIndicator,)


def _expandsums(args: list[sympy.Expr]) -> tuple[sympy.Expr, bool]:
    """
    Expand products of sums into sums of products.

    This function takes a list of sympy expressions and separates them into
    additive expressions (those with is_Add=True) and other expressions.
    It then computes the distributive product, expanding (a+b)*(c+d) into a*c + a*d + b*c + b*d.

    Args:
        args: A list of sympy expressions to expand

    Returns:
        A tuple containing:
        - The expanded expression as a sympy.Expr
        - A boolean indicating whether expansion occurred (True if multiple additive
          expressions were present or if there was at least one additive and one other expression)
    """
    adds, other = [], []
    for arg in args:
        if arg.is_Add:
            adds.append(arg)
        else:
            other.append(arg)

    result = [sympy.Mul(*other)]
    for add in adds:
        result = [a * b for a, b in itertools.product(result, add.args)]

    result = sympy.Add(*result)
    return result, len(adds) > 1 or (len(adds) > 0 and len(other) > 0)


def _fast_expand(expr: _SympyT) -> _SympyT:
    """
    A faster implementation of sympy's expand function for common cases.

    This function expands expressions like (a+b)^n or (a+b)*(c+d) into sums of products,
    but avoids the expensive checks and features of sympy's full expand implementation.
    It only recreates objects when necessary to avoid expensive operations.

    Args:
        expr: A sympy expression to expand

    Returns:
        The expanded expression
    """

    # The expand algorithm in sympy is slow due to all the features is supports
    # For eg: e^(-x)*(x-1)/(x+1) is expanded to (x-1)/(e^x + e^x*x) if x is
    # positive and (e^(-x)*x-e^(-x))/(x+1) if x is negative. We do not implement
    # such features here to avoid expensive checks. We also make sure that we
    # only re-create the objects if any of the args changed to avoid expensive
    # checks when re-creating objects.
    new_args = [_fast_expand(arg) for arg in expr.args]  # type: ignore[arg-type]
    # pyrefly: ignore [missing-attribute]
    if any(arg is not new_arg for arg, new_arg in zip(expr.args, new_args)):
        # pyrefly: ignore [missing-attribute]
        return _fast_expand(expr.func(*new_args))

    # pyrefly: ignore [missing-attribute]
    if expr.is_Pow:
        base: sympy.Expr
        exp: sympy.Expr
        base, exp = expr.args  # type: ignore[assignment]
        if exp.is_Integer and base.is_Add:
            if exp > 1:
                return sympy.expand_multinomial(expr, deep=False)
            elif exp < 0:
                return S.One / sympy.expand_multinomial(S.One / expr, deep=False)
    # pyrefly: ignore [missing-attribute]
    elif expr.is_Mul:
        num: list[sympy.Expr] = []
        den: list[sympy.Expr] = []
        # pyrefly: ignore [missing-attribute]
        for arg in expr.args:
            if arg.is_Pow and arg.args[1] == -1:
                den.append(S.One / arg)  # type: ignore[operator, arg-type]
            else:
                num.append(arg)  # type: ignore[arg-type]

        num, num_changed = _expandsums(num)
        den, den_changed = _expandsums(den)
        if num_changed or den_changed:
            return num / den

    return expr


@lru_cache(256)
def safe_expand(r: _SympyT) -> _SympyT:
    """
    Expand the given symbolic expression by recursively rewriting product of
    sums into sum of products (with the product being either a multiplication or
    exponentiation).

    NOTE: using this on an intermediate expression may prevent simplification
    down the line, e.g., if we eagerly expand `(a + b)^2` into `a^2 + 2ab + b^2`,
    we won't be able to simplify `(a^2 + 2ab + b^2) / (a + b)` as easily.
    """
    if hasattr(r, "expand"):
        try:
            return _fast_expand(r)
        except RecursionError:
            log.warning("RecursionError in _fast_expand(%s)", r)
            return r
    else:
        return r


class _SymbolInfo(NamedTuple):
    k: sympy.Symbol
    vr: ValueRanges | None
    val: sympy.Integer | None
    is_size_like: bool


@lru_cache(None)
def _maybe_evaluate_static_worker(
    expr: _SympyT,
    # NB: this is a tuple to ensure it can be LRU cached
    symbol_info: tuple[_SymbolInfo, ...],
    unbacked_only: bool,
    size_oblivious: bool,
) -> _SympyT | None:
    """
    This variant of ShapeEnv._maybe_evaluate_static has no dependence on
    ShapeEnv and thus can be cached indefinitely.  It does the "heavy" lifting
    for static evaluation, including nontrivial reliance on Sympy simplification
    that occurs when we reallocate the symbols
    """

    # Simplify making use of value range lower bound
    new_shape_env = {}
    new_range_env = {}
    for idx, sinfo in enumerate(symbol_info):
        k, vr, val, is_size_like = sinfo
        if isinstance(val, SingletonInt):
            # Skip var_ranges logic for SingletonInt which is only used
            # for jagged layout NestedTensors today
            continue
        if vr is None:
            raise AssertionError(f"vr must not be None for symbol {k}")
        if size_oblivious and is_size_like:
            lower = max(2, vr.lower)
            # Clamping size-oblivious to some quantity below sys.maxsize
            # helps us determine that f(u0) != sys.maxsize, which is a
            # test that is looking for sys.maxsize as a sentinel, but you
            # don't really want to worry about it for unbacked SymInts.
            # This is similar to the flavor where size oblivious omits
            # 0/1, it changes semantics but in a benign way.
            upper = min(2**48, vr.upper)
            # Excluding the very upper bound can be helpful
            if upper > lower:
                upper = upper - 1
            # This is a bit dodgy: what this means is that there was a
            # size-like unbacked symbol whose upper bound < 2.  This
            # causes... problems.
            if lower <= upper:
                vr = ValueRanges(lower, upper)
        else:
            lower = vr.lower
        # Don't do anything if we don't have a nontrivial lower bound
        # Also don't do anything if we asked only to simplify unbacked
        # SymInt
        if lower is -int_oo or (unbacked_only and val is not None) or not vr.is_int:
            new_range_env[k] = vr
            continue
        # The goal is to take our symbols which have various lower bounds
        # and reallocate them into new symbols which are exactly positive;
        # e.g., if we have s0 in [2, inf], we want to turn it into ess0 in
        # [1, inf], where s0 = ess0 + 1.  This gives the most information
        # to sympy for subsequent simplifications.
        #
        # Positive means >= 1
        # Positive - 1 means >= 0
        # Positive + lower - 1 means >= lower
        # The new symbol 's' is "too low", so when we substitute it in
        # we have to increase it by offset (and conversely, the new
        # variables have to have their value range bounds adjusted as
        # well)
        s = sympy.Symbol(f"evaluate_static_shape_{idx}", positive=True, integer=True)

        # Note:
        #   Offset might be a fraction(e.g. aten.split.Tensor), but shapes are always integers.
        #   Sympy might give unexpected results when comparing an integer with a non-integer
        #   Therefore, we cast offset to int here.
        #   For example:
        #       shape_0 = sympy.Symbol("shape_0", positive=True, integer=True)
        #       expr = sympy.Eq(shape_0 - 1/3, 4)
        #       expr.xreplace({}) # False
        offset = int(lower - 1)
        new_shape_env[k] = s + offset
        new_range_env[s] = SymPyValueRangeAnalysis.add(vr, -offset)

    # TODO: remove this try catch (esp for unbacked_only)
    try:
        # pyrefly: ignore [missing-attribute]
        new_expr = expr.xreplace(new_shape_env)
    except RecursionError:
        log.warning("RecursionError in sympy.xreplace(%s, %s)", expr, new_shape_env)
        return None

    # We need to canonicalize, as after expand we may have something like `a + b = a` and
    # sympy will not simplify the a. The two appearances of the a will then make value ranges
    # analysis give lose bounds
    new_expr = canonicalize_bool_expr(safe_expand(new_expr))
    if new_expr.is_number:
        return new_expr

    # Check if the range can solve it statically
    out = bound_sympy(new_expr, new_range_env)
    if out.is_singleton():
        return out.lower

    return new_expr if unbacked_only else None



class RuntimeAssert:
    """
    This is pretty similar to ShapeGuard but it also comes with a message,
    and is exclusively used for things that MUST be true (unlike guards,
    which can evaluate False, in which case you just choose not to use
    a particular specialization)
    """

    expr: SympyBoolean
    msg: str = field(repr=False)
    stack: CapturedTraceback = field(repr=False)


# Used for printing SymExprs in compile_fx
class SymExprPrinter(PythonPrinter):
    def _print_Float(self, expr: sympy.Float) -> str:
        return str(float(expr))


class _ShapeGuardPrinter(abc.ABC):
    """
    Abstract base class for printers that convert symbolic expressions to string representations.

    This class provides common functionality for printing symbolic expressions with
    special handling for symbols that represent tensor shapes, strides, etc.
    Subclasses implement specific formatting for different output languages.

    Args:
        symbol_to_source: Mapping from sympy symbols to their source objects
        source_ref: Function to convert a source to its string representation
        var_to_sources: Mapping from sympy symbols to their source objects (for error reporting)
    """

    def __init__(
        self,
        symbol_to_source: Mapping[sympy.Symbol, list[Source]],
        source_ref: Callable[[Source], str],
        var_to_sources: Mapping[sympy.Symbol, list[Source]],
    ) -> None:
        self.symbol_to_source = symbol_to_source
        self.source_ref = source_ref
        self.var_to_sources = var_to_sources
        super().__init__()

    def _print_Float(self, expr: sympy.Float) -> str:
        """Convert a sympy Float to a Python float string representation."""
        return str(float(expr))

    def _print_Symbol(self, expr: sympy.Symbol) -> str:
        """
        Convert a sympy Symbol to its source representation.

        This method looks up the symbol in symbol_to_source mapping and returns
        the string representation of its first source. If the symbol is not in
        symbol_to_source (which can happen when symbols appear in guard expressions
        through simplification or substitution), it falls back to var_to_sources.

        Args:
            expr: The sympy Symbol to convert

        Returns:
            String representation of the symbol's source

        Raises:
            AssertionError: If the symbol is not found in either mapping
        """
        if not isinstance(expr, sympy.Symbol):
            raise AssertionError(f"Expected sympy.Symbol, got {type(expr)}")

        # Try symbol_to_source first, fall back to var_to_sources if not found
        if source := self.symbol_to_source.get(expr):
            return self.print_source(source[0])
        elif source := self.var_to_sources.get(expr):
            return self.print_source(source[0])
        else:

            def repr_sources(src: Mapping[sympy.Symbol, list[Source]]) -> str:
                return repr(
                    {
                        symbol: [s.name for s in sources]
                        for symbol, sources in src.items()
                    }
                )

            raise RuntimeError(
                f"{expr} not in {repr_sources(self.symbol_to_source)} or "
                f"{repr_sources(self.var_to_sources)}.  This could be due to "
                "the issue described in https://github.com/pytorch/pytorch/pull/90665"
            )

    @abc.abstractmethod
    def print_source(self, source: Source) -> str:
        """
        Convert a source object to its string representation.

        Args:
            source: The source object to convert

        Returns:
            String representation of the source
        """
        ...

    @abc.abstractmethod
    def doprint(self, expr: sympy.Expr) -> str:
        """
        Convert a sympy expression to its string representation.

        Args:
            expr: The sympy expression to convert

        Returns:
            String representation of the expression
        """
        ...


class ShapeGuardPythonPrinter(_ShapeGuardPrinter, PythonPrinter):
    """
    Python printer for shape guards that extends the base ShapeGuardPrinter.

    This class provides functionality to print symbolic expressions as Python code,
    with caching to improve performance when printing the same expressions multiple times.
    It handles printing of sources and expressions according to Python syntax.

    Args:
        *args: Arguments passed to the parent classes.
    """

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._print_cache: dict[sympy.Expr, str] = {}

    def print_source(self, source: Source) -> str:
        """
        Convert a source object to its string representation using the source_ref function.

        Args:
            source: The source object to convert

        Returns:
            String representation of the source
        """
        return self.source_ref(source)

    def doprint(self, expr: sympy.Expr) -> str:
        """
        Convert a sympy expression to its Python string representation with caching.

        This method first checks if the expression is already in the cache.
        If found, it returns the cached result; otherwise, it delegates to
        PythonPrinter's doprint method and caches the result.

        Args:
            expr: The sympy expression to convert

        Returns:
            String representation of the expression in Python syntax
        """
        val = self._print_cache.get(expr, None)
        if val is not None:
            return val
        else:
            res = PythonPrinter.doprint(self, expr)
            self._print_cache[expr] = res
            return res


@deprecated(
    "`torch.fx.experimental.symbolic_shapes.ShapeGuardPrinter` is deprecated, "
    "please use `torch.fx.experimental.symbolic_shapes.ShapeGuardPythonPrinter` instead.",
    category=FutureWarning,
)
class ShapeGuardPrinter(ShapeGuardPythonPrinter):
    pass


class _ShapeGuardCppPrinter(_ShapeGuardPrinter, CppPrinter):
    def __init__(self, *args: Any) -> None:
        self.all_symbols: set[str] = set()
        self.source_to_symbol: dict[Source, sympy.Symbol] = {}
        super().__init__(*args)

    def print_source(self, source: Source) -> str:
        if source in self.source_to_symbol:
            return self.source_to_symbol[source].name

        source_name = source.name
        mangled_name = re.sub("[^0-9a-zA-Z_]+", "_", source_name)
        old_mangled_name = mangled_name
        count = 0
        while mangled_name in self.all_symbols:
            mangled_name = f"{old_mangled_name}_{count}"
            count += 1
        self.source_to_symbol[source] = sympy.Symbol(mangled_name)
        self.all_symbols.add(mangled_name)
        return mangled_name

    def doprint(self, expr: sympy.Expr) -> str:
        return CppPrinter.doprint(self, expr)


# A dataclass for storing shape guards
@dataclass(frozen=True, slots=True)
class _ShapeGuardsHelper:
    exprs: list[str]


# A dataclass for storing C++ expressions and helper variables
@dataclass(frozen=True, slots=True)
class _CppShapeGuardsHelper(_ShapeGuardsHelper):
    source_to_symbol: dict[Source, sympy.Symbol]


class LoggingShapeGuardPrinter(ShapeGuardPythonPrinter):
    def __init__(self, var_to_sources: Mapping[sympy.Symbol, list[Source]]):
        super().__init__(var_to_sources, lambda n: n.name, var_to_sources)



TLS = threading.local()


@dataclass(frozen=True, slots=True)
class ShapeEnvSettings:
    """
    Encapsulates all shape env settings that could potentially affect
    FakeTensor dispatch. Used when creating dispatch cache keys.
    """

    allow_scalar_outputs: bool
    allow_dynamic_output_shape_ops: bool
    assume_static_by_default: bool
    specialize_zero_one: bool
    duck_shape: bool
    prefer_deferred_runtime_asserts_over_guards: bool
    trace_asserts: bool


@dataclass(slots=True)
class ValueRangesSLoc:
    """
    Locations of the guards that triggered lower and upper bound.
    """

    lower: SLoc
    upper: SLoc


@contextmanager
def _suppress_guards(shape_env: ShapeEnv) -> Iterator[None]:
    shape_env._suppress_guards_enter()
    try:
        yield
    finally:
        shape_env._suppress_guards_exit()


@dataclass(slots=True)
class _FrameLocalResult:
    loc: str | None = None
    locals: dict[str, Any] = field(default_factory=dict)
    symbols: dict[str, str] = field(default_factory=dict)


class ShapeEnv:
    # This is a wrapper over the actual __init__ function.
    #
    # Where to add a new constructor parameter to ShapeEnv?
    # =====================================================
    # This __init__ function should be used only for parameters related to event recording.
    # These are parameters that we don't wish to pass down the road to new ShapeEnv instances
    # created from replaying events.
    #
    # If you wish to add a parameter to the constructor of ShapeEnv, unrelated to event
    # recording, do so in the _init function.
    def __init__(
        self,
        *,
        should_record_events: bool | None = None,
        tracked_fakes: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._init(**kwargs)

        # Disable event recording when replaying.
        kwargs["should_record_events"] = False

        from torch.fx.experimental.validator import translation_validation_enabled

        self._translation_validation_enabled = translation_validation_enabled()

        # If not specified, enable event recording if both:
        #   - Translation validation is on
        #   - Translation validation bisection is not disabled
        self.should_record_events = (
            should_record_events
            if should_record_events is not None
            else (
                self._translation_validation_enabled
                and not config.translation_validation_no_bisect
            )
        )

        # Enable event recording check if both:
        #   - It should record events
        #   - The recording check is enabled
        self.check_recorded_events = (
            self.should_record_events and config.check_shape_env_recorded_events
        )

        # This will make sure we only record the top-level function call.
        self.is_recording = False
        # Keep track of the list of tracked fakes.
        self.tracked_fakes = tracked_fakes
        # List of events for reconstructing ShapeEnv at arbitrary points in time.
        self.events: list[ShapeEnvEvent] = (
            [ShapeEnvEvent(ShapeEnv, kwargs=kwargs)]
            if self.should_record_events
            else []
        )

        # FakeTensor per-ShapeEnv operation cache. This is used for caching
        # operations that contain symbolic shapes which have guards on the
        # ShapeEnv (so are ShapeEnv-dependent).
        #
        # NOTE: It's important that SymNodes in this cache have their ShapeEnv
        # stripped otherwise you end up with cycles which can only be cleaned
        # with the GC.
        self.fake_tensor_cache: dict[
            torch._subclasses.fake_tensor._DispatchCacheKey,
            torch._subclasses.fake_tensor._DispatchCacheEntry,
        ] = {}

    # Pro-tip: if you add new field to ShapeEnv, this affects some accept
    # tests.  Accept their output with:
    #
    #   EXPECTTEST_ACCEPT=1 python test/dynamo/test_dynamic_shapes.py -k test_shape_env_equal
    #
    def _init(
        self,
        *,
        allow_scalar_outputs: bool = True,
        allow_dynamic_output_shape_ops: bool = True,
        # NB: These are legacy configuration that help us make good choices
        # when the constraint/dynamic dims are not explicitly passed to us.
        # Ideally we will fix all call sites to be explicit and not have
        # implicit choices, but this apparently was pretty involved.
        assume_static_by_default: bool = False,
        # Note - On 0/1 specialization
        #
        # The following options affect decisions we make about eager
        # specialization.  Disabling them will increase trace time (as we do
        # more symbolic reasoning) and can also harm the quality of generated
        # code (because inductor may not be able to specialize for bounds
        # being equal--although if we later respecialize because of a guard,
        # your code may be just as good as it was before.)
        #
        # When True, eagerly specialize input sizes which have 0/1.
        specialize_zero_one: bool = True,
        # When True, assume input sizes which have the same size are
        # symbolically equal.
        duck_shape: bool | None = None,
        # For debugging
        co_fields: dict[str, str] | None = None,
        # When True, whenever safe, we will generate a deferred runtime assert
        # instead of a guard whenever we know that an expression must be True,
        # otherwise it would be an error, even for backed SymInts (where we
        # could ostensibly unconditionally generate guards).  This is useful
        # for export, where preventing "error checking" sizes from showing up
        # in guards is helpful, since these guards in some sense are overly
        # pedantic.  See also https://github.com/pytorch/pytorch/issues/121749
        prefer_deferred_runtime_asserts_over_guards: bool = False,
        # XXX Add any new settings that could affect FakeTensor evaluation
        # to: torch._subclasses.fake_tensor._ShapeEnvSettings
        trace_asserts: bool = False,
    ) -> None:
        if duck_shape is None:
            duck_shape = config.use_duck_shape

        self.settings = ShapeEnvSettings(
            # Not directly used by ShapeEnv; indirectly used by FakeTensor
            allow_scalar_outputs=allow_scalar_outputs,
            allow_dynamic_output_shape_ops=allow_dynamic_output_shape_ops,
            # End
            assume_static_by_default=assume_static_by_default,
            specialize_zero_one=specialize_zero_one,
            duck_shape=duck_shape,
            prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
            trace_asserts=trace_asserts,
        )

        self.guards: list[ShapeGuard] = []
        self.axioms: dict[sympy.Expr, sympy.Expr] = {}

        # A set of ids that have already been allocated. This is used
        # for when we allocate symbol ids using the hash of the source
        # names to ensure we don't have collisions via linear probing
        self.unique_ids: set[int] = set()
        # Maps symbolic ints to their original concrete values
        # Currently populated from tensors
        # When hint is overridden in mark_dynamic, the value stored here
        # is the overridden hint (this is the source of truth for backed
        # hints). The override is also recorded in var_to_hint_override
        # so it can be included in the FxGraphCache key.
        self.backed_var_to_val: dict[sympy.Symbol, sympy.Integer] = {}
        # Only set when propagate_real_tensors is on.
        # Used as last resort to avoid GuardOnDataDependent error in draft export.
        self.real_tensor_prop_unbacked_vals: dict[sympy.Symbol, sympy.Integer] = {}
        # Maps symbolic ints to their min/max range.  These ranges
        # are conservative: the int MUST fall in the range, but the
        # range may contain ints which may not actually appear in
        # practice
        self.var_to_range: dict[sympy.Symbol, ValueRanges] = {}
        self.var_to_range_sloc: dict[sympy.Symbol, ValueRangesSLoc] = {}
        self.source_name_to_debug_name: dict[str, str] = {}
        self.var_to_sources: dict[sympy.Symbol, list[Source]] = {}
        # A set of unbacked symbols that are inputs (i.e: not data dependent).
        self.unbacked_inputs: OrderedSet[sympy.Symbol] = OrderedSet()
        self.var_to_stack: dict[sympy.Symbol, CapturedTraceback] = {}
        # User-provided hint overrides from mark_dynamic/mark_unbacked.
        # Even though we never read hints for backed variables from this
        # dict (backed hints are read from backed_var_to_val), we still
        # want them to always be stored here, since this dict is used as
        # part of the FxGraphCache key.
        self.var_to_hint_override: dict[sympy.Symbol, int] = {}
        # Maps a source to the *original* symbol that was assigned to it
        self.source_to_var: dict[str, sympy.Symbol] = {}
        # Maps from sympy ints to expressions representing them
        # Populated from equality guards (i.e. a.shape[0] == b.shape[0])
        self.replacements: dict[sympy.Symbol, sympy.Expr] = {}
        # The sloc of the guard that triggered this replacement to be added
        self.replacements_slocs: dict[sympy.Symbol, SLoc] = {}
        self.unbacked_renamings: dict[sympy.Symbol, sympy.Symbol] = {}
        # Set holds a % b expressions that evaluate to 0.
        self.divisible: set[sympy.Expr] = set()
        # Exclusion constraints from automatic_dynamic transitions.
        # Each (symbol, excluded_value) pair represents one dim/scalar that
        # transitioned static → dynamic. All pairs are combined into a single
        # Or(Ne(...), ...) guard in produce_guards_verbose.
        self.exclusion_constraints: list[tuple[sympy.Symbol, int]] = []
        # Set that holds "size-like" symbols.  When we perform
        # "size-oblivious" tests, these can be assumed to be >= 2.
        self.size_like: set[sympy.Symbol] = set()
        # Duck-shaping says that if two input tensors have the same size,
        # they get assigned the same symbolic variable
        self.val_to_var: dict[int, sympy.Symbol] = {}
        self.unbacked_symfloat_counter = 0
        self.unbacked_symint_counter = 0
        # Similar to guards, but these MUST evaluate to true and can
        # only be evaluated at runtime midway through (i.e., they always
        # involve unbacked symints)
        #
        # For efficiency reasons, we index in the following way.  Suppose you have
        # a runtime assert i0 + i1 <= s1.  We pick the most recently allocated
        # symbol in the source expression and add the assert to the list for
        # that symbol e.g., {i1: [i0 + i1 <= s1]}.
        #
        # We access the runtime asserts in two situations:
        #
        #   - When we are guarding on an expression, we will attempt to
        #     statically evaluate it, in case the unbacked SymInts can
        #     simplify away.  If we have a runtime assert, we may be able
        #     to discharge the guard entirely.  We only need to attempt
        #     runtime asserts that mention freevars of the expression in
        #     question.
        #
        #   - When we are performing codegen (in Inductor for eager, or
        #     when finalizing the export FX graph), we need to know what
        #     extra runtime asserts to insert.  Whenever an unbacked
        #     SymInt comes into scope, all runtime asserts involving it
        #     become eligible for insertion (so long as all of their other
        #     free unbacked symbols are also in scope).  We technically
        #     can handle any choice of key by kicking inexpressible asserts
        #     to the next unbacked symbol to wait on, but if we choose the
        #     latest key, an assert will only show up at the moment when
        #     we can actually codegen it.
        self.deferred_runtime_asserts: dict[
            sympy.Symbol | None, list[RuntimeAssert]
        ] = {}
        # This exists so we can efficiently invalidate the cache (it's used as
        # part of the cache key); otherwise we'd have to iterate through
        # deferred_runtime_asserts to compute its length
        self.num_deferred_runtime_asserts = 0
        self.log = log
        self.log.info("create_env")
        self.frozen = False
        self._error_on_new_guards = False
        self.runtime_asserts_frozen = False
        self.dim_constraints: DimConstraints | None = None
        self.counter: Counter[str] = collections.Counter()
        # Mapping from sympy.Symbol to the number of guards which mention this
        # symbol
        self.symbol_guard_counter: Counter[sympy.Symbol] = collections.Counter()
        # A selection of important fields on co_field; solely used for
        # signpost_event
        self.co_fields = co_fields if co_fields else {}

        # Whenever we allocate a fresh unbacked Symbol, we add it to this
        # pending list.  Unbacked symbol allocation can occur at unpredictable
        # points during meta tensor propagation, but at some point, we
        # have to know what the binding site for an unbacked symbol is, and
        # this is computed when we actually place the node in the graph. The
        # important thing is that we always actually handle every unaccounted
        # for unbacked symbol, so this list helps us keep track of them and
        # then make sure they are all accounted for.
        #
        # We could potentially give rise to errors earlier by lexically
        # scoping when we do propagation, and only allowing unbacked symbols
        # to be allocated at this point in time.  However this is inconvenient
        # to do in Dynamo, because fake tensor propagation is far from when we
        # analyze binding sites (set_example_value), so we do it in a more
        # mutatey way.
        #
        # NB: fresh unbacked symbols NEVER get substitutions applied to them,
        # they are binding sites!
        self.pending_fresh_unbacked_symbols: list[sympy.Symbol] = []
        # These are symbols which we'd like to process as pending, but if
        # they're missing then it's okay too.
        self.ignorable_fresh_unbacked_symbols: list[sympy.Symbol] = []

        # Version counter used to invalidate cached values
        self._prev_cache_key = self._get_key()
        self._version_counter = 0
        # Separate counter tracking only replacement changes, used by
        # SymNode.expr
        self._replacements_version_counter = 0

        # Each time divisible is changed this should be set to True, this is set in _update_version_counter.
        self._resimplify_floor_div_axioms = True

        # Cache for FX nodes.
        # Maps an already built node a tuple of:
        #   1. node's target
        #   2. list of arguments
        # This drastically reduces the size of the FX graph, avoiding
        # duplicated nodes.
        self.fx_node_cache: dict[tuple[Callable, tuple[Any, ...]], torch.fx.Node] = {}
        self.source_to_symbol: dict[str, sympy.Symbol] = {}

        # Suppose you want to replace an unbacked symbol with another
        # unbacked symbol.  This is error prone because you can cause
        # references to unbacked symbols to time travel backwards.  E.g.,
        #
        # u1 = x.item()
        # ... use of u1 ...
        # u2 = y.item()
        # u3 = z.item()
        # torch._check(u1 == u2 + u3)
        #
        # If you replace u1 with u2 + u3, then the use of u1 now
        # references u2 and u3 prior to them actually being bound at
        # runtime.
        #
        # To control for this, we track the order unbacked symbols
        # were allocated, and only allow substitutions if they respect
        # the dependency from this order; an unbacked symbol can only
        # be substituted with unbacked symbols that come before it in the
        # order.
        #
        # This also imposes an ordering on the unbacked symbol binding
        # sites themselves: you are not allowed to reorder unbacked symbol
        # bindings.  At the moment, this is not tracked, but we potentially
        # could track this at the IR level using a higher order operator
        # with something like effect token tracking.
        self.unbacked_alloc_order: dict[sympy.Symbol, int] = {}

        self.specialization_stacks: dict[Source, traceback.StackSummary] = {}

        # Used by _get_unbacked_replacements / _sub_unbacked_exprs for
        # optimization_hint canonicalization of unbacked expressions.
        self._equality_graph: dict[sympy.Expr, OrderedSet[sympy.Expr]] | None = None
        self._unbacked_replacements: dict[sympy.Expr, sympy.Expr] | None = None

        self.trace_asserts = trace_asserts

        self.specializations: OrderedSet[Specialization] = OrderedSet()

        from torch.fx.experimental.validator import translation_validation_enabled

        self._translation_validation_enabled = translation_validation_enabled()

        if self._translation_validation_enabled:
            from torch.fx.experimental.validator import TranslationValidator

            self.validator = TranslationValidator()
            self.graph = torch.fx.Graph()
            # Create an output graph and start inserting before that.
            # This is needed when 'deepcopy'-ing this object.
            self.graph.inserting_before(self.graph.output(None))

            # Mapping of each node name to the node itself.
            #
            # This is useful for matching an FX node from a recorded ShapeEnv.graph
            # to the FX node of the ShapeEnv we are running the event on.
            #
            # Whenever you add a node to self.graph, you must add a mapping to this
            # variable. Otherwise, the built FX graph on the replayed ShapeEnv will
            # not be valid.
            self.name_to_node: dict[str, torch.fx.Node] = {}

        # Maps shape_id to the first unbacked symbol allocated for that id.
        # When mark_unbacked is called with a shape_id, we allocate fresh
        # symbols but add runtime equality checks via torch._check to ensure
        # all dims with the same shape_id are treated as the same symbol.
        self._shape_id_to_unbacked_symbol: dict[str, sympy.Expr] = {}

    @property
    def allow_scalar_outputs(self) -> bool:
        return self.settings.allow_scalar_outputs

    @property
    def allow_dynamic_output_shape_ops(self) -> bool:
        return self.settings.allow_dynamic_output_shape_ops

    @property
    def assume_static_by_default(self) -> bool:
        return self.settings.assume_static_by_default

    @property
    def specialize_zero_one(self) -> bool:
        return self.settings.specialize_zero_one

    @property
    def duck_shape(self) -> bool:
        return self.settings.duck_shape

    @property
    def prefer_deferred_runtime_asserts_over_guards(self) -> bool:
        return self.settings.prefer_deferred_runtime_asserts_over_guards

    @contextmanager
    def patch_source_specialization(
        self, source: Source, check_fn: Callable[[sympy.Symbol], sympy.Expr]
    ) -> Iterator[None]:
        """
        Temporarily add symbol-level axioms to the ShapeEnv. This is useful when you want to "fork"
        and have parallel universes of ShapeEnvs. For example, we use this when doing multi-graph
        compile so we can support various graphs with varying levels of specializations.

        This context manager allows for temporarily adding constraints to the shape environment
        based on a specialization function applied to a symbol associated with a source.

        Args:
            source: The source of the symbol to specialize
            check_fn: A function that takes a sympy Symbol and returns a sympy expression
                     representing a constraint/specialization to be applied
        """
        name = source.name
        sym = self.source_to_var[name]
        expr = check_fn(SymInt(SymNode(sym, self, int, None))).node._expr
        new_axioms = dict(self.get_implications(self.simplify(expr)))
        added_replacements = {}

        for axiom in new_axioms:
            if (
                isinstance(axiom, sympy.Eq)
                and isinstance(axiom.lhs, sympy.Symbol)
                and isinstance(axiom.rhs, sympy.Integer)
                and axiom.lhs not in self.replacements
            ):
                self.replacements[axiom.lhs] = axiom.rhs
                added_replacements[axiom.lhs] = axiom.rhs
        self.axioms.update(new_axioms)

        # We need to freeze the ShapeEnv because any additional modification of
        # the ShapeEnv will cause unsoundness for subsequent specialization calls.
        self.frozen = True
        try:
            yield
        finally:
            for k in new_axioms:
                self.axioms.pop(k, None)
            for k in added_replacements:
                self.replacements.pop(k, None)
            self.frozen = False

    def check_equal(self, other: ShapeEnv) -> None:
        """Compare another ShapeEnv for equivalence"""
        # ShapeEnv fields that are not relevant for the outcome of
        # ShapeEnv.produce_guards call:
        #   - Debugging variables
        #   - Translation validation related variables
        #   - Events recording related variables
        non_state_variable_names = (
            "counter",
            "log",
            "var_to_stack",
            "fx_node_cache",
            "graph",
            "validator",
            "check_recorded_events",
            "should_record_events",
            "is_recording",
            "tracked_fakes",
            "events",
            "source_name_to_debug_name",
            "_prev_cache_key",
            "_version_counter",
            "dim_constraints",
            # source locations are OK to diverge
            "var_to_range_sloc",
            "replacements_slocs",
            "_replacements_version_counter",
            "_resimplify_floor_div_axioms",
            "_expr_sym_node_id",
            "specialization_stacks",
            # Cached state for optimization_hint unbacked canonicalization
            "_equality_graph",
            "_unbacked_replacements",
        )

        # Mapping of the value of each to-be-compared field into the values that
        # should actually be compared.
        #
        # You should modify this if, for example, the field that holds state and
        # debugging information. e.g. ShapeGuard holds the actual guard (sympy.Expr)
        # and the stack when it was added to the set of guards. In order to compare
        # it, we throw away the stack information.
        def map_value(key: str, value: Any) -> Any:
            if key == "guards":
                # Transform the list of ShapeGuard into a list of expressions.
                return [g.expr for g in value]
            elif key == "deferred_runtime_asserts":
                # Transform the list of RuntimeAsserts into a list of expressions.
                return {s: [ra.expr for ra in ras] for s, ras in value.items()}
            elif key == "name_to_node":
                # Compare just the set of keys is the same.
                return set(value.keys())
            elif key in (
                "symbol_guard_counter",
                "pending_fresh_unbacked_symbols",
                "fake_tensor_cache",
            ):
                # Skip this for comparisons
                return None
            return value

        shape_env_check_state_equal(self, other, non_state_variable_names, map_value)

    def _snapshot_tracked_fakes(self) -> list[Any] | None:
        if self.tracked_fakes is None:
            return None

        from torch._dynamo.variables.builder import TrackedFake

        def maybe_transform_fake(fake: TrackedFake) -> TrackedFake:
            inner_fake = (
                fake.fake
                if isinstance(fake.fake, (torch.SymInt, torch.SymFloat))
                else FakeTensorMeta.from_fake(fake.fake)
            )
            # Even though TrackedFake accepts either a Union[SymInt, FakeTensor], here we give it a
            # FakeTensorMeta for two reasons:
            #   1. this is all the information we need when recording ShapeEnvEvents.
            #   2. it works even if each TrackedFake changes its metadata.
            return TrackedFake(inner_fake, fake.source, fake.symbolic_context)  # type: ignore[arg-type]

        return [maybe_transform_fake(fake) for fake in self.tracked_fakes]

    def _last_event_index(self) -> int:
        return len(self.events) - 1

    @contextmanager
    def _recording(self) -> Iterator[None]:
        self.is_recording = True
        try:
            yield
        finally:
            self.is_recording = False

    @record_shapeenv_event()
    def _eliminate_unbacked(self, orig_s: sympy.Symbol, new_s: sympy.Expr) -> None:
        self._set_replacement(orig_s, new_s, "eliminate_unbacked")

    @record_shapeenv_event()
    def set_real_tensor_prop_unbacked_vals(self, k: sympy.Symbol, v: int) -> None:
        """Used only when propagate_real_tensors; registers a value for an
        unbacked symbol, which can be used last resort to resolve hints."""
        log.info("set_real_tensor_prop_unbacked_vals %s = %s", k, v)
        self.real_tensor_prop_unbacked_vals[k] = sympy.sympify(v)

    # Unlike set_replacement, this records a shapeenv event
    @record_shapeenv_event()
    def _rename_unbacked_to(self, orig_s: sympy.Symbol, new_s: sympy.Symbol) -> None:
        if not isinstance(orig_s, sympy.Symbol):
            raise AssertionError(f"Expected sympy.Symbol, got {orig_s}")
        if not isinstance(new_s, sympy.Symbol):
            raise AssertionError(f"Expected sympy.Symbol, got {new_s}")
        if not free_unbacked_symbols(new_s):
            raise AssertionError(
                f"Expected new_s to have free unbacked symbols: {new_s}"
            )
        if not free_unbacked_symbols(orig_s):
            raise AssertionError(
                f"Expected orig_s to have free unbacked symbols: {orig_s}"
            )
        dest = self.replacements.get(orig_s)
        if dest is not None:
            if free_unbacked_symbols(dest):
                raise AssertionError(f"{orig_s} -> {dest}")
        self._set_replacement(orig_s, new_s, "rename_unbacked_to")
        self.unbacked_renamings[orig_s] = new_s
        if dest is not None:
            self._set_replacement(new_s, dest, "rename_unbacked_to_dest")

    @record_shapeenv_event()
    def _constrain_is_bounded(self, a: sympy.Symbol, upper_bound: int) -> None:
        # TODO: Do something nontrivial when upper_bound is expression
        pass

    @record_shapeenv_event()
    def _constrain_range_for_size(
        self, a: sympy.Symbol, min: int | None = None, max: int | None = None
    ) -> None:
        if min is None:
            min = 0
        if max is None:
            max = int_oo

        if max < min:
            raise ValueError(
                "Maximum value to constrain_as_size can't be less than the specified min value, "
                f"received min={min} and max={max}"
            )

        self.constrain_symbol_range(
            a,
            compiler_min=min,
            compiler_max=max,
        )
        self.size_like.add(a)

    @record_shapeenv_event()
    def _constrain_range(self, a: sympy.Expr, min: int, max: int) -> None:
        if isinstance(a, sympy.Integer):
            if not (min <= int(a) <= max):
                raise ValueRangeError(f"Invalid value {int(a)} for range [{min}:{max}]")
            return

        # TODO: Shouldn't we install a guard if the symbol is backed?  Or is the
        # semantics that this is an "unchecked" assert (but it this actually
        # something useful?  Might be better to restrict only for unbacked
        # SymInt).
        if isinstance(a, sympy.Symbol):
            self.constrain_symbol_range(
                a,
                compiler_min=min,
                compiler_max=max,
            )

    @record_shapeenv_event()
    def _constrain_unify(self, a: SymInt, b: SymInt) -> None:
        """
        Given two SymInts, constrain them so that they must be equal.  NB:
        this will not work with SymInts that represent nontrivial expressions
        (yet!)
        """
        # TODO: this does not install a deferred runtime assert yet

        # TODO: Maybe dedupe this with _maybe_guard_rel?
        # Update Feb 2024: this is extra important to do, this doesn't handle
        # unbacked replacements properly nor does it generate deferred runtime
        # asserts
        if not isinstance(a, SymInt):
            if not isinstance(b, SymInt):
                if a != b:
                    raise AssertionError(f"Expected {a} == {b}")
            else:
                if not isinstance(b.node.expr, sympy.Symbol):
                    raise AssertionError("constraining non-Symbols NYI")
                if b.node.shape_env is not self:
                    raise AssertionError("b.node.shape_env must be self")
                self.replacements[b.node.expr] = sympy.Integer(a)
        else:
            # TODO: Actually, we can support this as long as one of them is a symbol.
            # NB: We can't actually do "unification" as our operators are not
            # injective
            if not isinstance(a.node.expr, sympy.Symbol):
                raise AssertionError("constraining non-Symbols NYI")
            if a.node.shape_env is not self:
                raise AssertionError("a.node.shape_env must be self")
            if not isinstance(b, SymInt):
                self.replacements[a.node.expr] = sympy.Integer(b)
            else:
                if a.node.shape_env is not b.node.shape_env:
                    raise AssertionError("a.node.shape_env must be b.node.shape_env")
                if not isinstance(b.node.expr, sympy.Symbol):
                    raise AssertionError("constraining non-Symbols NYI")
                new_var = self._find(a.node.expr)
                self.replacements[b.node.expr] = new_var

    def _ignore_fresh_unbacked_symbols_tls(self) -> bool:
        return getattr(TLS, "ignore_fresh_unbacked_symbols", False)

    @record_shapeenv_event()
    def _ignore_fresh_unbacked_symbols_set(self, b: bool) -> bool:
        prev = self._ignore_fresh_unbacked_symbols_tls()
        TLS.ignore_fresh_unbacked_symbols = b
        return prev

    @contextmanager
    def ignore_fresh_unbacked_symbols(self) -> Iterator[None]:
        """
        Indicates that the newly allocated unbacked SymInts are being
        discarded
        """
        prev = self._ignore_fresh_unbacked_symbols_set(True)
        try:
            yield
        finally:
            self._ignore_fresh_unbacked_symbols_set(prev)

    @record_shapeenv_event()
    def freeze(self) -> None:
        """Freeze this ShapeEnv to stop accumulating guards

        A frozen ShapeEnv will ignore any further guards generated on it and
        only emit a warning which may lead to accuracy problems.
        """
        self.frozen = True

    @record_shapeenv_event()
    def freeze_runtime_asserts(self) -> None:
        """Freeze this ShapeEnv to stop adding deferred runtime asserts.

        We will error if you try to install a new runtime assert when it is
        frozen.  This would indicate a lowering violation, or perhaps something
        we know statically is already True but we are checking it again in a way
        that is not clearly dischargeable.
        """
        # self.prefer_deferred_runtime_asserts_over_guards = False
        self.runtime_asserts_frozen = True

    def _create_symbol_for_source(self, source: Source) -> sympy.Symbol | None:
        if not self._translation_validation_enabled:
            return None
        srcname = source.name
        if source not in self.source_to_symbol:
            self.source_to_symbol[srcname] = sympy.Symbol(srcname, integer=True)
        return self.source_to_symbol[srcname]

    def _add_z3var(self, symbol: sympy.Symbol, type: type) -> None:
        if self._translation_validation_enabled:
            self.validator.add_var(symbol, type)

    def _add_target_expr(self, expr: SympyBoolean) -> None:
        if self._translation_validation_enabled:
            self.validator.add_target_expr(expr)

    def _add_assertion(self, expr: SympyBoolean) -> None:
        if self._translation_validation_enabled:
            self.validator.add_assertion(expr)

    def _check_translation_validate(self) -> None:
        if self._translation_validation_enabled:
            self.validator.validate()

    @record_shapeenv_event()
    def _create_fx_call_function(
        self,
        op: Callable,
        args: tuple,
    ) -> tuple[torch.fx.Node | None, bool]:
        # Cache this tuple in order to avoid duplicated nodes.
        node_key = (op, args)
        # Flags whether the returned node was cached or not.
        fresh = False

        if self._translation_validation_enabled and node_key not in self.fx_node_cache:
            # Presence of None in the arguments implies that we should ignore this operation.
            if any(a is None for a in args):
                # We check if we are not mixing SymNode that should not be ignored
                # (fx_node is not None) with those that should (fx_node is None).
                if not all(not isinstance(a, torch.fx.Node) for a in args):
                    raise AssertionError(
                        "Cannot mix SymNodes with fx_node and without fx_node"
                    )
                return None, fresh

            fresh = True

            # If translation validation is enabled, all arguments must have its
            # own FX node.
            if not all(a is not None for a in args):
                raise AssertionError(f"missing arg in FX graph ({op.__name__}): {args}")
            node = self.fx_node_cache[node_key] = self.graph.call_function(op, args)
            self.name_to_node[node.name] = node

        return self.fx_node_cache.get(node_key, None), fresh

    def _create_fx_placeholder_and_z3var(
        self,
        symbol: sympy.Symbol,
        type: type,
    ) -> torch.fx.Node | None:
        if not self._translation_validation_enabled:
            return None

        node_key = (self.graph.placeholder, (symbol,))

        # Check if we haven't added this symbol already.
        # If so, skip the placeholder creation, as it
        # generates invalid Python code.
        if node_key not in self.fx_node_cache:
            # Add a Z3 variable according to 'type'.
            self._add_z3var(symbol, type)
            # Create the FX placeholder out of a mangled name.
            mangled_name = re.sub(
                r"[^a-zA-Z0-9]", "_", re.sub(r"[()]", "", symbol.name)
            )
            node = self.fx_node_cache[node_key] = self.graph.placeholder(mangled_name)
            self.name_to_node[node.name] = node
            # Attach the 'symbol' to the placeholder so that we can retrieve
            # the Z3 variable later.
            node.meta["symbol"] = symbol

        return self.fx_node_cache[node_key]

    def _remove_fx_node(self, node: torch.fx.Node | None) -> None:
        if self._translation_validation_enabled and node is not None:
            self.name_to_node.pop(node.name)
            self.graph.erase_node(node)

    def _add_fx_node_metadata(self, node: torch.fx.Node) -> None:
        from torch._dynamo.utils import get_current_node

        if self.should_record_events:
            node.meta[SHAPEENV_EVENT_KEY] = self._last_event_index()
            node.meta[CURRENT_NODE_KEY] = get_current_node()

    @staticmethod
    def _suppress_guards_tls() -> bool:
        return getattr(TLS, "suppress_guards", False)

    @record_shapeenv_event()
    def _suppress_guards_enter(self) -> None:
        if not hasattr(TLS, "suppress_guards_stack"):
            TLS.suppress_guards_stack = []
        old = self._suppress_guards_tls()
        TLS.suppress_guards_stack.append(old)
        TLS.suppress_guards = True

    @record_shapeenv_event()
    def _suppress_guards_exit(self) -> None:
        old = (
            TLS.suppress_guards_stack.pop()
            if len(TLS.suppress_guards_stack) > 0
            else False
        )
        TLS.suppress_guards = old

    def suppress_guards(self) -> _GeneratorContextManager[None]:
        """Context manager to ignore all guards generated inside."""
        return _suppress_guards(self)

    @contextmanager
    def error_on_new_guards(self) -> Iterator[None]:
        """Context manager that raises _ShapeEnvGuardError if a guard is attempted.

        Temporarily freezes the ShapeEnv and makes _check_frozen raise
        instead of warn, so that guard-installing code paths produce an
        exception that is not cached by the _inner_evaluate_expr LRU cache.
        """
        old_frozen = self.frozen
        old_error = self._error_on_new_guards
        self.frozen = True
        self._error_on_new_guards = True
        try:
            yield
        finally:
            self.frozen = old_frozen
            self._error_on_new_guards = old_error

    def _get_key(self) -> tuple[int, int, int, int]:
        """
        Defines the current "state" of the guards we've accumulated in this ShapeEnv.
        Determines when we need to invalidate our cache
        """
        return (
            len(self.replacements),
            len(self.divisible),
            self.num_deferred_runtime_asserts,
            len(self.real_tensor_prop_unbacked_vals),
        )

    def _update_version_counter(self) -> None:
        # if the change to shape env effects self.divisible set
        # _resimplify_floor_div_axioms.
        # This is used to trigger a resimplication of FloorDiv to CleanDivs
        # in implication inside the function resimplify_floor_div.
        if len(self.divisible) != self._prev_cache_key[1]:
            self._resimplify_floor_div_axioms = True

        # The shape environment is queried orders of magnitude more often than
        # it is changed, so we summarise the cache key into a linearly
        # increasing version counter which is cheaper to check in _lru_cache

        # Only update version counter if the state actually changed
        cur_key = self._get_key()

        if self._prev_cache_key != cur_key:
            self._prev_cache_key = cur_key
            self._version_counter += 1

    def _produce_dyn_sizes(
        self,
        ex_size: Sequence[IntLikeType],
        source: Source,
        symbolic_context: SymbolicContext,
    ) -> list[sympy.Expr]:
        return self._produce_dyn_sizes_from_int_tuple(
            tuple(ex_size), source, symbolic_context
        )

    def _produce_dyn_sizes_from_int_tuple(
        self,
        tensor_size: Sequence[IntLikeType],
        source: Source,
        symbolic_context: SymbolicContext,
        hint_overrides: dict[int, int] | None = None,
    ) -> list[sympy.Expr]:
        if not all(not is_symbolic(val) for val in tensor_size):
            raise AssertionError(
                f"Expect size to be a plain tuple of ints but got {tensor_size}"
            )
        from torch._dynamo.source import TensorProperty, TensorPropertySource

        if not hint_overrides:
            hint_overrides = {}

        _assert_symbol_context(symbolic_context)
        dynamic_dims = symbolic_context.dynamic_sizes  # type: ignore[attr-defined]
        constraint_dims = symbolic_context.constraint_sizes  # type: ignore[attr-defined]
        size = []
        for i, val in enumerate(tensor_size):
            sym = self.create_symbol(
                hint_overrides.get(i, val),
                TensorPropertySource(source, TensorProperty.SIZE, i),
                dynamic_dims[i],
                constraint_dims[i],
                do_not_specialize_zero_one=config.backed_size_oblivious,
                symbolic_context=symbolic_context,
            )
            if (
                isinstance(symbolic_context, StatelessSymbolicContext)
                and symbolic_context.specialize_on
            ):
                for specialization in symbolic_context.specialize_on[i]:
                    self.specializations.add(
                        Specialization(
                            TensorPropertySource(source, TensorProperty.SIZE, i),
                            specialization,
                        )
                    )
            if (
                config.backed_size_oblivious
                and isinstance(sym, sympy.Symbol)  # could be static
                and symbol_is_type(sym, SymT.SIZE)
            ):
                self.size_like.add(sym)
            size.append(sym)
        return size

    def create_symbolic_sizes_strides_storage_offset(
        self,
        ex: torch.Tensor,
        source: Source,
        *,
        symbolic_context: SymbolicContext | None = None,
    ) -> tuple[
        tuple[IntLikeType, ...],
        tuple[IntLikeType, ...],
        IntLikeType,
    ]:
        """
        Returns a list of symbolic sizes and strides for the given tensor.
        We try our best to express stride in terms of the sizes, so as to not
        introduce new symbolic variables.
        """

        ex_size = tuple(
            self._maybe_specialize_sym_int_with_hint(sz) for sz in ex.size()
        )
        ex_stride = tuple(
            self._maybe_specialize_sym_int_with_hint(sd) for sd in ex.stride()
        )
        ex_storage_offset = self._maybe_specialize_sym_int_with_hint(
            ex.storage_offset()
        )

        return self._create_symbolic_sizes_strides_storage_offset(
            ex_size,
            ex_stride,
            ex_storage_offset,
            [_is_dim_dynamic(ex, i) for i in range(ex.dim())],
            source,
            symbolic_context=symbolic_context,
        )

    # Dynamo may want to wrap FakeTensors with SymInt sizes up e.g. make_fx(opt_f(), tracing_mode="symbolic").
    # We create symbols in shape_env using the backed hints behind SymInt.

    # Case 1: when SymInt is backed, dynamo can proceed with FakeTensors that have concrete shape.
    # produce_guards will trigger specializations on the outer stuff

    # Case 2: when the SymInt is unbacked, we will throw a data dependent error in guarding_hint_or_throw().
    #
    # It's probably good for now but it's important to note that this approach has implications for
    # the original shape_env when checking guards in different order.

    # Example:
    # ---------
    # Consider a function "opt_f" as shown below:

    # @torch.compile()
    # def opt_f(x: bool, y: Tensor):
    #   if x == True:
    #     return y + torch.randn([4])
    #   else:
    #     return y
    # Depending on the sequence of calls, we might install two different sets of guards:

    # 1. opt_f(False, y):
    #    - "x == False" (always works for any size y)

    # 2. opt_f(True, y):
    #    - Triggers recompilation and results in guards like:
    #      - "x == True and y.size(0) == 4"
    #      - (or "y.size(0) == 4 and x == True")

    # The order of checking the guards matters. In this specific example:
    # If True branch guard check precedes False branch and for True branch, y.size(0) check precedes x == True,
    # we may have an unnecessary shape specialization for y.
    def _maybe_specialize_sym_int_with_hint(
        self, maybe_sym: IntLikeType
    ) -> IntLikeType:
        if not isinstance(maybe_sym, (int, torch.SymInt)):
            raise AssertionError(f"Expected int or SymInt, got {type(maybe_sym)}")
        if is_symbolic(maybe_sym):
            if maybe_sym.node.shape_env is self:
                raise AssertionError(
                    "expect the symbol is created from an shape env other than current one."
                )
            return guarding_hint_or_throw(maybe_sym.node)
        return maybe_sym

    @record_shapeenv_event()
    def _create_symbolic_sizes_strides_storage_offset(
        self,
        # NB: SymInt is allowed here due to nested int, normally you don't
        # actually pass true symbolic sizes to this function
        ex_size: Sequence[IntLikeType],
        ex_stride: Sequence[IntLikeType],
        ex_storage_offset: IntLikeType,
        is_dim_dynamic: Sequence[bool],
        source: Source,
        *,
        symbolic_context: SymbolicContext | None = None,
        hint_overrides: dict[int, int] | None = None,
    ) -> tuple[
        tuple[IntLikeType, ...],
        tuple[IntLikeType, ...],
        IntLikeType,
    ]:
        dim = len(ex_size)

        if not hint_overrides:
            hint_overrides = {}

        # Reimplement the legacy behavior
        if symbolic_context is None:
            constraint_sizes: list[DimConstraint] = [None] * dim
            constraint_strides: list[DimConstraint] = [None] * dim
            dynamic_dims = []
            dynamic_strides = []
            for i in range(dim):
                # NB: This is encapsulation breaking!  Legacy behavior was
                # bad.
                if is_dim_dynamic[i]:
                    r = DimDynamic.DYNAMIC
                elif self.assume_static_by_default:
                    r = DimDynamic.STATIC
                else:
                    r = DimDynamic.DUCK
                dynamic_dims.append(r)
                dynamic_strides.append(r)
            dynamic_dims = [DimDynamic.DUCK] * dim
            dynamic_strides = [DimDynamic.INFER_STRIDE] * dim
            # symbolic_context is None - set one
            symbolic_context = StatelessSymbolicContext(
                dynamic_sizes=dynamic_dims,
                dynamic_strides=dynamic_strides,
                constraint_sizes=constraint_sizes,
                constraint_strides=constraint_strides,
            )
        # We got a StatelessSymbolicContext
        _assert_symbol_context(symbolic_context)
        constraint_sizes = symbolic_context.constraint_sizes  # type: ignore[attr-defined]
        constraint_strides = symbolic_context.constraint_strides  # type: ignore[attr-defined]
        dynamic_sizes = symbolic_context.dynamic_sizes  # type: ignore[attr-defined]
        dynamic_strides = symbolic_context.dynamic_strides  # type: ignore[attr-defined]

        # TODO: make this configurable from outside symbolic_context; we made a symbolic_context
        # decision here where if all sizes are static, we are going to
        # specialize all of the inner strides/offset too. We don't have to
        # do this, and arguably we should ALWAYS allow for dynamic offset,
        # this is cheap.
        # TODO: This should be DYNAMIC, using DUCK for BC
        dynamic_offset = (
            DimDynamic.STATIC
            if all(r == DimDynamic.STATIC for r in dynamic_sizes)
            else DimDynamic.DUCK
        )
        are_sizes_static = all(r == DimDynamic.STATIC for r in dynamic_sizes)

        if len(dynamic_sizes) != dim:
            raise AssertionError(f"{len(dynamic_sizes)} != {dim}")
        if len(dynamic_strides) != dim:
            raise AssertionError(f"{len(dynamic_strides)} != {dim}")
        if len(constraint_sizes) != dim:
            raise AssertionError(f"len(constraint_sizes) != {dim}")
        if len(constraint_strides) != dim:
            raise AssertionError(f"len(constraint_strides) != {dim}")

        from torch._dynamo.source import TensorProperty, TensorPropertySource

        size: list[sympy.Expr] = self._produce_dyn_sizes_from_int_tuple(
            ex_size, source, symbolic_context, hint_overrides=hint_overrides
        )
        # Record tensor exclusion constraints for stable graph selection.
        # The ndim check guards against stale excluded_sizes from graph
        # breaks where the resumed tensor may have different dimensionality.
        # Skip dims with hint overrides: the overridden hint in
        # backed_var_to_val would mismatch the excluded value, causing the
        # not-all check in produce_guards_verbose to emit a guard that
        # immediately fails.
        excluded_sizes = getattr(symbolic_context, "excluded_sizes", None)
        if (
            excluded_sizes
            and len(excluded_sizes) == dim
            and any(v is not None for v in excluded_sizes)
        ):
            for i in range(dim):
                ev = excluded_sizes[i]
                if (
                    ev is not None
                    and isinstance(size[i], sympy.Symbol)
                    and i not in (hint_overrides or {})
                ):
                    self._record_exclusion_constraint(size[i], ev)
        stride = self._compute_symbolic_stride(
            source,
            size,
            ex_size,
            ex_stride,
            dynamic_strides,
            constraint_strides,
            are_sizes_static,
            symbolic_context,
        )

        sym_sizes = [
            self.create_symintnode(
                sym,
                hint=hint_overrides.get(i, hint),
                source=TensorPropertySource(source, TensorProperty.SIZE, i),
            )
            for i, (sym, hint) in enumerate(zip(size, ex_size))
        ]

        for i, sym in enumerate(sym_sizes):
            if isinstance(sym, torch.SymInt) and i in hint_overrides:
                self.var_to_hint_override[sym.node.expr] = hint_overrides[i]

        sym_stride = []
        for i, stride_expr in enumerate(stride):
            # NB: Don't duck size the stride; instead use the expression
            # we computed
            if stride_expr is None:
                raise AssertionError(f"stride_expr is None for index {i}")
            # self.backed_var_to_val will have the up to date hint value for each symbols
            # including overridden hints.
            hint_stride = stride_expr.xreplace(self.backed_var_to_val)
            if isinstance(hint_stride, (int, sympy.core.numbers.Integer)):
                hint_stride = int(hint_stride)
            else:
                hint_stride = ex_stride[i]
            sym_stride.append(
                self.create_symintnode(
                    stride_expr,
                    hint=hint_stride,
                    source=TensorPropertySource(source, TensorProperty.STRIDE, i),
                )
            )
        sym_storage_offset = self.create_symintnode(
            self.create_symbol(
                ex_storage_offset,
                TensorPropertySource(source, TensorProperty.STORAGE_OFFSET),
                dynamic_dim=dynamic_offset,
                constraint_dim=None,
                symbolic_context=symbolic_context,
            ),
            hint=ex_storage_offset,
            source=TensorPropertySource(source, TensorProperty.STORAGE_OFFSET),
        )
        return tuple(sym_sizes), tuple(sym_stride), sym_storage_offset

    def _compute_symbolic_stride(
        self,
        source: Source,
        size: Sequence[sympy.Expr],
        ex_size: Sequence[IntLikeType],
        ex_stride: Sequence[IntLikeType],
        dynamic_strides: Sequence[DimDynamic],
        constraint_strides: Sequence[
            StrictMinMaxConstraint | RelaxedUnspecConstraint | None
        ],
        are_sizes_static: bool,
        symbolic_context: SymbolicContext,
    ) -> list[sympy.Expr]:
        from torch._dynamo.source import TensorProperty, TensorPropertySource

        stride: list[sympy.Expr | None] = [None] * len(size)
        candidates: dict[IntLikeType, sympy.Expr] = {}

        # iterate over unbound strides in val ascending order with
        # index descending as a tie breaker since for cases like
        # [(1, 1), (1, 0)], we want to fill in the right most
        # stride first.
        val_list = [(val, -i) for i, val in enumerate(ex_stride)]
        val_list.sort(key=_nested_int_aware_sort)

        for val, neg_i in val_list:
            i = -neg_i
            contiguous_stride = (
                i != len(ex_stride) - 1
                and ex_stride[i] == ex_size[i + 1] * ex_stride[i + 1]
            )
            if val in (0, 1) and not contiguous_stride:
                out_stride = sympy.Integer(val)
            else:
                dynamic_stride = dynamic_strides[i]
                if dynamic_stride == DimDynamic.INFER_STRIDE and val in candidates:
                    # Set stride to a candidate only for DimDynamic.INFER_STRIDE
                    out_stride = candidates[val]
                else:
                    # Set INFER_STRIDE to STATIC or DUCK depending on sizes
                    dyn_stride = dynamic_stride
                    if dynamic_stride == DimDynamic.INFER_STRIDE:
                        dyn_stride = (
                            DimDynamic.STATIC if are_sizes_static else DimDynamic.DUCK
                        )
                    out_stride = self.create_symbol(
                        val,
                        TensorPropertySource(source, TensorProperty.STRIDE, i),
                        dynamic_dim=dyn_stride,
                        constraint_dim=constraint_strides[i],
                        symbolic_context=symbolic_context,
                    )
            stride[i] = out_stride
            candidates[ex_size[i] * val] = size[i] * out_stride

        if not all(x is not None for x in stride):
            raise AssertionError("All stride elements must be non-None")
        return stride

    @record_shapeenv_event()
    def create_symintnode(
        self,
        sym: sympy.Expr,
        *,
        hint: int | None,
        source: Source | None = None,
    ) -> IntLikeType:
        """Create a SymInt value from a symbolic expression

        If you know what the current hint value of the SymInt to be created
        is, pass it into hint.  Otherwise, pass None and we will make our best
        guess

        """
        if self._translation_validation_enabled and source is not None:
            # Create a new symbol for this source.
            symbol = self._create_symbol_for_source(source)
            if symbol is None:
                raise AssertionError("symbol must not be None")

            # Create a new FX placeholder and Z3 variable for 'symbol'.
            fx_node = self._create_fx_placeholder_and_z3var(symbol, int)

            # Add an equality assertion for the newly created symbol and 'sym'.
            self._add_assertion(sympy.Eq(symbol, sym))
        else:
            fx_node = None

        out: IntLikeType
        if isinstance(sym, sympy.Integer):
            if hint is not None:
                if int(sym) != hint:
                    raise AssertionError(f"int(sym)={int(sym)} != hint={hint}")
            out = int(sym)
        else:
            # How can this occur? When we mark_unbacked, we end up with a real
            # tensor that has hints for all sizes, but we MUST NOT create a
            # SymNode with a hint, because we're hiding the hint from our eyes
            # with the unbacked Symbol.  And in fact, the hint compute may be
            # inconsistent with size oblivious tests.
            if free_unbacked_symbols(sym):
                hint = None
            out = SymInt(SymNode(sym, self, int, hint, fx_node=fx_node))
        return out

    @record_shapeenv_event()
    def create_symfloatnode(
        self,
        sym: sympy.Expr,
        *,
        hint: int | float | bool | None,
        source: Source | None = None,
    ) -> FloatLikeType:
        """Create a SymFloat value from a symbolic expression"""
        if self._translation_validation_enabled and source is not None:
            # Create a new symbol for this source.
            symbol = self._create_symbol_for_source(source)
            if symbol is None:
                raise AssertionError("symbol must not be None")

            # Create a new FX placeholder and Z3 variable for 'symbol'.
            fx_node = self._create_fx_placeholder_and_z3var(symbol, float)

            # Add an equality assertion for the newly created symbol and 'sym'.
            self._add_assertion(sympy.Eq(symbol, sym))
        else:
            fx_node = None

        out: FloatLikeType
        if isinstance(sym, sympy.Float):
            if hint is not None:
                if float(sym) != hint:
                    raise AssertionError(f"float(sym)={float(sym)} != hint={hint}")
            out = float(sym)
        else:
            # You could give this the same treatment as SymInt above if
            # you supported mark_unbacked on a float, but it's a kind of
            # strange thing to do though because floats don't get 0/1
            # specialization anyway
            if free_unbacked_symbols(sym):
                if hint is not None:
                    raise AssertionError(
                        f"hint must be None for unbacked symbol: {sym}"
                    )
            out = SymFloat(SymNode(sym, self, float, hint, fx_node=fx_node))
        return out

    @record_shapeenv_event()
    def _record_exclusion_constraint(self, sym: sympy.Symbol, val: int) -> None:
        self.exclusion_constraints.append((sym, val))

    @record_shapeenv_event()
    def create_unspecified_symint_and_symbol(
        self,
        value: int,
        source: Source,
        dynamic_dim: DimDynamic,
        excluded_value: int | None = None,
    ) -> IntLikeType:
        """Create a SymInt wrapping a new unspecified symbol"""
        sym = self.create_unspecified_symbol(
            value,
            source=source,
            dynamic_dim=dynamic_dim,
        )
        if excluded_value is not None:
            self._record_exclusion_constraint(sym, excluded_value)
        return self.create_symintnode(
            sym,
            hint=value,
            source=source,
        )

    def create_symboolnode(self, sym: sympy.Expr) -> SymBool:
        """Create a SymBool object from a sympy boolean expression"""
        # This function is only being used in serialization, so we do not track it
        # for validation.
        return SymBool(SymNode(sym, self, bool, None))

    def _log_create_unbacked_symbol(
        self,
        prefix: str,
        symbol: sympy.Symbol,
        vr: ValueRanges,
        source: Source | None = None,
        sym_node: SymNode | None = None,
    ) -> None:
        is_debug = config.extended_debug_create_symbol is not None and str(
            symbol
        ) in config.extended_debug_create_symbol.split(",")
        sloc: str | SLoc
        if source is None:
            sloc, maybe_extra_debug = self._get_stack_summary(is_debug)
        else:
            sloc, maybe_extra_debug = source.name, ""
        log.info(
            "%s %s [%s, %s] %s%s",
            prefix,
            symbol,
            vr.lower,
            vr.upper,
            sloc,
            maybe_extra_debug,
            stack_info=is_debug,
        )
        trace_structured(
            "create_unbacked_symbol",
            metadata_fn=lambda: {
                "symbol": str(symbol),
                "node_id": id(sym_node),
                "vr": f"[{vr.lower}, {vr.upper}]",
                "user_stack": structured.get_user_stack(3),
                "stack": structured.get_framework_stack(),
            },
        )

    @record_shapeenv_event()
    def create_unbacked_symfloat(self) -> SymFloat:
        """Create a symbolic float without a hint value"""
        symbol: sympy.Symbol = make_symbol(
            SymT.UNBACKED_FLOAT, self.unbacked_symfloat_counter
        )
        self.unbacked_symfloat_counter += 1
        self.counter["create_unbacked_symbol"] += 1
        if not self._ignore_fresh_unbacked_symbols_tls():
            self.pending_fresh_unbacked_symbols.append(symbol)
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        vr = self.var_to_range[symbol] = ValueRanges.unknown()
        if not vr.is_float:
            raise AssertionError("vr must be float")
        sloc = self._get_sloc()
        self.var_to_range_sloc[symbol] = ValueRangesSLoc(sloc, sloc)

        # Create a new FX placeholder and Z3 variable for 'symbol'.
        fx_node = self._create_fx_placeholder_and_z3var(symbol, float)

        sym_node = SymNode(symbol, self, float, None, fx_node=fx_node)
        self._log_create_unbacked_symbol(
            "create_unbacked_symfloat", symbol, vr, sym_node=sym_node
        )

        return SymFloat(sym_node)

    @record_shapeenv_event()
    def create_unbacked_symint(self, source: Source | None = None) -> SymInt:
        """Create a symbolic integer without a hint value"""
        symbol: sympy.Symbol = make_symbol(
            SymT.UNBACKED_INT, self.unbacked_symint_counter, integer=True
        )
        self.unbacked_symint_counter += 1
        if not self._ignore_fresh_unbacked_symbols_tls():
            self.pending_fresh_unbacked_symbols.append(symbol)
        self.counter["create_unbacked_symbol"] += 1
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        vr = self.var_to_range[symbol] = self._default_unspecified_value_range()
        if not vr.is_int:
            raise AssertionError("vr must be int")
        sloc = self._get_sloc()
        self.var_to_range_sloc[symbol] = ValueRangesSLoc(sloc, sloc)

        # Create a new FX placeholder and Z3 variable for 'symbol'.
        fx_node = self._create_fx_placeholder_and_z3var(symbol, int)

        sym_node = SymNode(symbol, self, int, None, fx_node=fx_node)
        self._log_create_unbacked_symbol(
            "create_unbacked_symint", symbol, vr, source, sym_node=sym_node
        )
        return SymInt(sym_node)

    def is_unbacked_symint(self, symbol: sympy.Symbol) -> bool:
        """Check if a sympy symbol matches the naming convention for unbacked symbols"""
        return symbol_is_type(symbol, SymT.UNBACKED_INT)

    @record_shapeenv_event()
    def create_unbacked_symbool(self) -> SymBool:
        """Create a symbolic boolean without a hint value"""
        symbol: sympy.Symbol = make_symbol(
            SymT.UNBACKED_INT, self.unbacked_symint_counter, integer=True
        )
        self.unbacked_symint_counter += 1
        if not self._ignore_fresh_unbacked_symbols_tls():
            self.pending_fresh_unbacked_symbols.append(symbol)
        self.counter["create_unbacked_symbol"] += 1
        self.var_to_stack[symbol] = CapturedTraceback.extract(skip=1)
        vr = self.var_to_range[symbol] = ValueRanges(0, 1)
        if not vr.is_int:
            raise AssertionError("vr must be int")
        sloc = self._get_sloc("default value range for unbacked SymBool")
        self.var_to_range_sloc[symbol] = ValueRangesSLoc(sloc, sloc)

        # Create a new FX placeholder and Z3 variable for 'symbol'.
        fx_node = self._create_fx_placeholder_and_z3var(symbol, bool)

        sym_node = SymNode(sympy.Eq(symbol, 1), self, bool, None, fx_node=fx_node)
        self._log_create_unbacked_symbol(
            "create_unbacked_symbool", symbol, vr, sym_node=sym_node
        )

        return SymBool(sym_node)

    @record_shapeenv_event()
    def create_unspecified_symbol(
        self,
        val: int | SymInt | float | SymFloat,
        source: Source,
        dynamic_dim: DimDynamic = DimDynamic.DUCK,
        constraint_dim: DimConstraint = None,  # NB: includes None
        symbolic_context: StatelessSymbolicContext | None = None,
    ) -> sympy.Expr:
        """
        Create a symbol with an unspecified value

        Compared to standard symbols we do not assume the value is positive,
        nor do we specialze on zero or one values.
        """
        # 'positive' is None for unspecified symbols, since we can't
        # assume that it will be neither positive nor negative.

        # We don't want to specialize zero one val for unspecified symbol
        # so that we can always get a new symbol despite val.
        return self.create_symbol(
            val,
            source,
            dynamic_dim,
            constraint_dim,
            positive=None,
            do_not_specialize_zero_one=True,
            symbolic_context=symbolic_context,
        )

    @record_shapeenv_event()
    def create_symbol(
        self,
        val: int,
        source: Source,
        dynamic_dim: DimDynamic = DimDynamic.DUCK,
        constraint_dim: DimConstraint = None,  # NB: includes None
        positive: bool | None = True,
        do_not_specialize_zero_one: bool = False,
        symbolic_context: StatelessSymbolicContext | None = None,
    ) -> sympy.Expr:
        """Create a new symbol which is tracked by this ShapeEnv"""
        # check if constraint_dim is actually static integer
        if (
            isinstance(constraint_dim, StrictMinMaxConstraint)
            and constraint_dim.vr.lower == constraint_dim.vr.upper
        ):
            dynamic_dim = DimDynamic.STATIC
            if constraint_dim.vr.lower != val:
                raise ConstraintViolationError(
                    f"Static shape constraint of {constraint_dim.vr.lower} does not match input size of {val}, "
                    f"for {source.name}"
                )
            if symbolic_context:
                from torch._dynamo.source import TensorPropertySource

                if not isinstance(source, TensorPropertySource):
                    raise AssertionError(
                        f"Expected TensorPropertySource, got {type(source)}"
                    )
                # TODO: storage_offset handling?
                if source.idx is None:
                    raise AssertionError("source.idx must not be None")
                symbolic_context.dynamic_sizes[source.idx] = dynamic_dim
                symbolic_context.constraint_sizes[source.idx] = None
            constraint_dim = None

        # see note [Tensor Fakification and Symbol Caching]
        source_name = source.name
        if (
            isinstance(symbolic_context, StatefulSymbolicContext)
            and id(self) not in symbolic_context.shape_env_to_source_to_symbol_cache
        ):
            symbolic_context.shape_env_to_source_to_symbol_cache[id(self)] = {}

        if (
            isinstance(symbolic_context, StatefulSymbolicContext)
            and source_name
            and (
                source_name
                in symbolic_context.shape_env_to_source_to_symbol_cache[id(self)]
            )
        ):
            return symbolic_context.shape_env_to_source_to_symbol_cache[id(self)][
                source_name
            ]

        if dynamic_dim is DimDynamic.UNBACKED:
            # Check if this unbacked dimension has a shape_id.
            # If so, we allocate a fresh symbol but add a runtime equality check
            # via torch._check against the existing symbols with the same shape_id.
            shape_id = None
            unbacked_min = None
            unbacked_max = None
            if (
                isinstance(symbolic_context, StatelessSymbolicContext)
                and symbolic_context.shape_ids is not None
            ):
                from torch._dynamo.source import TensorPropertySource

                if isinstance(source, TensorPropertySource) and source.idx is not None:
                    shape_id = symbolic_context.shape_ids.get(source.idx)

            # Check for unbacked bounds
            if (
                isinstance(symbolic_context, StatelessSymbolicContext)
                and symbolic_context.unbacked_bounds is not None
            ):
                from torch._dynamo.source import TensorPropertySource

                if isinstance(source, TensorPropertySource) and source.idx is not None:
                    bounds = symbolic_context.unbacked_bounds.get(source.idx)
                    if bounds is not None:
                        unbacked_min, unbacked_max = bounds

            # Always allocate a fresh unbacked symbol
            out = self.create_unbacked_symint(source).node.expr
            self._constrain_range_for_size(out)

            # Apply min/max bounds via torch._check if specified
            if unbacked_min is not None or unbacked_max is not None:
                out_symint = self.create_symintnode(out, hint=None)
                if unbacked_min is not None:
                    torch._check(out_symint >= unbacked_min)
                if unbacked_max is not None:
                    torch._check(out_symint <= unbacked_max)

            # Add runtime equality check for shape_id if applicable
            if shape_id is not None:
                if shape_id in self._shape_id_to_unbacked_symbol:
                    # Add runtime equality check instead of reusing the same symbol
                    existing_sym = self._shape_id_to_unbacked_symbol[shape_id]
                    existing_symint = self.create_symintnode(existing_sym, hint=None)
                    out_symint = self.create_symintnode(out, hint=None)
                    torch._check(out_symint == existing_symint)
                else:
                    self._shape_id_to_unbacked_symbol[shape_id] = out

            self.unbacked_inputs.add(out)

            if isinstance(symbolic_context, StatefulSymbolicContext) and source_name:
                symbolic_context.shape_env_to_source_to_symbol_cache[id(self)][
                    source_name
                ] = out
            return out

        if do_not_specialize_zero_one:
            specialize_zero_one = False
        else:
            specialize_zero_one = self.specialize_zero_one

        if not isinstance(source, Source):
            raise AssertionError(f"{type(source)} {source}")
        if positive and val < 0:
            raise AssertionError(f"positive set for negative value: {val}")
        # It's always sound to allocate a symbol as DYNAMIC.  If the user
        # constrained the symbol, force the symbolic_context to DYNAMIC, because our
        # constraint code will do weird stuff if, e.g., it's duck shaped
        if constraint_dim is not None:
            dynamic_dim = DimDynamic.DYNAMIC

        if dynamic_dim is DimDynamic.STATIC:
            out = sympy.Integer(val)
            if isinstance(symbolic_context, StatefulSymbolicContext) and source_name:
                symbolic_context.shape_env_to_source_to_symbol_cache[id(self)][
                    source_name
                ] = out
            return out

        elif dynamic_dim is DimDynamic.DUCK:
            # duck_shape can be used to globally turn off duck shaping, even
            # if it was requested
            duck = self.duck_shape
        elif dynamic_dim is DimDynamic.DYNAMIC:
            duck = False
        else:
            raise AssertionError(f"unhandled dynamic_dim {dynamic_dim}")

        sloc = self._get_sloc()

        if val in (0, 1) and specialize_zero_one:
            if val == 0:
                return sympy.S.Zero
            else:
                return sympy.S.One
        elif not duck or val not in self.val_to_var:
            # If we're not duck shaping, we always create a new symbol
            # Even if we're duck shaping, if we haven't seen this particular
            # value before, we also create a new symbol
            symbol_id = self._generate_unique_id(source.name)
            if type(val) is int or is_nested_int(val):
                sympy_expr = make_symbol(
                    SymT.SIZE, symbol_id, positive=positive, integer=True
                )
            else:
                sympy_expr = make_symbol(
                    SymT.FLOAT, symbol_id, positive=positive, real=True
                )
            self.source_to_var[source_name] = sympy_expr
            # We always associate vars to vals
            if isinstance(val, int):
                self.backed_var_to_val[sympy_expr] = sympy.Integer(val)
            elif isinstance(val, float):
                self.backed_var_to_val[sympy_expr] = sympy.Float(val)
            else:
                # Only used for jagged layout nested tensors
                self.backed_var_to_val[sympy_expr] = SingletonInt(
                    val.node.nested_int(), coeff=val.node.nested_int_coeff()
                )

            # Do the appending later, because we always want to populate this
            self.var_to_sources[sympy_expr] = []
            # Create a Z3 variable for the new symbol.
            self._add_z3var(sympy_expr, int)

            if duck:
                # Make sure to reuse this symbol for subsequent duck shaping

                self.val_to_var[val] = sympy_expr

            if isinstance(val, int):
                if positive:
                    # Add assertions for the newly created symbols
                    self._add_assertion(sympy_expr > 1)

                    # Apply default range, which assumes not zero-one
                    self.var_to_range[sympy_expr] = self._default_value_range(
                        do_not_specialize_zero_one
                    )
                    self.var_to_range_sloc[sympy_expr] = ValueRangesSLoc(
                        self._get_sloc(
                            "user code shown is first use of this value--the guard itself is not "
                            "due user code but due to 0/1 specialization in the framework; to "
                            "avoid specialization try torch._dynamo.decorators.mark_unbacked(tensor, dim)"
                            if self.specialize_zero_one
                            else None
                        ),
                        sloc,
                    )
                else:
                    self.var_to_range[sympy_expr] = (
                        self._default_unspecified_value_range()
                    )
                    self.var_to_range_sloc[sympy_expr] = ValueRangesSLoc(sloc, sloc)

                # Small performance optimization: if we have a min-max constraint,
                # we can proactively narrow to that range
                if isinstance(constraint_dim, StrictMinMaxConstraint):
                    if duck:
                        raise AssertionError(
                            "duck must be False for StrictMinMaxConstraint"
                        )
                    self._update_var_to_range(
                        sympy_expr, constraint_dim.vr, is_constraint=True
                    )

                vr = self.var_to_range[sympy_expr]
                if not vr.is_int:
                    raise AssertionError("vr must be int")

                if val not in vr:
                    raise ConstraintViolationError(
                        f"{val} not in range [{vr.lower}, {vr.upper}]"
                    )

                range_str = f"[{vr.lower}, {vr.upper}]"
            elif isinstance(val, float):
                self.var_to_range[sympy_expr] = vr = ValueRanges(-sympy.oo, sympy.oo)
                self.var_to_range_sloc[sympy_expr] = ValueRangesSLoc(sloc, sloc)
                range_str = f"[{vr.lower}, {vr.upper}]"
                if not vr.is_float:
                    raise AssertionError("vr must be float")
            else:
                # Skip var_range logic for SingletonInt
                # Only used for jagged layout nested tensors
                range_str = ""

            r = sympy_expr

            is_debug = config.extended_debug_create_symbol is not None and str(
                sympy_expr
            ) in config.extended_debug_create_symbol.split(",")
            maybe_more_info = ""
            if not is_debug and os.getenv("TORCHDYNAMO_EXTENDED_ADVICE", "1") not in (
                "0",
                "",
            ):
                maybe_more_info = (
                    ", for more info run with "
                    f'TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="{sympy_expr}" '
                    "or to suppress this message run with "
                    'TORCHDYNAMO_EXTENDED_ADVICE="0"'
                )
            sloc, maybe_extra_debug = self._get_stack_summary(is_debug)
            self.log.info(
                "create_symbol %s = %s for %s %s %s%s%s",
                sympy_expr,
                val,
                source.name,
                range_str,
                sloc,
                maybe_more_info,
                maybe_extra_debug,
                stack_info=is_debug,
            )
            trace_structured(
                "create_symbol",
                metadata_fn=lambda: {
                    "symbol": str(sympy_expr),
                    "val": repr(val),
                    "vr": range_str,
                    "source": source.name,
                    "user_stack": structured.from_traceback(
                        TracingContext.extract_stack()
                    ),
                    "stack": structured.from_traceback(
                        CapturedTraceback.extract(skip=1).summary()
                    ),
                },
            )

            self.counter["create_symbol"] += 1
        else:
            # This implements duck-shaping: input sizes that match are assigned
            # the same symint
            r = self.val_to_var[val]
            self.source_to_var[source_name] = r
            self.log.debug("create_symbol %s duck sized %s", r, source.name)

        if isinstance(r, sympy.Symbol):
            r_sources = self.var_to_sources[r]
            r_sources.append(source)
            if not source.is_ephemeral() and r_sources[0].is_ephemeral():
                # prefer non-ephemeral source first since it may be guarded on later
                r_sources[0], r_sources[-1] = r_sources[-1], r_sources[0]

            # This ensures we get zeros in symbol_guard_counts, which makes
            # some queries simpler (since we will accumulate mass on 0 this
            # way)
            self.symbol_guard_counter[r] = 0

        if isinstance(symbolic_context, StatefulSymbolicContext) and source_name:
            symbolic_context.shape_env_to_source_to_symbol_cache[id(self)][
                source_name
            ] = r
        return r

    def add_backed_var_to_val(self, expr: sympy.Symbol, val: int) -> None:
        """Adds a new symbol to the symbolic environment."""
        log.debug("add_backed_var_to_val %s %s", expr, val, stack_info=True)
        if expr in self.backed_var_to_val:
            raise AssertionError(f"{expr} already exists")
        self.backed_var_to_val[expr] = sympy.Integer(val)

    @property
    @deprecated(
        "var_to_val is deprecated, use backed_var_to_val instead",
        category=FutureWarning,
    )
    def var_to_val(self) -> dict[sympy.Symbol, sympy.Integer]:
        """Deprecated: use backed_var_to_val instead."""
        return self.backed_var_to_val

    @deprecated(
        "add_var_to_val is deprecated, use add_backed_var_to_val instead",
        category=FutureWarning,
    )
    def add_var_to_val(self, expr: sympy.Symbol, val: int) -> None:
        """Deprecated: use add_backed_var_to_val instead."""
        return self.add_backed_var_to_val(expr, val)

    def _debug_name(self, source: Source) -> str:
        src_name = source.name
        return self.source_name_to_debug_name.get(src_name, src_name)

    def _render_range_for_constraint_violation(
        self, source: Source, c: StrictMinMaxConstraint | RelaxedUnspecConstraint
    ) -> str:
        if isinstance(c, StrictMinMaxConstraint):
            lower, upper = c.vr.lower, c.vr.upper
            default = self._default_value_range()
            if lower <= default.lower:
                lower = None
            if upper >= default.upper:
                upper = None
            c_render = (
                f"{self._debug_name(source)} = {source.name} in the specified range"
            )
            if lower is not None and upper is not None:
                c_render += f" {lower} <= {self._debug_name(source)} <= {upper}"
            elif lower is None and upper is not None:
                c_render += f" {self._debug_name(source)} <= {upper}"
            elif lower is not None and upper is None:
                c_render += f" {lower} <= {self._debug_name(source)}"
            return c_render
        return c.render(source)

    def produce_guards(self, *args: Any, **kwargs: Any) -> list[str]:
        """
        Like produce_guards_verbose, but only returns the non-verbose python guard expressions
        (no verbose guards produced.)
        """
        return self.produce_guards_verbose(*args, **kwargs, langs=("python",))[0].exprs

    def produce_guards_verbose(
        self,
        placeholders: Sequence[FakeTensor],
        sources: Sequence[Source],
        source_ref: Callable[[Source], str] = lambda n: n.name,
        *,
        guards: list[ShapeGuard] | None = None,
        input_contexts: DimList[SymbolicContext] | None = None,
        # Encodes user-specified input shape equations of the form s = s' and s = fn(s').
        # (See docs on EqualityConstraint for details of the encoding.)
        equalities_inputs: EqualityConstraint | None = None,
        _simplified: bool = False,
        # Indicates if we should produce guards for known static values.
        ignore_static: bool = True,
        langs: tuple[str, ...] = ("python", "verbose_python"),
    ) -> list[_ShapeGuardsHelper]:
        """
        Generates a list of guards strings which, when evaluated in a context that
        defines tensors for all the sources, returns True or False depending
        on if the guards in the list evaluated to True or not.  Primarily used by Dynamo,
        but this is also helpful for manual testing of guards (see
        evaluate_guards_for_args)

        For convenience in testing, a source is allowed to be a str,
        in which case we will assume it is a LocalSource

        simplified lets you omit duck sizing, equality and 0/1 guards.
        This is useful for testing when you don't care about the boilerplate
        guards, and it may be helpful for user output too (be careful though;
        some equality guards are nontrivial!  It would be nice to get simplified
        output to print them too).  It's private because it's not
        intended for normal use

        Returns guards in python and python with verbose comments (verbose) by
        default.
        """
        self.log.info("produce_guards")

        # Check if we get to the same ShapeEnv state by replaying the recorded events.
        # This will create a new ShapeEnv instance, and call all recorded function
        # calls on this new instance. Finally, it will check whether this new instance
        # has equal state.
        #
        # It's important that we do it in the beginning of this function, since it modifies
        # self.dim_constraints through its execution. Changes that happen in this method
        # aren't interesting, since this is the function call we wish to reproduce at the
        # end. If we wish to simply reproduce ShapeEnv instances even after this call,
        # this method should also be recorded.
        if self.check_recorded_events:
            shape_env = replay_shape_env_events(self.events)
            self.check_equal(shape_env)

        if len(placeholders) != len(sources):
            raise AssertionError(f"len({placeholders}) != len({sources})")
        Tensorlike = (torch.Tensor, FakeTensorMeta)

        def _create_no_constraints_context(t: Tensor) -> StatelessSymbolicContext:
            return StatelessSymbolicContext(
                # Ignored; only the constraints part is relevant below.
                dynamic_sizes=[DimDynamic.DYNAMIC] * t.dim(),
                dynamic_strides=[DimDynamic.INFER_STRIDE] * t.dim(),
                constraint_sizes=[None] * t.dim(),
                constraint_strides=[None] * t.dim(),
            )

        # Expand optional inputs, or verify invariants are upheld
        if input_contexts is None:
            # pyrefly: ignore [bad-assignment]
            input_contexts = [
                # pyrefly: ignore [bad-argument-type]
                _create_no_constraints_context(t) if isinstance(t, Tensorlike) else None
                for t in placeholders
            ]
        else:
            if len(input_contexts) != len(placeholders):
                raise AssertionError("len(input_contexts) != len(placeholders)")

            for i, (t, context) in enumerate(zip(placeholders, input_contexts)):
                if isinstance(t, Tensorlike):
                    if context is None:
                        input_contexts[i] = _create_no_constraints_context(t)
                else:
                    if not isinstance(t, (SymInt, int, SymFloat, float)):
                        raise AssertionError(
                            f"Expected SymInt, int, SymFloat, or float, got {type(t)}"
                        )
                    if isinstance(context, list):
                        raise AssertionError("context must not be a list")

        # It took a lot of sweat to figure out the algorithm here.  Let's
        # explain how it works.
        #
        # The ShapeEnv lifecycle looks something like this:
        #
        # - For each input, you either generate a fresh Sympy symbol (s0) to
        #   represent its value (a binding site), or you reuse some
        #   preexisting symbol or expression, skipping the symbol allocation
        #   (e.g., duck sizing to a preexisting symbol, or expressing a
        #   stride as a multiplication of a separate stride and size.)
        #   Naively, you might expect to bind a fresh Sympy symbol for
        #   every input, but this is fairly wasteful as most of these
        #   symbols immediately simplify away, and if you don't eagerly
        #   specialize, e.g., 0/1 symbols, you end up with very complicated
        #   expressions that are not optimizable in practice.
        #
        # - You perform some compute on these symbols, occasionally
        #   introducing guards on boolean expressions on these symbols.
        #   In particular, whenever we guard on equality (_maybe_guard_rel),
        #   we can simplify shapes; e.g., when s0 == s1 * 2, we can now
        #   replace all occurrences of s0 with s1 * 2.  Sometimes, a
        #   boolean expression evaluation doesn't introduce a guard, as
        #   the guard is already entailed by the simplifications we have
        #   applied.
        #
        # - In the end, you have a bunch of replacements (saying how to
        #   simplify shapes) and a bunch of guards (all the equality guards
        #   are trivial, because they're covered by the replacements).
        #
        # From the ShapeEnv, we must generate a Python expression that, when
        # evaluated on a set of inputs, tells us whether or not these boolean
        # expressions would have evaluated in the same way.  However,
        # we cannot easily compute this, as we elide recording boolean
        # expressions when we think they are vacuously true.  Thus, we seek
        # an approximation: we must generate an expression, if true, would have
        # produced an "equivalent" ShapeEnv, which would answer guard
        # expressions in the same way.
        #
        # Our notion of equivalence is a bit subtle.  For example, consider
        # the ShapeEnv created from an input of size (5, 4) versus (4, 4)
        # (no other guards.)  Duck sizing would generate (s0, s1) in the first
        # case but (s0, s0) in the second.  We do NOT assume that size
        # variables are disjoint; so in fact a graph that assumes the input
        # could be (s0, s1) subsumes (s0, s0) (setting s0 == s1), but not
        # vice versa.  However, consider an analogous case (1,) versus (2,).
        # Duck sizing generates (1,) and (s0,); the (s0,) graph does NOT
        # subsume the (1,) graph because we assume that any size variables
        # is NOT 0/1 (and make simplifications according to this; e.g., if
        # we queried s0 == 0, we would immediately return False without
        # returning a guard.)
        #
        # So, it is perhaps easier to flip things on their head: the guard
        # expressions we generate here say what simplifications are valid,
        # and what are not. Below, we explain each of the guard expressions
        # we generate

        # TODO: Make this more efficient by binding all the size/stride/offsets
        # to locals before performing tests on them.

        from torch._dynamo.source import TensorProperty, TensorPropertySource

        # Actual codegen must be delayed as we don't necessarily know what
        # the symbol mapping is
        input_guards = []

        symbol_to_source: dict[sympy.Symbol, list[Source]] = collections.defaultdict(
            list
        )
        symbol_to_constraints: defaultdict[sympy.Symbol, set[Constraint]] = (
            collections.defaultdict(set)
        )
        constraint_violations: list[tuple[bool, str, Callable[[], str]]] = []

        printers: list[_ShapeGuardPrinter] = []
        py_printer = ShapeGuardPythonPrinter(
            symbol_to_source, source_ref, self.var_to_sources
        )
        for lang in langs:
            if lang in ["python", "verbose_python"]:
                printers.append(py_printer)
            elif lang == "cpp":
                printers.append(
                    _ShapeGuardCppPrinter(
                        symbol_to_source, source_ref, self.var_to_sources
                    )
                )
            else:
                raise NotImplementedError(f"Unknown lang: {lang}")

        def record_constraint_violation(
            warn_only: bool,
            debug_name: str,
            msg: str,
            hint: Callable[[], str] | None = None,
        ) -> None:
            constraint_violations.append(
                (warn_only, debug_name, lambda: f"{msg}{hint()}" if hint else msg)
            )

        def is_dim(src: object) -> TypeGuard[TensorPropertySource]:
            return (
                isinstance(src, TensorPropertySource)
                and src.prop is TensorProperty.SIZE
            )

        if equalities_inputs:
            source_index = {}
            for i, src in enumerate(sources):
                source_index[src.name] = i

            def get_expression(tensor_dim_src: Source) -> sympy.Expr:
                fake = placeholders[source_index[tensor_dim_src.base.name]]  # type: ignore[attr-defined]
                if tensor_dim_src.idx is None:  # type: ignore[attr-defined]
                    raise AssertionError("tensor_dim_src.idx must not be None")
                symint = fake.shape[tensor_dim_src.idx]  # type: ignore[attr-defined]
                if isinstance(symint, torch.SymInt):
                    return symint.node.expr
                else:
                    if type(symint) is not int:
                        raise AssertionError(f"Expected int, got {type(symint)}")
                    return sympy.Integer(symint)

            for src1, src2 in equalities_inputs.source_pairs:
                expr1, expr2 = get_expression(src1), get_expression(src2)  # type: ignore[]
                # Check whether given input shape values satisfy a specified equation s = s'.
                # - Raise when the equation was violated by the given input shape values.
                # - Otherwise issue a guard to constrain them.
                concrete_val = self.evaluate_expr(sympy.Eq(expr1, expr2))
                if not concrete_val:
                    raise ConstraintViolationError(
                        f"{src1.name} = {expr1 if isinstance(expr1, int) else expr1.xreplace(self.backed_var_to_val)}"
                        " is not equal to "
                        f"{src2.name} = {expr2 if isinstance(expr2, int) else expr2.xreplace(self.backed_var_to_val)}"
                    )

            for srcEq, root, fn in equalities_inputs.derived_equalities:
                expr1 = get_expression(srcEq)
                # recall that root is either a phantom symbol or an input source
                if isinstance(root, sympy.Symbol):
                    expr2, debug_name = root, self.var_to_sources[root][0].name
                elif isinstance(root, sympy.Integer):
                    expr2, debug_name = root, str(root)
                else:
                    expr2, debug_name = get_expression(root), self._debug_name(root)
                expr2_ = fn(expr2)
                # Check whether given input shape values satisfy a specified equation s = fn(s').
                # - Raise when the equation was violated by the given input shape values.
                # - Otherwise issue a guard to constrain them.
                concrete_val = self.evaluate_expr(sympy.Eq(expr1, expr2_))
                if not concrete_val:
                    raise ConstraintViolationError(
                        f"Expected input {srcEq.name} to be equal to "
                        f"{fn(sympy.Symbol(debug_name))}, "
                        f"where {debug_name} = {expr2.xreplace(self.backed_var_to_val)}, "
                        f"but got {expr1.xreplace(self.backed_var_to_val)}"
                    )

            for phantom_symbol in equalities_inputs.phantom_symbols:
                if isinstance(phantom_symbol, sympy.Symbol):
                    # we created additional phantom symbols that are not input shape dimensions
                    symbol_to_source[phantom_symbol].extend(
                        self.var_to_sources[phantom_symbol]
                    )

        # How do we know what the value of s0 is?  Fresh variables can only be
        # bound by inputs, so there MUST be some other input which binds the
        # variable.  If there is no such input, this is an error in our
        # system.  We record where all symbols come from, to help you diagnose
        # why those symbols didn't occur.
        #
        # In fact, generally speaking it is only possible for the "outermost"
        # user of a ShapeEnv to evaluate the guards, because some inputs may
        # not be available to inner levels.  For example, Dynamo can guard on
        # tensors that never actually become graph arguments (they are
        # pruned).  In this case, only Dynamo knows about these arguments.
        def track_symint(
            source: Source, val: IntLikeType, constraint: DimConstraint = None
        ) -> None:
            log.debug(
                "track_symint %s %s %s",
                LazyString(lambda: source.name),
                val,
                constraint,
            )
            if isinstance(val, SymInt) and not is_symbolic(val):
                raise AssertionError("val must be symbolic if it is a SymInt")

            if isinstance(val, SymInt) and val.node.maybe_as_int() is not None:
                val = val.node.maybe_as_int()

            if isinstance(val, SymInt):
                s = val.node.expr
                if isinstance(s, sympy.Symbol):
                    symbol_to_source[s].append(source)
                    if constraint is not None and not isinstance(
                        constraint, RelaxedUnspecConstraint
                    ):
                        symbol_to_constraints[s].add(constraint)
                else:
                    constraint_violated = False
                    if isinstance(constraint, StrictMinMaxConstraint):
                        # try inferring the ranges of the expr s
                        sym_vrs = {
                            x: self.var_to_range.get(x, None) for x in s.free_symbols
                        }
                        if any(vr is None for vr in sym_vrs.values()):
                            # some of the free symbols in s don't have ranges
                            constraint_violated = True
                    elif isinstance(constraint, RelaxedUnspecConstraint):
                        if s.is_number:
                            i = int(s)
                            # Don't complain about 0/1 specialization, we
                            # expect to have to compile in this case anyway
                            if i not in (0, 1):
                                constraint_violated = True
                    if constraint_violated:
                        if constraint is None:
                            raise AssertionError("constraint must not be None")

                        def hint(s: sympy.Expr) -> str:
                            sexpr = py_printer.doprint(s)
                            return f"{sexpr}."

                        var_with_range = self._render_range_for_constraint_violation(
                            source, constraint
                        )
                        msg = (
                            f"Not all values of {var_with_range} are valid because "
                            f"{self._debug_name(source)} was inferred to be equal to "
                        )
                        record_constraint_violation(
                            constraint.warn_only,
                            self._debug_name(source),
                            msg,
                            hint=functools.partial(hint, s),
                        )

                input_guards.append((source, s))
            else:
                s = sympy.Integer(val)
                input_guards.append((source, s))
                constraint_violated = False
                if isinstance(constraint, StrictMinMaxConstraint):
                    if not (
                        s == constraint.vr.lower == constraint.vr.upper
                    ):  # allow static constraints
                        constraint_violated = True
                elif isinstance(constraint, RelaxedUnspecConstraint):
                    # Don't complain about 0/1 specialization, we
                    # expect to have to compile in this case anyway
                    if val not in (0, 1):
                        constraint_violated = True
                if constraint_violated:
                    if constraint is None:
                        raise AssertionError("constraint must not be None")
                    var_with_range = self._render_range_for_constraint_violation(
                        source, constraint
                    )
                    user_stack = self.specialization_stacks.get(source, None)
                    msg = (
                        f"You marked {self._debug_name(source)} as dynamic but your code "
                        f"specialized it to be a constant ({val}). If you're using mark_dynamic, "
                        f"either remove it or use maybe_mark_dynamic. If you're using Dim.DYNAMIC, "
                        f"replace it with either Dim.STATIC or Dim.AUTO."
                        + (
                            "\n\nUser stack:\n" + "".join(user_stack.format())
                            if user_stack
                            else ""
                        )
                    )
                    record_constraint_violation(
                        constraint.warn_only, self._debug_name(source), msg
                    )

        def track_symfloat(source: Source, val: FloatLikeType) -> None:
            log.debug("track_symfloat %s %s", LazyString(lambda: source.name), val)
            if isinstance(val, SymFloat) and not is_symbolic(val):
                raise AssertionError("val must be symbolic if it is a SymFloat")

            if isinstance(val, SymFloat) and val.node.maybe_as_float() is not None:
                val = val.node.maybe_as_float()

            if isinstance(val, SymFloat):
                s = val.node.expr
                if isinstance(s, sympy.Symbol):
                    symbol_to_source[s].append(source)
                input_guards.append((source, s))
            else:
                s = sympy.Float(val)
                input_guards.append((source, s))

        # pyrefly: ignore [bad-argument-type, no-matching-overload]
        for t, source, context in zip(placeholders, sources, input_contexts):
            if isinstance(source, str):
                from torch._dynamo.source import LocalSource

                source = LocalSource(source)
            if not isinstance(source, Source):
                raise AssertionError(f"Expected Source, got {type(source)}")
            if t is None:
                continue
            if isinstance(t, (SymInt, int)):
                constraint = (
                    None if context is None else getattr(context, "constraint", None)
                )
                track_symint(source, t, constraint)
                continue
            elif isinstance(t, (SymFloat, float)):
                track_symfloat(source, t)
                continue
            if not isinstance(t, Tensorlike):
                raise AssertionError(f"Expected Tensorlike, got {type(t)}")
            if is_traceable_wrapper_subclass(t):
                from torch._dynamo.source import AttrSource

                if not isinstance(context, SubclassSymbolicContext):
                    raise AssertionError(
                        f"Expected SubclassSymbolicContext, got {type(context)}"
                    )

                # For subclasses, we need to track symints on BOTH the outer
                # and inner tensors.
                # TODO: type this better
                sources_tensors_constraints: list[tuple[Source, Any, Any, Any]] = [
                    (source, t, context.constraint_sizes, context.constraint_strides)
                ]
                attrs, _ = t.__tensor_flatten__()
                for attr in attrs:
                    match getattr(t, attr):
                        case torch.Tensor() as inner_t:
                            inner_context = context.inner_contexts[attr]
                            sources_tensors_constraints.append(
                                (
                                    AttrSource(source, attr),
                                    inner_t,
                                    inner_context.constraint_sizes,  # type: ignore[attr-defined]
                                    inner_context.constraint_strides,  # type: ignore[attr-defined]
                                )
                            )
                        case OpaqueBase():
                            pass
                        case unexpected:
                            raise AssertionError(
                                f"expected Tensor or OpaqueBase, got {type(unexpected)}"
                            )
            else:
                sources_tensors_constraints = [
                    (source, t, context.constraint_sizes, context.constraint_strides)  # type: ignore[attr-defined]
                ]

            for (
                src,
                curr_t,
                constraint_size,
                constraint_stride,
            ) in sources_tensors_constraints:
                if is_sparse_any(curr_t):
                    for i, ss in enumerate(curr_t.size()):
                        property_source = TensorPropertySource(
                            src, TensorProperty.SIZE, i
                        )
                        track_symint(property_source, ss, constraint_size[i])
                else:
                    for i, ss in enumerate(curr_t.size()):
                        property_source = TensorPropertySource(
                            src, TensorProperty.SIZE, i
                        )
                        track_symint(property_source, ss, constraint_size[i])

                    for i, ss in enumerate(curr_t.stride()):
                        property_source = TensorPropertySource(
                            src, TensorProperty.STRIDE, i
                        )
                        track_symint(property_source, ss, constraint_stride[i])
                    track_symint(
                        TensorPropertySource(src, TensorProperty.STORAGE_OFFSET),
                        curr_t.storage_offset(),
                    )

        # 1. Every input must equal the final simplified symbolic expression
        #    stored on the placeholder.  Given a placeholder (s0*2, s1),
        #    if we have an input (2, 3), we must show s0*2 == 2 and s1 == 3.
        #    This does a lot of work: it covers duck sizing and equality guards.
        all_exprs: list[list[str]] = [[] for _ in langs]

        self.dim_constraints = DimConstraints(
            symbol_to_source,
            self.backed_var_to_val,
            set(symbol_to_constraints.keys()),
            self.source_name_to_debug_name,
        )

        if not _simplified:
            for source, expr in input_guards:
                srcname = source.name
                if self._translation_validation_enabled:
                    # Ignore sources that were not turned into SymInts.
                    if srcname in self.source_to_symbol:
                        self._add_target_expr(
                            sympy.Eq(self.source_to_symbol[srcname], expr)
                        )

                # Small optimization
                if (
                    isinstance(expr, sympy.Symbol)
                    and symbol_to_source.get(expr)
                    and source == symbol_to_source[expr][0]
                ):
                    continue

                # This logic excludes static values found on tensors from guarding, because
                # dynamo's check_tensor_fn does that (see guards.cpp).
                # However, for non tensor sources, we still need to guard here.
                if ignore_static and isinstance(source, TensorPropertySource):
                    if expr.is_number:
                        self.log.debug(
                            "Skipping guard %s", f"{source_ref(source)} == {expr}"
                        )
                        continue

                if is_dim(source):
                    self.dim_constraints.add_equality(source, expr)

                for exprs, printer, lang in zip(all_exprs, printers, langs):
                    res = f"{printer.print_source(source)} == {printer.doprint(expr)}"

                    if lang == "verbose_python":
                        if (s0 := self.source_to_var.get(srcname)) is not None:
                            if source != self.var_to_sources[s0][0]:
                                res = (
                                    f"{res}  # duck sizing added this equality because these "
                                    f"variables had the same size {self.backed_var_to_val[s0]} "
                                    "(to avoid this specialization, set torch.fx.experimental._config.use_duck_shape = False)"
                                )
                            elif (sloc := self.replacements_slocs.get(s0)) is not None:
                                res = f"{res}  # {sloc}"
                            else:
                                res = f"{res}  # (unknown var {s0}, please file a bug)"
                        else:
                            res = f"{res}  # (unknown source {srcname}, please file a bug)"
                    exprs.append(res)

                if (
                    isinstance(source, TensorPropertySource)
                    and source.prop is TensorProperty.SIZE
                    and equalities_inputs
                    and len(expr.free_symbols) == 1
                ):
                    symbol = next(iter(expr.free_symbols))
                    if (
                        isinstance(expr, sympy.Symbol)
                        and expr in symbol_to_constraints
                        and not equalities_inputs.is_equal(
                            source, symbol_to_source[expr][0]
                        )
                    ):
                        msg = (
                            f"The values of {self._debug_name(source)} = {source.name} and "
                            f"{self._debug_name(symbol_to_source[expr][0])} = {symbol_to_source[expr][0].name} "
                            "must always be equal."
                        )
                        record_constraint_violation(
                            equalities_inputs.warn_only, self._debug_name(source), msg
                        )

                    if (
                        not isinstance(expr, sympy.Symbol)
                        and symbol in symbol_to_constraints
                        and not equalities_inputs.is_derived(
                            source,
                            symbol_to_source[symbol][0],
                            lambda x: expr.xreplace({symbol: x}),
                        )
                    ):
                        src = symbol_to_source[symbol][0]
                        msg = (
                            f"The values of {self._debug_name(source)} = {source.name} must always be related to "
                            f"the values of {self._debug_name(src)} = {src.name} by "
                            f"{self._debug_name(source)} = {expr.xreplace({symbol: sympy.sympify(self._debug_name(src))})}."
                        )
                        record_constraint_violation(
                            equalities_inputs.warn_only, self._debug_name(source), msg
                        )

                # NB: Not necessary to report constraint violations here:
                # constraints are guaranteed to be on symbols (we've already
                # caught constants and non-atomic expressions), so we only
                # have relational constraints, but we don't support those
                # at the moment

        # 2. Every guard must evaluate to True (but remember many guards
        #    like s0 == s1*2 because trivial due to simplification)
        issued = set()

        def issue_guard(guard: ShapeGuard) -> None:
            expr = self.simplify(guard.expr)

            # Avoid re-issuing the same guard.
            if expr in issued:
                return

            issued.add(expr)

            try:
                is_trivial = False
                if any(
                    is_dim(source)
                    for s in expr.free_symbols
                    for source in symbol_to_source[s]
                ):
                    if self.dim_constraints is None:
                        raise AssertionError("dim_constraints must not be None")
                    is_trivial = self.dim_constraints.add(expr)

                for exprs, printer, lang in zip(all_exprs, printers, langs):
                    guard_expr = printer.doprint(expr)
                    if lang == "verbose_python":
                        guard_expr = f"{guard_expr}  # {guard.sloc}"
                    exprs.append(guard_expr)

                self._add_target_expr(expr)
                # A non-relational constraint on a single sizevar can violate
                # a constraint
                if not is_trivial and len(expr.free_symbols) == 1:
                    symbol = next(iter(expr.free_symbols))
                    source = symbol_to_source[symbol][0]
                    constraints = symbol_to_constraints[symbol]
                    for c in constraints:
                        if isinstance(c, StrictMinMaxConstraint):
                            var_with_range = (
                                self._render_range_for_constraint_violation(source, c)
                            )
                            msg = (
                                f"Not all values of {var_with_range} "
                                f"satisfy the generated guard {py_printer.doprint(expr)}."
                            )
                            record_constraint_violation(
                                c.warn_only, self._debug_name(source), msg
                            )
                        elif isinstance(c, RelaxedUnspecConstraint):
                            # This is fine, we allow guards here as long as it
                            # didn't constrain it to one value  (we don't
                            # actually know this; this depends on our
                            # ValueRanges reasoning capability)
                            pass
                        else:
                            raise AssertionError(f"unrecognized constraint {c}")
            except Exception:
                self.log.warning("Failing guard allocated at %s", guard.sloc)
                raise

        # First, issue all guards.
        # This removes all the checks that follow from bounds
        # We could simply emit those and also the bounds 2 <= size when necessary
        for guard in guards if guards is not None else self.guards:
            if (
                self._maybe_evaluate_static(
                    guard.expr, axioms=(), size_oblivious=guard.size_oblivious
                )
                is not None
            ):
                continue

            issue_guard(guard)

        # Because there are guards that export's constraint solver can suggest good fixes for, that we may have
        # deferred as runtime asserts, and that produce_guards() alone won't do anything with (e.g. divisiblity guards),
        # we want to send runtime asserts to export's constraint solver too. These will still stay in the graph as asserts,
        # but export's constraint solver can decide whether to do anything with them (i.e. raise an error and provide
        # suggested fixes, or decide it's out of scope and leave as a runtime assert in the graph).
        for ra in self.deferred_runtime_asserts.get(None, []):
            if self._maybe_evaluate_static(ra.expr, axioms=()) is not None:
                continue
            expr = self.simplify(ra.expr)

            self.dim_constraints.add(expr)

        # 3. Every symbol must be within its value range (this handles 0/1
        # specialization too).
        for symbol, sources in symbol_to_source.items():
            r = self.var_to_range.get(symbol)
            if r is None:
                continue
            vr_sloc = self.var_to_range_sloc[symbol]

            if not sources:
                raise AssertionError(f"sources must not be empty for symbol {symbol}")
            bounds = []
            rf = source_ref(sources[0])
            verbose_expr = ""
            if r.lower not in (-sympy.oo, -int_oo):
                if any(is_dim(source) for source in sources):
                    self.dim_constraints.add(sympy.Ge(symbol, r.lower))
                # Only print lower bound in simplified mode if it is not the
                # default
                if not _simplified or r.lower != self._default_value_range().lower:
                    bounds.append(sympy.Le(r.lower, symbol, evaluate=False))
                verbose_expr = f"{r.lower} <= {rf}  # {vr_sloc.lower}"
            if r.upper not in (sympy.oo, int_oo):
                if any(is_dim(source) for source in sources):
                    self.dim_constraints.add(sympy.Le(symbol, r.upper))
                # nontrivial upper bound is always interesting
                bounds.append(sympy.Le(symbol, r.upper, evaluate=False))
                if verbose_expr:
                    verbose_expr = f"{r.lower} <= {rf} <= {r.upper}  # {vr_sloc.lower} and {vr_sloc.upper}"
                else:
                    verbose_expr = f"{rf} <= {r.upper}  # {vr_sloc.upper}"
            if bounds:
                bound = sympy.And(*bounds, evaluate=False)

                for exprs, printer, lang in zip(all_exprs, printers, langs):
                    if lang == "verbose_python":
                        exprs.append(verbose_expr)
                    else:
                        exprs.append(printer.doprint(bound))
                # NB: verbose_exprs are done above

                # Check constraints
                constraints = symbol_to_constraints[symbol]
                for c in constraints:
                    if isinstance(c, StrictMinMaxConstraint):
                        # TODO: With int_oo, I think this condition is a noop
                        # now
                        if not (c.vr & self._default_value_range()).issubset(r):
                            source = sources[0]

                            expr = sympy.And(
                                sympy.Le(r.lower, symbol), sympy.Le(symbol, r.upper)
                            )
                            guard_expr = py_printer.doprint(expr)
                            var_with_range = (
                                self._render_range_for_constraint_violation(source, c)
                            )
                            msg = f"Not all values of {var_with_range} satisfy the generated guard {guard_expr}"
                            record_constraint_violation(
                                c.warn_only,
                                self._debug_name(source),
                                msg,
                            )
            # We NaN specialize, which means similar to 0/1 specialization we
            # should assume that the float is NOT nan.  This is load bearing
            # if you have something like an equality guard, nan will play
            # merry hell with the reasoning.
            if symbol_is_type(symbol, SymT.FLOAT):
                res = f"not math.isnan({py_printer.print_source(sources[0])})"
                for exprs, printer, lang in zip(all_exprs, printers, langs):
                    if lang == "verbose_python":
                        exprs.append(
                            f"{res}  # implicit guard for float input due to NaN specialization in the framework"
                        )
                    elif lang == "python":
                        exprs.append(res)
                    elif lang == "cpp":
                        exprs.append(f"~std::isnan({printer.print_source(sources[0])})")
                    else:
                        raise NotImplementedError(f"Unimplemented for lang: {lang}")

        # Exclusion guard for stable graph selection with automatic dynamic.
        #
        # When automatic_dynamic promotes a static dim to dynamic, the new
        # (more general) graph is inserted *before* the old (specialized) graph
        # in the guard cache.  Without an exclusion guard, inputs that exactly
        # match the old graph's static sizes would be captured by the new
        # dynamic graph instead, violating the invariant "once an input is
        # served by graph X it is always served by graph X". This condition
        # is true iff there is no branching on dynamic shapes.
        #
        # Soundness argument (cache-flip / LIFO order):
        #   Graph_new sits before Graph_old in the cache.  Graph_old accepts
        #   only inputs whose sizes match its static constraints exactly.
        #   Graph_new must therefore reject exactly that set of inputs so they
        #   fall through to Graph_old.  The excluded values are the static
        #   sizes from Graph_old, so the guard
        #       Or(Ne(s0, v0), Ne(s1, v1), ...)
        #   passes iff at least one dim differs from the old sizes — i.e. the
        #   input does NOT fully match Graph_old.  Conversely, when every dim
        #   matches the old sizes the guard fails and the input falls through
        #   to Graph_old, which is guaranteed to accept it.
        #
        # Theorem: For graphs G0, ..., Gn compiled via progressive dynamism
        # (one dim per step), each input is accepted by at most one graph.
        #
        #   Setup: G0 is all-static with shape S. Gk is created by making
        #   dim d_k dynamic, with exclusion guard d_k != S[d_k].
        #
        #   Proof by induction on n:
        #
        #   Base case (n=0): Only G0, all-static. Trivially unique.
        #
        #   Inductive step: Assume the property holds for G0, ..., G_{n-1}.
        #   We add Gn with newly-dynamic dim d_n and exclusion d_n != S[d_n].
        #
        #   For any input X that passes Gn's shape guards, exactly one of:
        #
        #   Case A — exclusion passes (X[d_n] != S[d_n]):
        #     Dim d_n is static in all G0, ..., G_{n-1} with value S[d_n],
        #     so X fails all prior graphs on that dim. Only Gn accepts X.
        #
        #   Case B — exclusion rejects (X[d_n] == S[d_n]):
        #     X matches Gn's shape guards on all other dims, and matches
        #     the static value for d_n. So X satisfies G_{n-1}'s shape
        #     guards. By the inductive hypothesis, exactly one of
        #     G0, ..., G_{n-1} accepts X. Gn rejects X.
        #
        #   Corollary: Evaluation order does not affect correctness.
        #
        # All exclusion pairs across all tensors and scalars are flattened
        # into a single list — each pair is just (symbol, excluded_int),
        # and the multi-tensor case is the same logic as multi-dim within
        # one tensor.  The combined Or rejects only when ALL pairs match
        # simultaneously, which is the exact condition for Graph_old to
        # accept.  If the current concrete values already match every
        # excluded value the guard is skipped (it would fail on creation).
        import torch._dynamo.config as dynamo_config

        if (
            dynamo_config.automatic_dynamic_exclusion_guard
            and not dynamo_config.enable_compiler_collectives
            and self.exclusion_constraints
        ):
            all_pairs = [
                (sym, val)
                for sym, val in self.exclusion_constraints
                if symbol_to_source.get(sym)
            ]
            if all_pairs and not all(
                self.backed_var_to_val.get(sym) == val for sym, val in all_pairs
            ):
                if len(all_pairs) == 1:
                    excl_expr = sympy.Ne(
                        all_pairs[0][0], all_pairs[0][1], evaluate=False
                    )
                else:
                    excl_expr = sympy.Or(
                        *[sympy.Ne(sym, val, evaluate=False) for sym, val in all_pairs]
                    )
                for exprs, printer, lang in zip(all_exprs, printers, langs):
                    guard_expr = printer.doprint(excl_expr)
                    if lang == "verbose_python":
                        guard_expr = (
                            f"{guard_expr}  # exclusion guard for automatic dynamic"
                        )
                    exprs.append(guard_expr)

        if constraint_violations:
            warn_msgs: list[str] = []
            error_msgs: list[str] = []
            debug_names = set()
            for warn_only, debug_name, msg_cb in constraint_violations:
                if warn_only:
                    str_msg = f"  {len(warn_msgs) + 1}. {msg_cb()}"
                    warn_msgs.append(str_msg)
                else:
                    str_msg = f"  - {msg_cb()}"
                    error_msgs.append(str_msg)
                    # pyrefly: ignore [bad-argument-type]
                    debug_names.add(debug_name)
            if len(error_msgs) > 0:
                debug_names_str = ", ".join(sorted(debug_names))
                err = "\n".join(error_msgs)
                raise ConstraintViolationError(
                    f"Constraints violated ({debug_names_str})! "
                    'For more information, run with TORCH_LOGS="+dynamic".\n'
                    f"{err}"
                )
            elif len(warn_msgs) > 0:
                log.debug("%s Warning only constraints violated", len(warn_msgs))

        signpost_event(
            "dynamic",
            "produce_guards",
            {
                **self.co_fields,
                **self.counter,
                "num_guards": len(all_exprs[0]),
                "free_symbols": sum(1 for v in symbol_to_source.values() if v),
                # The keys are meaningless from an aggregate perspective, so
                # don't include them.  Biggest first.
                "symbol_guard_counts": sorted(
                    self.symbol_guard_counter.values(), reverse=True
                ),
            },
        )

        if self._translation_validation_enabled:
            from torch.fx.experimental.validator import PopulateValidator

            # Add all deferred runtime assertions; these are not technically
            # handled by produce_guards but we need to put them in the target
            # set
            for ras in self.deferred_runtime_asserts.values():
                for ra in ras:
                    self._add_target_expr(ra.expr)

            # Add value range bound guards for all symbols with no trivial bounds.
            # Reason: '_maybe_evaluate_static' may eliminate guards based on the
            # refined value ranges.
            for sym, vr in self.var_to_range.items():
                if vr.lower not in (-sympy.oo, -int_oo):
                    self._add_target_expr(sympy.Le(vr.lower, sym))
                if vr.upper not in (sympy.oo, int_oo):
                    self._add_target_expr(sympy.Le(sym, vr.upper))

            # Before validating, populate the input of the validator with the
            # built FX graph.
            with fx_traceback.preserve_node_meta():
                PopulateValidator(self.graph, self.validator).run()

        # Only run translation validation when we are not passing custom guards
        if guards is None:
            self._check_translation_validate()

        helpers: list[_ShapeGuardsHelper] = []
        for exprs, printer, lang in zip(all_exprs, printers, langs):
            if lang == "cpp":
                if not isinstance(printer, _ShapeGuardCppPrinter):
                    raise AssertionError(
                        f"Expected _ShapeGuardCppPrinter, got {type(printer)}"
                    )
                helpers.append(_CppShapeGuardsHelper(exprs, printer.source_to_symbol))
            else:
                helpers.append(_ShapeGuardsHelper(exprs))
        return helpers

    def produce_guards_expression(
        self,
        placeholders: Sequence[SymInt | FakeTensor],
        *,
        guards: list[ShapeGuard] | None = None,
        ignore_static: bool = True,
    ) -> str | None:
        """
        Expected to be used with evaluate_guards_expression(). Produces the guards
        for the given placeholders and returns a string expression to be evaluated
        by evaluate_guards_expression given concrete values for the placeholders.
        """
        from torch._dynamo.source import LocalSource

        arg_names = [f"t{i}" for i in range(len(placeholders))]
        produced_guards = self.produce_guards(
            placeholders,
            [LocalSource(a) for a in arg_names],
            guards=guards,
            ignore_static=ignore_static,
        )
        if produced_guards:
            return " and ".join(produced_guards)
        return None

    def evaluate_symexpr(self, code: str) -> int | float | bool:
        """
        To be used by compile_fx to evaluate symexprs
        """
        args = {str(e): val for e, val in self.backed_var_to_val.items()}
        return eval(code, SYMPY_INTERP, args)

    def deserialize_symexpr(self, code: str) -> SymInt | SymFloat | SymBool:
        """
        To be used by compile_fx to deserialize symexprs
        """
        args = {
            str(e): SymInt(SymNode(e, self, int, int(val), fx_node=None))
            for e, val in self.backed_var_to_val.items()
        }
        return eval(code, SYMPY_INTERP, args)

    def evaluate_guards_expression(self, code: str, args: Sequence[object]) -> bool:
        """
        Expected to be used with produce_guards_expression(). Evaluates an expression
        generated by produce_guards_expression for the given concrete args.
        """
        arg_names = [f"t{i}" for i in range(len(args))]
        return eval(code, SYMPY_INTERP, {"L": dict(zip(arg_names, args))})

    def evaluate_guards_for_args(
        self,
        placeholders: Sequence[FakeTensor],
        args: Sequence[Tensor],
        *,
        ignore_static: bool = True,
    ) -> bool:
        """Generate guards for a graph's placeholder values and evaluate the guards with args"""
        code = self.produce_guards_expression(placeholders, ignore_static=ignore_static)
        if code:
            return self.evaluate_guards_expression(code, args)
        return True

    def get_pruned_guards(self, symints: Sequence[torch.SymInt]) -> list[ShapeGuard]:
        """
        Get a list of guards, but pruned so it only provides guards that
        reference symints from the passed in input
        """
        # pyrefly: ignore [bad-assignment]
        symints = {
            s.node.expr for s in symints if isinstance(s.node.expr, sympy.Symbol)
        }
        guards = [
            g for g in self.guards if all(s in symints for s in g.expr.free_symbols)
        ]
        return guards

    def bind_symbols(
        self, placeholders: Sequence[FakeTensor], args: Sequence[Tensor]
    ) -> dict[sympy.Symbol, int]:
        """
        Given a paired list of placeholders (fake tensors with
        symbolic sizes) and concrete arguments (regular tensors
        with real sizes), returns a dictionary mapping each
        symbol to its real value.  So for example, if you
        have a placeholder with size (s0, s1), binding
        (2, 4) to it will give you {s0: 2, s1: 4}.  This is
        not guaranteed to bind ALL symbols in the ShapeEnv;
        we can't bind a symbol if it doesn't occur in any placeholder,
        and symbols that already have replacements won't get bindings.

        This is a little duplicative with evaluate_guards but
        it's different enough that it seemed cleanest to make
        another copy.  This assumes the guards are already checked,
        though if it's cheap we'll check for shenanigans
        """
        bindings: dict[sympy.Symbol, int] = {}

        def bind_symint(arg: object, val: object) -> None:
            if isinstance(val, SymInt):
                if not isinstance(arg, int):
                    raise AssertionError(f"Expected int, got {type(arg)}")
                s = val.node.expr

                if isinstance(s, sympy.Symbol):
                    if s in bindings:
                        if bindings[s] != arg:
                            raise AssertionError(f"{bindings[s]} != {arg}")
                    else:
                        bindings[s] = arg
                elif isinstance(-s, sympy.Symbol):
                    if -s in bindings:
                        if bindings[-s] != -arg:
                            raise AssertionError(f"{bindings[-s]} != {-arg}")
                    else:
                        bindings[-s] = -arg

        for t, arg in zip(placeholders, args):
            if t is None:
                continue
            if isinstance(t, SymInt):
                bind_symint(arg, t)
                continue
            if not isinstance(t, torch.Tensor):
                raise AssertionError(f"Expected Tensor, got {type(t)}")
            for i, s in enumerate(t.size()):
                bind_symint(arg.size(i), s)
            for i, s in enumerate(t.stride()):
                bind_symint(arg.stride(i), s)
            bind_symint(arg.storage_offset(), t.storage_offset())

        return bindings

    def get_nontrivial_guards(self) -> list[SympyBoolean]:
        """Returns a list of guard expressions that aren't statically known (i.e. not trivial)"""
        return [
            self.simplify(guard.expr)
            for guard in self.guards
            if self._maybe_evaluate_static(
                guard.expr, axioms=(), size_oblivious=guard.size_oblivious
            )
            is None
        ]

    def format_guards(self, verbose: bool = False) -> str:
        """Format this shape env's guard expressions with optional traceback info if verbose"""

        return "\n".join(
            f" - {guard.expr}{' ' + str(guard.sloc) if verbose else ''}"
            for guard in self.guards
        )

    def bound_sympy(
        self, expr: sympy.Expr, size_oblivious: bool = False
    ) -> ValueRanges:
        """Given a sympy expression, computes a ValueRanges bound for what values it can be"""
        # TODO: maybe it's guaranteed x in is var_to_range?
        var_to_range = {x: self.var_to_range.get(x, None) for x in expr.free_symbols}
        if size_oblivious:
            # Clamp values of size-like variables
            # NB: discarding the old upper bound in intentional, per
            # https://github.com/pytorch/pytorch/pull/123675
            for x in self.size_like & var_to_range.keys():
                if var_to_range[x] is not None:
                    # NB: do NOT set upper to 2 ** 48, we're using this solely
                    # to determine if we can do size-like replacement, the
                    # upper bound is irrelevant here
                    var_to_range[x] = ValueRanges(2, int_oo)
        return bound_sympy(expr, var_to_range)  # type: ignore[arg-type]

    @_lru_cache
    def get_axioms(
        self,
        symbols: tuple[sympy.Symbol] | None = None,
        compute_hint: bool = False,
    ) -> tuple[SympyBoolean, ...]:
        """
        Given the symbols in an expression, it returns all the runtime asserts that have those symbols
        concatenated with all the guards.
        If symbols is None, it returns all the runtime asserts (and all the guards)
        """
        if symbols is None:
            runtime_asserts = (
                r.expr for rs in self.deferred_runtime_asserts.values() for r in rs
            )
        else:
            runtime_asserts = (
                r.expr
                for s in symbols
                if s not in self.backed_var_to_val
                for r in self.deferred_runtime_asserts.get(s, ())
            )
        guards: Iterator[SympyBoolean] = (g.expr for g in self.guards)
        axioms: Iterator[SympyBoolean] = itertools.chain(guards, runtime_asserts)
        if compute_hint:
            axioms = (
                canonicalize_bool_expr(a.xreplace(self.backed_var_to_val))
                for a in axioms
            )
        return tuple(dict.fromkeys(axioms).keys())

    @lru_cache(None)
    def get_implications(
        self, e: SympyBoolean
    ) -> tuple[tuple[SympyBoolean, sympy.logic.boolalg.BooleanAtom], ...]:
        """Given a expression, it returns a list of predicates that follow from it"""
        equiv: dict[SympyBoolean, sympy.logic.boolalg.BooleanAtom] = {}

        def add_expr(expr: SympyBoolean) -> None:
            expr = canonicalize_bool_expr(expr)
            if isinstance(expr, (sympy.Eq, sympy.Ne)):
                # No need to canonicalize
                # TODO We could further canonicalize Eq ordering the lhs and rhs somehow
                # With this, we could remove the need for the commutativity part
                opposite = sympy.Eq if isinstance(expr, sympy.Ne) else sympy.Ne
                # Commutativity of == and !=
                equiv[type(expr)(expr.lhs, expr.rhs, evaluate=False)] = sympy.true
                equiv[type(expr)(expr.rhs, expr.lhs, evaluate=False)] = sympy.true
                equiv[opposite(expr.lhs, expr.rhs, evaluate=False)] = sympy.false
                equiv[opposite(expr.rhs, expr.lhs, evaluate=False)] = sympy.false
            else:
                # Expr and negation
                equiv[expr] = sympy.true
                # we do not pass evaluate=False like others on purpose here!
                # we want not(a<b) to be a>=b and not ~(a<b).
                equiv[canonicalize_bool_expr(sympy.Not(expr))] = sympy.false

        add_expr(e)
        # Other relational expressions this expression implies
        if isinstance(e, sympy.Eq):
            add_expr(sympy.Le(e.lhs, e.rhs, evaluate=False))
            add_expr(sympy.Ge(e.lhs, e.rhs, evaluate=False))
        elif isinstance(e, sympy.Lt):
            add_expr(sympy.Le(e.lhs, e.rhs, evaluate=False))
            add_expr(sympy.Ne(e.lhs, e.rhs, evaluate=False))
            if e.lhs.is_integer and e.rhs.is_integer:  # type: ignore[attr-defined]
                add_expr(sympy.Le(e.lhs, e.rhs - 1, evaluate=False))
        elif isinstance(e, sympy.Le):
            add_expr(sympy.Lt(e.lhs, e.rhs + 1, evaluate=False))

        return tuple(equiv.items())

    def _is_nonneg_term(self, term: sympy.Expr) -> bool:
        """Check if a single term is non-negative (symbol with non-neg range or non-neg constant)."""
        if term.is_Symbol:
            vr = self.var_to_range.get(term)
            return vr is not None and vr.lower >= 0
        if term.is_number:
            return term >= 0
        return False

    def _is_nonneg_sum(self, expr: sympy.Expr) -> bool:
        """
        Check if expr is a sum of non-negative terms (Add of symbols with non-neg range
        and non-negative constants). Returns True only for simple Add expressions.
        """
        if not isinstance(expr, sympy.Add):
            return self._is_nonneg_term(expr)

        # Check each arg in the Add
        for arg in expr.args:
            if not self._is_nonneg_term(arg):
                return False

        return True

    def _maybe_fast_eval_comparison(self, expr: sympy.Basic) -> sympy.Basic | None:
        """
        Fast path for trivial comparisons: sum of non-negative terms >= 0.
        Returns sympy.true if pattern matches, None otherwise.
        """
        if len(expr.args) != 2:
            return None

        lhs, rhs = expr.args

        # Handle: sum >= 0 (Ge) or 0 <= sum (Le)
        if isinstance(expr, sympy.Ge) and rhs == 0:
            sum_expr = lhs
        elif isinstance(expr, sympy.Le) and lhs == 0:
            sum_expr = rhs
        else:
            return None

        if self._is_nonneg_sum(sum_expr):
            return sympy.true

        return None

    def _maybe_evaluate_range_only(
        self,
        expr: sympy.Basic,
        fallback: sympy.Basic | None = None,
    ) -> sympy.Basic | None:
        """
        Lightweight range-based evaluation using only bound_sympy (value range
        analysis), without expensive simplification, axiom matching, or symbol
        reallocation.

        Returns the resolved value if range analysis determines it, otherwise
        returns fallback.
        """
        var_ranges = {x: self.var_to_range.get(x) for x in expr.free_symbols}
        out = bound_sympy(expr, var_ranges)  # type: ignore[arg-type]
        if out.is_singleton():
            return out.lower
        return fallback

    @_lru_cache
    def _maybe_evaluate_static(
        self,
        expr: sympy.Basic,
        *,
        unbacked_only: bool = False,
        compute_hint: bool = False,
        size_oblivious: bool = False,
        axioms: tuple[SympyBoolean] | None = None,
        var_to_range: tuple[tuple[sympy.Symbol, ValueRanges]] | None = None,
    ) -> sympy.Basic | None:
        """
        Tries to evaluate expr without introducing guards

        If unbacked_only == True, then we only do substitutions on
        unbacked SymInts (leaving regular hinted integers alone).  This could
        result in an expression that still contains backed SymInts, which you
        could then potentially guard on.

        Use compute_hint == True if you are trying to compute a non-binding
        hint for the particular hint values of backed and unbacked SymInts,
        e.g., if s0 happens to be 3 this run, compute_hint will substitute s0 with 3.
        """

        # axioms with compute hint NYE
        if compute_hint and axioms:
            raise AssertionError("compute_hint and axioms cannot both be set")
        expr = self.simplify(expr, size_oblivious)

        if compute_hint:
            expr = expr.xreplace(self.backed_var_to_val).xreplace(
                self.real_tensor_prop_unbacked_vals
            )

        expr = canonicalize_bool_expr(expr)

        def resimplify_floor_div(axioms: dict[sympy.Expr, sympy.Expr]) -> None:
            if not self._resimplify_floor_div_axioms:
                return
            self._resimplify_floor_div_axioms = False
            new_items = {}
            for k, v in list(axioms.items()):
                # A FloorDiv in implications could have became CleanDiv at this point, due to new facts
                # to the shapeEnv. This handles such issue but its not ideal. This is the only expression
                # simplification that depends on the global state of shape env.
                # TODO try to get rid of CleanDiv since it breaks the invariant that's simplifications of sympy
                # expressions only depend on the expression itself.
                if k.has(FloorDiv):
                    new_items.update({self.simplify(k): v})
            axioms.update(new_items)

        # Pattern matching
        if axioms is None:
            resimplify_floor_div(self.axioms)
            subst = self.axioms
        else:
            subst = {}
            for e in axioms:
                if e.free_symbols.issubset(expr.free_symbols):
                    subst.update(dict(self.get_implications(self.simplify(e))))

            resimplify_floor_div(subst)

        expr = expr.xreplace(subst)
        # TODO: compute hint might have gotten broken here

        fs = expr.free_symbols

        if not fs and (expr.is_number or expr.is_Boolean):
            return expr

        if var_to_range is None:
            var_ranges = self.var_to_range
        else:
            var_ranges = dict(var_to_range)

        symbol_info = tuple(
            _SymbolInfo(
                s,
                var_ranges.get(s),
                self.backed_var_to_val.get(s),
                s in self.size_like,
            )
            for s in sorted(fs, key=str)  # TODO: speed up sort?
        )

        r = _maybe_evaluate_static_worker(
            expr, symbol_info, unbacked_only, size_oblivious
        )
        return r

    @_lru_cache
    def replace(self, expr: _SympyT) -> _SympyT:
        """
        Apply symbol replacements to any symbols in the given expression.

        IMPORTANT: The output of this method MUST depend only on
        self.replacements and the input expr. Do not add dependencies on other
        mutable state. SymNode.expr uses _replacements_version_counter (which
        tracks only replacement changes) to cache calls to this method, so
        depending on other state would cause stale cache results.
        """
        replacements = {}
        # pyrefly: ignore [missing-attribute]
        for s in expr.free_symbols:
            r = self._find(s)

            # Micro-optimization: only do replacements if r and s are different
            # Otherwise, xreplace is not a no-op and will trigger expensive
            # assumption queries if expr has a relational node.
            if not r.is_Symbol or r != s:
                replacements[s] = r
        if replacements:
            # pyrefly: ignore [missing-attribute]
            return safe_expand(expr.xreplace(replacements))
        else:
            return expr

    @_lru_cache
    def _update_divisible(self) -> None:
        new_divisible = set()
        for k in self.divisible:
            res = self.replace(k)
            if not res.is_number:
                new_divisible.add(k)

        self.divisible = new_divisible
        self._update_version_counter()

    @_lru_cache
    def simplify(self, expr: _SympyT, size_oblivious: bool = False) -> _SympyT:
        """Use known constraints and replacements to simplify the given expr"""
        expr = safe_expand(expr)
        expr = self.replace(expr)

        # Simplify max(0/1, x) to x when x >= 0/1. max(1, x) is a commonly introduced
        # expression when creating contiguous strides.
        if not size_oblivious:
            min_max_replacements = {}
            for atom in expr.atoms(Max):  # type: ignore[has-type]
                if len(atom.args) > 2:
                    continue
                a, b = atom.args
                if b == 1 or b == 0:
                    a, b = b, a

                if a == 1 and self._maybe_evaluate_static(sympy.Ge(b, 1)):
                    min_max_replacements[atom] = b
                if a == 0 and self._maybe_evaluate_static(sympy.Ge(b, 0)):
                    min_max_replacements[atom] = b
            if min_max_replacements:
                expr = expr.xreplace(min_max_replacements)

        if expr.has(TruncToInt):
            trunc_replacements = {}
            for atom in expr.atoms(TruncToInt):
                if isinstance(atom.args[0], IntTrueDiv):
                    base, divisor = atom.args[0].args
                    if Mod(base, divisor) == 0:
                        trunc_replacements[atom] = CleanDiv(base, divisor)
                    else:
                        # TruncToInt(IntTrueDiv(a,b)) == FloorDiv(a, b)
                        trunc_replacements[atom] = FloorDiv(base, divisor)
            if trunc_replacements:
                expr = expr.xreplace(trunc_replacements)

        # TODO it would seem that this pass is not necessary given the
        # below replacement of // with /, but for nested FloorDivs
        # the non-recursive replacement doesn't work, and
        # recursive makes it hard to look up divisibility,
        # because existing divisibility info has FloorDiv in it, not /
        # for now just do a separate pass to catch common nested case
        if expr.has(FloorDiv):
            self._update_divisible()
            div_replacements = {}
            for atom in expr.atoms(FloorDiv):
                base, divisor = atom.args
                if isinstance(divisor, FloorDiv):
                    base1, divisor1 = divisor.args
                    if (
                        self.replace(Mod(base, divisor)) in self.divisible
                        and base == base1
                        and self.replace(Mod(base1, divisor1)) in self.divisible
                    ):
                        div_replacements[atom] = divisor1
            if div_replacements:
                expr = expr.xreplace(div_replacements)
                expr = safe_expand(expr)
        if expr.has(FloorDiv):
            div_replacements = {}
            pows = expr.atoms(sympy.Pow)
            rationals = expr.atoms(sympy.Rational).difference(expr.atoms(sympy.Integer))
            for fd in expr.atoms(FloorDiv):
                base, divisor = fd.args
                if self.replace(Mod(base, divisor)) in self.divisible:
                    div_replacements[fd] = CleanDiv(base, divisor)
            if div_replacements:
                new_expr = expr.xreplace(div_replacements)
                new_expr = safe_expand(new_expr)
                new_pows = new_expr.atoms(sympy.Pow)
                new_rationals = new_expr.atoms(sympy.Rational).difference(
                    new_expr.atoms(sympy.Integer)
                )
                # divisions simplified away
                if new_pows.issubset(pows) and new_rationals.issubset(rationals):
                    expr = new_expr
        return expr

    # TODO: overload for allow_none literal
    @deprecated(
        "use guarding_hint_or_throw or optimization_hint instead",
        category=FutureWarning,
    )
    @lru_cache(256)
    def size_hint(
        self, expr: sympy.Basic, *, allow_none: bool = False
    ) -> sympy.Basic | None:
        """
        Gets a size hint for a given expression from the underlying shapes we had.
        Does not introduce a guard, so only use this when you can guarantee that
        your code is still valid for arbitrary shapes (such as optimization decisions)
        """
        result_expr = safe_expand(expr).xreplace(self.backed_var_to_val)
        if not result_expr.is_number:
            from torch.utils._sympy.singleton_int import SingletonInt

            if isinstance(result_expr, SingletonInt):
                return None
            r = self._maybe_evaluate_static(result_expr, compute_hint=True)
            if r is not None:
                return r
            if allow_none:
                return None

            if self.real_tensor_prop_unbacked_vals:
                unsound_expr = result_expr.xreplace(self.real_tensor_prop_unbacked_vals)
                if not unsound_expr.free_symbols:
                    log.warning(
                        "propagate_real_tensors size_hint(%s) -> %s", expr, unsound_expr
                    )
                    trace_structured(
                        "propagate_real_tensors",
                        metadata_fn=lambda: {
                            "expr": repr(expr),
                            "result": repr(unsound_expr),
                            "stack": structured.from_traceback(
                                CapturedTraceback.extract(skip=1).summary()
                            ),
                        },
                    )
                    self.guard_or_defer_runtime_assert(
                        sympy.Eq(result_expr, unsound_expr),
                        f"propagate_real_tensors: {result_expr} == {unsound_expr}",
                    )
                    return unsound_expr

            raise self._make_data_dependent_error(result_expr, expr)
        return result_expr

    @lru_cache(256)
    def guarding_hint_or_throw(self, expr: sympy.Expr | int) -> int | bool:
        """
        Return a concrete hint for an expression.

        Returns Python bool (True/False) for boolean expressions (e.g. Eq, Ne),
        and Python int for integer expressions.
        """
        return _guarding_hint_or_throw_base(self, expr, {})

    @lru_cache(256)
    def has_guarding_hint(self, expr: sympy.Expr) -> bool:
        try:
            self.guarding_hint_or_throw(expr)
        except GuardOnDataDependentSymNode:
            return False
        return True

    def optimization_hint(
        self, expr: sympy.Expr | int, fallback: int | None = None
    ) -> int:
        """
        Return a concrete integer hint for an expression.

        This function should be used for non-guarding based optimizations. If you
        want a hint that you can guard on, use the guarding_hint API instead.

        This function will hint unbacked symbols using user provided optimization
        hints. If not provided, fallback will be used along with some heuristics
        that try to maximize consistency with the shape environment.

        Special cases:

        - Complex numbers (containing sympy.I): raises an error since tensor
          dimensions cannot be complex.
        - Infinity (int_oo, sympy.oo): returns sys.maxsize.
        - NaN (sympy.nan): returns the fallback value.
        """
        return _optimization_hint_base(
            self, expr, precomputed_replacements={}, fallback=fallback
        )

    def _make_data_dependent_error(
        self,
        expr: sympy.Basic,
        unhinted_expr: sympy.Basic,
        *,
        expr_sym_node_id: int | None = None,
    ) -> GuardOnDataDependentSymNode:
        # TODO: in a Dynamo context, having user code, and having the
        # name of the local, will be much better
        size_like_symbols = []
        for s in expr.free_symbols:
            stacktrace = "".join(self.var_to_stack[s].format())
            self.log.debug(
                "Data dependent variable '%s' allocated at:\n%s", s, stacktrace
            )
            if s in self.size_like:
                size_like_symbols.append(s)
        size_oblivious_result_msg = ""
        sloc, maybe_extra_debug = self._get_stack_summary(True)
        if expr.is_integer:  # type: ignore[attr-defined]
            desc = (
                "Could not extract specialized integer from data-dependent expression"
            )
        else:
            desc = "Could not guard on data-dependent expression"
            size_oblivious_result_msg = (
                "consider using data-dependent friendly APIs such as "
                "guard_or_false, guard_or_true and statically_known_true."
            )

        msg = (
            f"{desc} {expr} (unhinted: {unhinted_expr}).  "
            f"(Size-like symbols: {', '.join(map(str, size_like_symbols)) or 'none'})\n\n"
            f"{size_oblivious_result_msg}\n"
            f"Caused by: {sloc}\n"
            'For more information, run with TORCH_LOGS="dynamic"\n'
            "For extended logs when we create symbols, also add "
            f'TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="{",".join(map(str, expr.free_symbols))}"\n'
            "If you suspect the guard was triggered from C++, add TORCHDYNAMO_EXTENDED_DEBUG_CPP=1\n"
            "For more debugging help, see "
            "https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit?usp=sharing\n"
            + maybe_extra_debug
            # TODO: Help text about how to use our runtime tests to fix this
            # problem
        )

        dtrace_structured(
            "guard_on_data_dependent_error",
            metadata_fn=lambda: {
                "expr": repr(expr),
                "unhinted_expr": repr(unhinted_expr),
                "expr_id": self._expr_sym_node_id,
                "stack": structured.from_traceback(
                    CapturedTraceback.extract(skip=1).summary()
                ),
            },
        )
        return GuardOnDataDependentSymNode(expr, msg)

    def _update_var_to_range(
        self,
        symbol: sympy.Symbol,
        vr: ValueRanges,
        vr_sloc: ValueRangesSLoc | None = None,
        *,
        is_constraint: bool = False,
    ) -> None:
        lower, upper = vr.lower, vr.upper

        # If we have a size-like unbacked SymInt, refuse to refine the range to be
        # less than two.  This is because when we intersect this range
        # with [2, inf] for size oblivious tests, the range would be
        # unsatisfiable.  In other words, once you have a size-like
        # unbacked SymInt, we can never learn that it is exactly zero or one,
        # because we would now give inconsistent results for all size
        # oblivous tests!
        if upper < 2 and symbol in self.size_like:
            vr = ValueRanges(lower, 2)

        # Updates the range and the guards corresponding to each bound of the symbol.
        if symbol not in self.var_to_range:
            self.log.debug("_update_var_to_range %s = %s (new)", symbol, vr)
            self.var_to_range[symbol] = vr
            if vr_sloc is None:
                sloc = self._get_sloc()
                vr_sloc = ValueRangesSLoc(sloc, sloc)
            self.var_to_range_sloc[symbol] = vr_sloc
        else:
            old = self.var_to_range[symbol]
            new = old & vr
            if new != old:
                if vr_sloc is None:
                    sloc = self._get_sloc()
                    vr_sloc = ValueRangesSLoc(sloc, sloc)
                if new.lower != old.lower:
                    self.var_to_range_sloc[symbol].lower = vr_sloc.lower
                if new.upper != old.upper:
                    self.var_to_range_sloc[symbol].upper = vr_sloc.upper
                self.var_to_range[symbol] = new
                self.log.debug("_update_var_to_range %s = %s (update)", symbol, new)

        if (v := self.backed_var_to_val.get(symbol)) is not None:
            r = self.var_to_range[symbol]
            if v not in r:
                # For constraint failure, delay this for later
                # TODO: Rework all of this, the constraint logic is very
                # duplicative with regular reasoning
                if not is_constraint:
                    if v not in r:
                        raise AssertionError(f"{v} not in {r}")

    def _set_replacement(self, a: sympy.Symbol, tgt: sympy.Expr, msg: str) -> None:
        """
        Adds or updates a replacement for a symbol.
        Use this instead of `self.replacements[a] = tgt`.
        """

        if tgt == self.replacements.get(a, None):
            return

        if a in tgt.free_symbols:
            return

        # Precondition: a == tgt
        if not isinstance(a, sympy.Symbol):
            raise AssertionError(f"Expected sympy.Symbol, got {type(a)}")

        if (
            self.prefer_deferred_runtime_asserts_over_guards
            and not _is_supported_equivalence(tgt)
        ):
            return  # continuing leads to placeholder shapes having complex expressions that we can't resolve

        # Handles nested tensor symbolic variables which don't have
        # var_to_range bounds
        tgt_bound = None
        if a in self.var_to_range:
            src_bound = self.var_to_range[a]

            # First, refine the value range of a based on the computed value range
            # of tgt.  This is always OK to do, even if we decide not to do the
            # substitution in the end.  This might be a no-op, if a already has
            # a tighter bound
            tgt_bound = self.bound_sympy(tgt)
            self._update_var_to_range(a, tgt_bound)

            # Next, check if we can update the range of free symbols in tgt
            # based on the range in a. But only do it if:
            #  - the source bound non-trivially improves over what we get out of
            #    the existing bounds.
            #  - the replacement is univariate and we can invert the tgt expression
            if not tgt_bound.issubset(src_bound) and len(tgt.free_symbols) == 1:
                b = next(iter(tgt.free_symbols))
                # Try to invert the equality
                r = try_solve(sympy.Eq(a, tgt), b, floordiv_inequality=False)
                if r is not None:
                    self.log.debug(
                        "set_replacement: solve for %s in %s == %s gives %s",
                        b,
                        a,
                        tgt,
                        r,
                    )
                    # The solution here can be non-integral, for example, if
                    # we have s0 = 2*s1, then s1 = s0/2.  What we would like
                    # to do is calculated the bounds in arbitrary precision,
                    # and then requantize the bound to integers when we are
                    # done.
                    rat_b_bound = self.bound_sympy(r[1])
                    b_bound = ValueRanges(
                        CeilToInt(rat_b_bound.lower), FloorToInt(rat_b_bound.upper)
                    )
                    self._update_var_to_range(b, b_bound, self.var_to_range_sloc[a])
                    tgt_bound = self.bound_sympy(tgt)
                    if not tgt_bound.issubset(src_bound):
                        raise AssertionError(
                            f"{tgt_bound=} not a subset of {src_bound=}"
                        )

            # TODO: Should we propagate size-like-ness?
            #
            # Pros: if u0 is size-like, intuitively u0 == u1 should cause u1
            # to become size-like.
            #
            # Cons: if u0 is size-like, what about u0 - 1 == u1?  You CAN'T
            # propagate in this case, because what if u0 == 0, then u1 is negative
            # and clearly isn't a size.  So, at minimum, any f(x) whose value
            # range isn't [0, inf] given x in [0, inf] cannot propagate
            # size-like-ness.  But there are many situations where you could
            # imagine u1 is going to be size-like and actually you just didn't
            # have a refined enough value range on u0.  Since even innocuous
            # looking arithmetic operations can destroy size-like-ness, it's
            # best to not propagate it at all and force the user to annotate it
            # as necessary.
            #
            # Compromise: we preserve size-like-ness only for exact equality
            # and nothing else.
            if a in self.size_like and isinstance(tgt, sympy.Symbol):
                self.size_like.add(tgt)
            elif isinstance(tgt, sympy.Symbol) and tgt in self.size_like:
                self.size_like.add(a)

            # Now, decide if we will do the substitution.
            #
            #  - If the source has a non-trivial range, only substitute if
            #    we preserve this range.  Note that we may have propagated
            #    the src_range to free variables in tgt when tgt is univariate
            #    and we could find an inverse, which helps us achieve this.
            #    This ensures we never "forget" about user defined ranges,
            #    even if they end up being defined on composite formulas
            #    like s0 + s1.
            #
            #  - If the variable is unbacked, only substitute if the substitution
            #    would preserve the bounds also under size-like-ness conditions.

            if not tgt_bound.issubset(src_bound):
                self.log.debug(
                    "skipped set_replacement %s = %s (%s) [%s not subset of %s]",
                    a,
                    tgt,
                    msg,
                    tgt_bound,
                    src_bound,
                )
                return
            elif a in self.size_like:
                tgt_bound_so = self.bound_sympy(tgt, size_oblivious=True)
                src_bound_so = self.bound_sympy(a, size_oblivious=True)
                if not tgt_bound_so.issubset(src_bound_so):
                    self.log.debug(
                        "skipped set_replacement %s = %s (%s) "
                        "[%s not subset of %s (size-oblivious conditions)]",
                        a,
                        tgt,
                        msg,
                        tgt_bound_so,
                        src_bound_so,
                    )
                    return

        if isinstance(tgt, (sympy.Integer, sympy.Float)):
            # specializing to a constant, which is likely unexpected (unless
            # you specified dynamic=True)

            user_tb = TracingContext.extract_stack()
            trace_structured(
                "symbolic_shape_specialization",
                metadata_fn=lambda: {
                    "symbol": repr(a),
                    "sources": [s.name for s in self.var_to_sources.get(a, [])],
                    "value": repr(tgt),
                    "reason": msg,
                    "stack": structured.from_traceback(
                        CapturedTraceback.extract(skip=1).summary()
                    ),
                    "user_stack": (
                        structured.from_traceback(user_tb) if user_tb else None
                    ),
                },
            )

            for source in self.var_to_sources.get(a, []):
                if user_tb:
                    self.specialization_stacks[source] = user_tb

            if config.print_specializations:
                self.log.warning(
                    "Specializing %s to %s", self.var_to_sources[a][0].name, tgt
                )
                self.log.debug("SPECIALIZATION", stack_info=True)
        log.info("set_replacement %s = %s (%s) %s", a, tgt, msg, tgt_bound)
        self.replacements[a] = tgt
        # NB: the replacement may get refined, but the user will find the
        # FIRST one most useful (TODO: Maybe we could consider tracking all of
        # them)
        if a not in self.replacements_slocs:
            self.replacements_slocs[a] = self._get_sloc()
        self._replacements_version_counter += 1
        self._update_version_counter()

        # When specializing 'a == tgt', the equality should be also conveyed to
        # Z3, in case an expression uses 'a'.
        self._add_target_expr(sympy.Eq(a, tgt, evaluate=False))

    def _add_divisible(self, expr: sympy.Expr) -> None:
        self.divisible.add(expr)
        self._update_version_counter()

    @_lru_cache
    @record_shapeenv_event()
    def _find(self, a: sympy.Symbol) -> sympy.Expr:
        """
        Implements a DSU-like algorithm to find the variable that represents a
        Also handles transitive non-identity replacements.

        a: b + c
        c: d

        IMPORTANT: The output of this method MUST depend only on
        self.replacements and the input symbol. Do not add dependencies on other
        mutable state. SymNode.expr uses _replacements_version_counter (which
        tracks only replacement changes) to cache calls to replace() (and
        transitively this method), so depending on other state would cause
        stale cache results. (Note: _set_replacement,  may read other fields
        like var_to_range, but those are side effects that do not affect the
        returned value.)
        """
        if a not in self.replacements:
            return a
        res = self.replacements[a]
        cur_replace = {s: self._find(s) for s in res.free_symbols}
        replaced, changed = self.replacements[a]._xreplace(cur_replace)
        if changed:
            self._set_replacement(a, replaced, "find")
        return self.replacements[a]

    @lru_cache(256)
    def _maybe_guard_rel(self, expr: sympy.Expr) -> None:
        """
        The relational guard is guarded to be true.  Use this information to
        simplify shapes (i.e. a == b or a % 5 == 0)
        """
        if isinstance(expr, sympy.And):
            for arg in expr.args:
                self._maybe_guard_rel(arg)
            return
        elif not isinstance(expr, sympy.Rel):
            return

        # A good example of what goes wrong if you don't do this is
        # python test/functorch/test_aotdispatch.py -k
        # test_aot_autograd_symbolic_module_exhaustive_nn_LazyConv3d_cpu_float32
        if isinstance(expr, sympy.Ne):
            return

        free = list(expr.free_symbols)

        if len(free) == 0:
            raise AssertionError(
                f"The expression should not be static by this point: {expr}"
            )
        # In case of really gnarly expression, we don't blow up
        if len(free) > 5:
            return

        # Prioritize unbacked symints for solving by ordering them last.
        # Prefer to simplify out lexicographically higher symbols (i.e. simplify out s4 over s3).
        #   (NB: this unfortunately isn't strictly equivalent to simplifying out newer symbols)
        # Prefer to simplify out symbols with ephemeral sources.
        def _smart_symbol_sort(x: sympy.Symbol) -> tuple[int, int, str]:
            has_only_ephemeral_sources = x in self.var_to_sources and all(
                s.is_ephemeral() for s in self.var_to_sources[x]
            )

            hint = self.backed_var_to_val.get(x)
            if hint is None or isinstance(hint, SingletonInt):
                # NB: size_hint is int, not sympy.Expr, do not use int_oo here.
                # SingletonInt is used to represent jagged/nested tensor dimensions
                # (e.g. the irregular ragged dimension). It cannot be converted to
                # int, so we treat it the same as an unknown size. This matches the
                # behavior of size_hint(), which returns None for SingletonInt.
                size = sys.maxsize
            elif symbol_is_type(x, SymT.SIZE):
                size = int(hint)
            else:
                size = sys.maxsize
            name = x.name
            # 1 puts ephemeral sourced symbols first when sorting in reverse
            return (1 if has_only_ephemeral_sources else 0, size, name)

        free = sorted(free, key=_smart_symbol_sort, reverse=True)  # type: ignore[attr-defined]
        lhs = expr.lhs
        rhs = expr.rhs

        self._refine_ranges(expr)

        # The rest of this stuff is for equality only
        if not isinstance(expr, sympy.Eq):
            return

        if not expr.has(Mod):
            try:
                floor_div_atoms = lhs.atoms(FloorDiv).union(rhs.atoms(FloorDiv))
                if len(floor_div_atoms) > 0 and any(
                    a.divisor != 1 for a in floor_div_atoms
                ):
                    raise NotImplementedError

                # Never replace unbacked symbols with other unbacked symbols that are
                # not function arguments. (ex:mark_unbacked symbols are fine to replace
                # other unbacked, but not those coming from .item() calls).

                # This is error prone because you can cause references to
                # unbacked symbols to time travel backwards.  E.g.,
                #
                # u1 = x.item()
                # ... use of u1 ...
                # u2 = y.item()
                # u3 = z.item()
                # torch._check(u1 == u2 + u3)
                #
                # If you replace u1 with u2 + u3, then the use of u1 now
                # references u2 and u3 prior to them actually being bound at
                # runtime.  It's pretty inconvenient to setup control
                # dependencies for substitutions, so ban it entirely.
                def trivial_solve(lhs: sympy.Expr, rhs: sympy.Expr) -> bool:
                    if isinstance(lhs, sympy.Symbol):
                        if free_unbacked_symbols(
                            lhs
                        ) and not _free_non_source_unbacked_symbols(
                            rhs, self.unbacked_inputs
                        ):
                            return True
                        if symbol_is_type(lhs, SymT.FLOAT):
                            return True
                        # TODO: Maybe trivial solutions for int should also be
                        # done?
                    return False

                # short-circuit when no solving is needed
                if trivial_solve(lhs, rhs):
                    self._set_replacement(lhs, self._find(rhs), "trivial_lhs")
                elif trivial_solve(rhs, lhs):
                    self._set_replacement(rhs, self._find(lhs), "trivial_rhs")
                else:
                    r = try_solve(expr, free[0], floordiv_inequality=False)
                    if r is not None and all(
                        t.is_integer for t in sympy.preorder_traversal(r[1])
                    ):
                        new_var = self._find(r[1])
                        ok = len(free_unbacked_symbols(new_var)) == 0
                        if ok:
                            self._set_replacement(free[0], new_var, "solve")

            except NotImplementedError:
                pass
        else:
            # expression has mod.
            mod_expr = next(iter(expr.atoms(Mod)))
            try:
                r = try_solve(expr, mod_expr, floordiv_inequality=False)
                if r is not None and r[1] == 0:
                    self._add_divisible(mod_expr)
            except NotImplementedError:
                pass
        return

    # See: Note - On 0/1 specialization
    def _default_value_range(
        self, do_not_specialize_zero_one: bool = False
    ) -> ValueRanges:
        lower = 0 if (do_not_specialize_zero_one or not self.specialize_zero_one) else 2
        return ValueRanges(lower, int_oo)

    def _default_unspecified_value_range(self) -> ValueRanges:
        return ValueRanges.unknown_int()

    @_lru_cache
    def _simplify_floor_div(self, expr: sympy.Expr) -> sympy.Expr:
        floor_divs = tuple(expr.atoms(FloorDiv))
        # we expect floor_divs to be exact,
        # and thus add the guards for the exact floordivs,
        # even if tracing doesn't require them otherwise
        for fd in reversed(floor_divs):
            base, divisor = fd.args
            mod_expr = Mod(base, divisor)
            eq_expr = sympy.Eq(mod_expr, 0)
            # add necessary mod guards
            self.evaluate_expr(eq_expr)
        return self.simplify(expr)

    # We're about to add a guard/runtime assert, check if the ShapeEnv is frozen
    # and if so issue a warning (or raise if error_on_new_guards is set)
    def _check_frozen(self, expr: sympy.Basic, concrete_val: sympy.Basic) -> None:
        if self._error_on_new_guards:
            raise _ShapeEnvGuardError(
                f"Guard attempted while ShapeEnv guards are frozen: {expr} == {concrete_val}"
            )
        if self.frozen:
            self.counter["ignored_backward_guard"] += 1
            signpost_event(
                "dynamic",
                "evaluate_expr_frozen",
                {
                    **self.co_fields,
                    "ignored_guard": f"{expr} == {concrete_val}",
                    # no version = original state (this signpost is expected)
                    # version 2 = dynamic backwards is eagerly compiled
                    "version": 2,
                },
            )
            log.info(
                "Ignored guard %s == %s, this could result in accuracy problems",
                expr,
                concrete_val,
                # only print stack trace when debug mode is on (e.g. TORCH_LOGS="dynamic")
                stack_info=log.getEffectiveLevel() < logging.WARNING,
            )

    def _get_user_frame(self) -> types.FrameType | None:
        frame = inspect.currentframe()
        while frame is not None:
            if frame.f_code.co_filename not in uninteresting_files():
                return frame
            frame = frame.f_back
        return frame

    def _get_stack_summary(
        self, is_debug: bool = False, framework_loc: str | None = None
    ) -> tuple[SLoc, str]:
        floc: str | traceback.FrameSummary | None = framework_loc
        if floc is None:
            frame = self._get_user_frame()
            try:
                if frame is not None:
                    floc = traceback.FrameSummary(
                        frame.f_code.co_filename,
                        frame.f_lineno,
                        frame.f_code.co_name,
                    )
            finally:
                del frame

        # NB: this stack is truncated, but it's fine because the main
        # stack_info will give you the rest of the info you need
        maybe_user_loc = None
        user_tb = TracingContext.extract_stack()
        if user_tb:
            idx = len(user_tb) - 1
            while idx > 0 and user_tb[idx].filename in uninteresting_files():
                idx -= 1
            maybe_user_loc = format_frame(user_tb[idx], line=True)

        maybe_extra_debug = ""
        if is_debug and user_tb:
            maybe_extra_debug = (
                "\nUser Stack (most recent call last):\n"
                + "  (snipped, see stack below for prefix)\n"
                + "".join(traceback.format_list(user_tb))
            )
        if is_debug and config.extended_debug_cpp:
            cpp_stack = CapturedTraceback.extract(cpp=True)
            maybe_extra_debug += "\nC++ stack trace:\n" + "".join(cpp_stack.format())
        elif is_debug:
            maybe_extra_debug += (
                "\nFor C++ stack trace, run with TORCHDYNAMO_EXTENDED_DEBUG_CPP=1"
            )

        return SLoc(floc, maybe_user_loc), maybe_extra_debug

    # Pass in framework_loc to override the framework location info
    def _get_sloc(self, framework_loc: str | None = None) -> SLoc:
        sloc, _ = self._get_stack_summary(framework_loc=framework_loc)
        return sloc

    def _generate_unique_id(self, source_name: str) -> int:
        attempt = int(hashlib.sha256(source_name.encode()).hexdigest(), 16) % 100
        while attempt in self.unique_ids:
            attempt += 1
        self.unique_ids.add(attempt)
        return attempt

    def _find_frame_locals(self) -> _FrameLocalResult:
        """
        Given the current user code frame, finds the relevant lines of code,
        values of symbolic locals, and free symbols involved.
        """
        frame_locals: dict[str, Any] = {}
        frame_symbols: dict[str, str] = {}

        if (
            frame := _find_user_code_frame()
        ) is None or frame.f_code.co_filename == "<string>":
            return _FrameLocalResult()

        # find bytecode instructions relevant to the frame
        instructions = list(dis.Bytecode(frame.f_code))
        co_lines, offset = inspect.getsourcelines(frame.f_code)
        start, end, cur = None, None, None
        # pyrefly: ignore [bad-assignment]
        for i, instr in enumerate(instructions):
            if instr.starts_line is not None:
                cur = instr.starts_line
            if cur != frame.f_lineno:
                continue
            if start is None:
                start = end = i
            else:
                end = i

        if start is None or end is None:  # no instructions found
            return _FrameLocalResult()

        # track involved locals and free symbols
        def go(x: Any) -> str | None:
            if isinstance(x, torch.Tensor):
                for y in x.size():
                    go(y)
                for y in x.stride():
                    go(y)
                go(x.storage_offset())
                return (
                    f"Tensor(shape: {x.size()}, "
                    f"stride: {x.stride()}, "
                    f"storage_offset: {x.storage_offset()})"
                )
            elif isinstance(x, (SymBool, SymInt, SymFloat)):
                for s in x.node.expr.free_symbols:
                    if str(s) in frame_symbols:  # type: ignore[operator]
                        continue
                    if s in self.var_to_sources:
                        frame_symbols[str(s)] = self.var_to_sources[s][0].name  # type: ignore[assignment]
                return str(x)
            return None

        # go through instructions, seeing linenos & involved locals
        last_lineno = frame.f_lineno
        for instr in instructions[start : end + 1]:
            if (lineno := instr.starts_line) is not None:
                last_lineno = max(last_lineno, lineno)
            if isinstance(instr.argval, str) and instr.argval in frame.f_locals:
                flat_locals = pytree.tree_flatten(frame.f_locals[instr.argval])[0]
                frame_locals[instr.argval] = [
                    go(flat_local) for flat_local in flat_locals
                ]

        # store LOC
        locs = co_lines[frame.f_lineno - offset : last_lineno + 1 - offset]
        if not locs:
            return _FrameLocalResult()

        indent = len(locs[0]) - len(locs[0].lstrip())
        frame_loc = "".join([loc[indent:] for loc in locs]).strip()  # type: ignore[assignment]
        return _FrameLocalResult(
            loc=frame_loc, locals=frame_locals, symbols=frame_symbols
        )

    def _log_guard(self, prefix: str, g: SympyBoolean, forcing_spec: bool) -> None:
        dtrace_structured(
            "guard_added",
            metadata_fn=lambda: {
                "expr": str(g),
                "prefix": prefix,
                "expr_node_id": self._expr_sym_node_id,
                "user_stack": structured.get_user_stack(3),
                "stack": structured.get_framework_stack(3),
                "symbol_to_sources": {
                    str(v): k
                    for k, v in self.source_to_var.items()
                    if v in g.free_symbols
                },
                "frame_locals": asdict(self._find_frame_locals()),
            },
        )
        trace_structured(
            "guard_added_fast",
            metadata_fn=lambda: {
                "expr": str(g),
                "user_stack": structured.from_traceback(TracingContext.extract_stack()),
                "stack": structured.from_traceback(
                    CapturedTraceback.extract(skip=1).summary()
                ),
            },
        )
        if self.log.isEnabledFor(logging.INFO):
            str_g = str(g)
            is_debug = (
                config.extended_debug_guard_added is not None
                and str_g == config.extended_debug_guard_added
            )
            sloc, maybe_extra_debug = self._get_stack_summary(is_debug)
            maybe_more_info = ""
            if not is_debug:
                maybe_more_info = (
                    ", for more info run with "
                    f'TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="{str_g}"'
                )
            self.log.info(
                "%s %s [guard added] %s%s%s",
                prefix if not forcing_spec else f"{prefix} (forcing_spec)",
                str_g,
                sloc,
                maybe_more_info,
                maybe_extra_debug,
                stack_info=is_debug,
            )

    # A local variable to evaluate_expr stored in the class to avoid
    # using it for the lru_cache that is on top of it since it does
    # not effect the results. When needed its read directly.
    _expr_sym_node_id: int | None = None

    def evaluate_sym_node(
        self,
        sym_node: SymNode,
        size_oblivious: bool = False,
        fallback_value: bool | None = None,
    ) -> sympy.Basic:
        """
        Given a a SymNode, evaluates sym_node.expr, adding guards if necessary.
        """

        self._expr_sym_node_id = id(sym_node)
        return self.evaluate_expr(
            sym_node.expr,
            sym_node.hint,
            sym_node.fx_node,
            size_oblivious,
            fallback_value=fallback_value,
        )

    def _is_python_assert(self) -> bool:
        # Check if this boolean is used in an assertion, bytecode pattern for
        # assertions is pretty stable for Python 3.7--3.13, ported with minimal
        # changes from torch/fx/proxy.py
        # Bytecode pattern for `assert` statements:
        #     TO_BOOL / COMPARE_OP  # Only for Python >= 3.13
        #     POP_JUMP_IF_TRUE
        #     LOAD_ASSERTION_ERROR
        #     RAISE_VARARGS
        frame = self._get_user_frame()
        if frame is None:
            raise AssertionError("frame must not be None")

        insts = list(dis.get_instructions(frame.f_code))
        if sys.version_info >= (3, 11):
            # For Python >= 3.11, instructions can be 2-4 bytes long.
            from bisect import bisect_left

            cur = bisect_left(insts, frame.f_lasti, key=lambda x: x.offset)
        else:
            # For Python <= 3.10, instructions are always 2 bytes.
            cur = frame.f_lasti // 2

        if sys.version_info >= (3, 13):
            if insts[cur].opname in ("TO_BOOL", "COMPARE_OP"):
                # Peek 1 instruction further.
                cur += 1

        assert_insts = torch._dynamo.symbolic_convert.get_assert_bytecode_sequence(
            False
        )

        cur_insts = insts[cur + 1 : cur + 1 + len(assert_insts)]
        cur_insts = [inst.opname for inst in cur_insts]
        return cur_insts == assert_insts

    def _log_real_tensor_propagation(
        self, orig_expr: sympy.Basic, unsound_result: sympy.Basic
    ) -> None:
        log.warning(
            "propagate_real_tensors evaluate_expr(%s) -> %s",
            orig_expr,
            unsound_result,
        )
        trace_structured(
            "propagate_real_tensors",
            metadata_fn=lambda: {
                "expr": repr(orig_expr),
                "result": repr(unsound_result),
                "stack": structured.from_traceback(
                    CapturedTraceback.extract(skip=1).summary()
                ),
            },
        )
        dtrace_structured(
            "propagate_real_tensors_provenance",
            metadata_fn=lambda: {
                "expr": repr(orig_expr),
                "result": repr(unsound_result),
                "expr_node_id": self._expr_sym_node_id,
                "user_stack": structured.get_user_stack(3),
                "stack": structured.get_framework_stack(3),
                "symbol_to_sources": {
                    str(v): k
                    for k, v in self.source_to_var.items()
                    if v in orig_expr.free_symbols
                },
                "frame_locals": asdict(self._find_frame_locals()),
            },
        )

    def evaluate_expr(
        self,
        orig_expr: sympy.Basic,
        hint: int | bool | float | None = None,
        fx_node: torch.fx.Node | None = None,
        size_oblivious: bool = False,
        fallback_value: bool | None = None,
        *,
        forcing_spec: bool = False,
    ) -> sympy.Basic:
        """
        Given an expression, evaluates it, adding guards if necessary
        When fallback_value is not None the function return fallback_value instead of failing with data dependent error.
        """

        # Add extra state that evaluate_expr() depends on.
        suppress_guards_tls = ShapeEnv._suppress_guards_tls()
        return self._inner_evaluate_expr(
            orig_expr,
            hint,
            fx_node,
            size_oblivious,
            forcing_spec,
            suppress_guards_tls,
            fallback_value,
        )

    @lru_cache(256)
    @record_shapeenv_event(save_tracked_fakes=True, name="evaluate_expr")
    def _inner_evaluate_expr(
        self,
        orig_expr: sympy.Basic,
        hint: int | bool | float | None,
        fx_node: torch.fx.Node | None,
        size_oblivious: bool,
        forcing_spec: bool,
        _suppress_guards_tls: bool,
        fallback_value: bool | None = None,
    ) -> sympy.Basic:
        try:
            return self._evaluate_expr(
                orig_expr,
                hint,
                fx_node,
                size_oblivious,
                fallback_value,
                forcing_spec=forcing_spec,
            )
        except Exception as e:
            if isinstance(e, GuardOnDataDependentSymNode):
                pass
            else:
                self.log.warning(
                    "failed during evaluate_expr(%s, hint=%s, size_oblivious=%s, forcing_spec=%s",
                    orig_expr,
                    hint,
                    size_oblivious,
                    forcing_spec,
                )
            raise

    def _log_suppressed_dde(self, a: SymBool, assumed_value: bool) -> None:
        sloc, extra = self._get_stack_summary(True)
        log.info(
            "could not evaluate %s due to data dependency, it was assumed to be %s with no runtime assertions %s %s",
            a,
            assumed_value,
            sloc,
            extra,
        )

    def _evaluate_expr(
        self,
        orig_expr: sympy.Basic,
        hint: bool | int | float | None = None,
        fx_node: torch.fx.Node | None = None,
        size_oblivious: bool = False,
        fallback_value: bool | None = None,
        *,
        forcing_spec: bool = False,
    ) -> sympy.Basic:
        # TODO: split conjunctions and evaluate them separately
        if isinstance(
            orig_expr,
            (sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse),
        ):
            return orig_expr

        # Don't track this one. (Because this cache is inside this function the
        # cache only lasts for the invocation of this function call)
        @functools.cache
        def compute_concrete_val() -> sympy.Basic:
            if hint is None:
                # This is only ever called for expressions WITHOUT unbacked
                # symbols.  guarding_hint_or_throw returns Python bool for
                # boolean expressions and int for integer expressions;
                # sympify converts them to the proper sympy types
                # (True -> sympy.true, 5 -> Integer(5)).
                try:
                    return sympy.sympify(self.guarding_hint_or_throw(orig_expr))
                except GuardOnDataDependentSymNode:
                    # guarding_hint_or_throw only does backed-symbol replacement.
                    # For expressions with unbacked symbols resolvable via axioms
                    # (e.g. Eq(x, 0) when torch._check(Ne(x, 0)) was previously
                    # asserted), fall back to static evaluation with compute_hint.
                    r = self._maybe_evaluate_static(orig_expr, compute_hint=True)
                    if r is not None:
                        return r
                    raise
            else:
                return sympy.sympify(hint)

        concrete_val: sympy.Basic | None

        # Check if:
        #   1. 'translation_validation' is set
        #   2. the corresponding 'fx_node' is not 'None'
        #   3. the guard should not be suppressed
        #   4. the guard doesn't contain backed symfloat symbols
        #      since z3 can't handle floats
        #   5. fallback_value is none.
        # If all of the above check, we create an FX node representing the
        # actual expression to be guarded.
        node = None
        fresh = False
        if (
            self._translation_validation_enabled
            and fx_node is not None
            and not self._suppress_guards_tls()
            and not size_oblivious
            and not any(symbol_is_type(s, SymT.FLOAT) for s in orig_expr.free_symbols)
            and fallback_value is None
        ):
            # TODO: does this even worked with unbacked :think:
            concrete_val = compute_concrete_val()
            if concrete_val is sympy.true:
                node, fresh = self._create_fx_call_function(torch._assert, (fx_node,))
            elif concrete_val is sympy.false:
                neg, _ = self._create_fx_call_function(operator.not_, (fx_node,))
                node, fresh = self._create_fx_call_function(torch._assert, (neg,))
            else:
                eql, _ = self._create_fx_call_function(
                    operator.eq, (fx_node, concrete_val)
                )
                node, fresh = self._create_fx_call_function(torch._assert, (eql,))

            if node is None:
                raise AssertionError("node must not be None")
            # If this is a fresh node, we have to remember the event index that
            # corresponds to this assertion node.
            # Reason: so that, given an assertion node, we can replay the ShapeEnv
            # events until the point where this assertion node was freshly created.
            if fresh:
                self._add_fx_node_metadata(node)

        # After creating the FX node corresponding to orig_expr, we must make sure that
        # no error will be raised until the end of this function.
        #
        # Reason: the translation validation may become invalid otherwise.
        #
        # If an error is raised before the end of this function, we remove the FX node
        # inserted, and re-raise the error.
        guard = None

        try:
            if orig_expr.is_number:
                self.log.debug("eval %s [trivial]", orig_expr)
                if hint is not None:
                    if isinstance(hint, bool):
                        if orig_expr != hint:
                            raise AssertionError(f"{orig_expr} != {hint}")
                    else:
                        if not sympy.Eq(orig_expr, hint):
                            raise AssertionError(f"{orig_expr} != {hint}")
                return orig_expr

            expr = orig_expr

            # Try to quickly evaluate trivially true/false comparisons
            # using var_to_range, before calling expensive _maybe_evaluate_static.
            if (
                torch.fx.experimental._config.aggressive_guard_free_semantics
                < AggressiveGuardFreeMode.SKIP_RANGE_ANALYSIS
            ):
                fast_result = self._maybe_fast_eval_comparison(expr)
                if fast_result is not None:
                    return fast_result

            # Aggressive guard-free semantics:
            # VALUE_RANGE_ANALYSIS: use value range analysis (bound_sympy) before returning fallback
            # SKIP_RANGE_ANALYSIS: skip range analysis entirely, just return fallback_value
            aggressive_level = (
                torch.fx.experimental._config.aggressive_guard_free_semantics
            )
            if hint is None and aggressive_level > 0 and fallback_value is not None:
                if aggressive_level >= AggressiveGuardFreeMode.SKIP_RANGE_ANALYSIS:
                    # Skip range analysis entirely
                    self._log_suppressed_dde(orig_expr, fallback_value)
                    return fallback_value
                else:
                    # Level 1: try range analysis first
                    range_result = self._maybe_evaluate_range_only(expr, fallback_value)
                    if range_result is fallback_value:
                        self._log_suppressed_dde(orig_expr, fallback_value)
                    return range_result

            static_expr = self._maybe_evaluate_static(
                expr, size_oblivious=size_oblivious
            )
            if static_expr is not None:
                self.log.debug(
                    "eval %s == %s [statically known]",
                    (
                        f"size_oblivious({orig_expr})"
                        if size_oblivious
                        else size_oblivious
                    ),
                    static_expr,
                )
                if (
                    not size_oblivious
                    and config.backed_size_oblivious
                    and hint is not None
                ):
                    # TODO: maybe reconcile this with use of counterfactual hints
                    # in unbacked case
                    if static_expr != hint:
                        raise AssertionError(f"{static_expr} != {hint}")
                return static_expr

            transmute_into_runtime_assert = False

            concrete_val = None
            if not (expr.free_symbols <= self.backed_var_to_val.keys()):
                # TODO: dedupe this with _maybe_evaluate_static
                # Attempt to eliminate the unbacked SymInt
                new_expr = self._maybe_evaluate_static(expr, unbacked_only=True)
                if new_expr is None:
                    raise AssertionError("new_expr must not be None")
                if not (new_expr.free_symbols <= self.backed_var_to_val.keys()):
                    ok = False

                    # fallback_value is set when guard_or_true or guard_or_false are used.
                    if not ok and fallback_value is not None:
                        self._log_suppressed_dde(orig_expr, fallback_value)
                        return fallback_value

                    # real_tensor_prop_unbacked_vals is not None iff propagate_real_tensors is on.
                    # if propagate_real_tensors is on, we check the example values to generate (unsound_result)
                    # and if they pass we add a runtime assertions and continue.
                    if (
                        not ok
                        and self.real_tensor_prop_unbacked_vals
                        and not (
                            unsound_result := orig_expr.xreplace(
                                self.real_tensor_prop_unbacked_vals
                            ).xreplace(self.backed_var_to_val)
                        ).free_symbols
                    ):
                        self._log_real_tensor_propagation(orig_expr, unsound_result)
                        transmute_into_runtime_assert = True

                        concrete_val = unsound_result
                        ok = True

                    # Check if this is coming from a python assert statement, if so, convert it to a runtime assertion
                    # instead of failing.
                    if not ok and self.trace_asserts and self._is_python_assert():
                        concrete_val = sympy.true
                        transmute_into_runtime_assert = True
                        ok = True

                    if not ok:
                        raise self._make_data_dependent_error(
                            expr.xreplace(self.backed_var_to_val),
                            expr,
                            expr_sym_node_id=self._expr_sym_node_id,
                        )
                else:
                    expr = new_expr

            if concrete_val is None:
                concrete_val = compute_concrete_val()
            self._check_frozen(expr, concrete_val)

            if (
                config.inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY
                and isinstance(hint, bool)
                and isinstance(expr, (sympy.Eq, sympy.Ne))
            ):
                expr = sympy.Not(expr)

            # Turn this into a boolean expression, no longer need to consult
            # concrete_val
            if concrete_val is sympy.true:
                g = cast(SympyBoolean, expr)
            elif concrete_val is sympy.false:
                g = sympy.Not(expr)
            else:
                g = sympy.Eq(expr, concrete_val)  # type: ignore[arg-type]

            if transmute_into_runtime_assert:
                self.guard_or_defer_runtime_assert(
                    g, f"propagate_real_tensors: {orig_expr} == {concrete_val}"
                )
                return concrete_val

            if not self._suppress_guards_tls():
                self._log_guard("eval", g, forcing_spec=forcing_spec)

                # TODO: If we successfully eliminate a symbol via equality, it
                # is not actually necessary to save a guard for the equality,
                # as we will implicitly generate a guard when we match that
                # input against the symbol.  Probably the easiest way to
                # implement this is to have maybe_guard_rel return a bool
                # saying if it "subsumed" the guard (and therefore the guard
                # is no longer necessary)
                self._maybe_guard_rel(g)

                if (
                    torch.compiler.is_exporting()
                    and self.prefer_deferred_runtime_asserts_over_guards
                ):
                    # it's fine to defer simple guards here without checking,
                    # the _maybe_guard_rel() call above will set replacements if possible,
                    # and so the result here will be statically known
                    self.guard_or_defer_runtime_assert(g, f"evaluate_expr: {orig_expr}")
                else:
                    # at this point, we've evaluated the concrete expr value, and have
                    # flipped/negated the guard if necessary. Now we know what to guard
                    # or defer to runtime assert on.
                    guard = ShapeGuard(
                        g, self._get_sloc(), size_oblivious=size_oblivious
                    )
                    self.guards.append(guard)
                    self.axioms.update(dict(self.get_implications(self.simplify(g))))
            else:
                self._log_guard("eval [guard suppressed]", g, forcing_spec=forcing_spec)

        except Exception:
            if fresh:
                self._remove_fx_node(node)
            raise

        if not self._suppress_guards_tls():
            if guard is not None:  # we might have deferred this to runtime assert
                for s in g.free_symbols:
                    self.symbol_guard_counter[s] += 1
                    # Forcing_spec to avoid infinite recursion
                    if (
                        not forcing_spec
                        and config.symbol_guard_limit_before_specialize is not None
                        and self.symbol_guard_counter[s]
                        > config.symbol_guard_limit_before_specialize
                    ):
                        # Force specialization
                        self.log.info(
                            "symbol_guard_limit_before_specialize=%s exceeded on %s",
                            config.symbol_guard_limit_before_specialize,
                            s,
                        )
                        self.evaluate_expr(s, forcing_spec=True)

        return concrete_val

    def cleanup(self) -> None:
        """
        Break reference cycles.

        This destroys the stacks. If you really want to keep them, we
        just need some way to break references on code objects.
        """
        for s in self.var_to_stack.values():
            s.cleanup()
        for ras in self.deferred_runtime_asserts.values():
            for ra in ras:
                ra.stack.cleanup()

    def _should_skip_static_eval(self, expr: SympyBoolean) -> bool:
        """Check if we should skip _maybe_evaluate_static for the given expression.

        Skips static evaluation for single unbacked symbol >= 0 (or 0 <= symbol)
        when the symbol has unknown range [-int_oo, int_oo].
        This pattern is common during tracing and doesn't benefit from static evaluation
        since the symbol has no constraints.
        Note that the first time this is called value range will be updated and next time
        it's called (if any) we would call _maybe_evaluate_static and it would return True.
        """
        unbacked_sym = None
        if isinstance(expr, sympy.GreaterThan) and expr.rhs == 0:
            unbacked_sym = expr.lhs
        elif isinstance(expr, sympy.LessThan) and expr.lhs == 0:
            unbacked_sym = expr.rhs
        if isinstance(unbacked_sym, sympy.Symbol) and symbol_is_type(
            unbacked_sym, SymT.UNBACKED_INT
        ):
            vr = self.var_to_range[unbacked_sym]
            if vr.lower == -int_oo and vr.upper == int_oo:
                return True
        return False

    @lru_cache(256)
    @record_shapeenv_event(save_tracked_fakes=True)
    def guard_or_defer_runtime_assert(
        self, orig_expr: SympyBoolean, msg: str, fx_node: torch.fx.Node | None = None
    ) -> bool:
        """
        Adds a guard that orig_expr is True if we can or fall back to adding an assert
        that is checked at runtime.

        Args:
            orig_expr (sympy.Expr): Boolean expression to assert is true
            msg (str): Message to display on assertion failure
            fx_node (Optional, torch.fx.Node): node in ``self.graph`` corresponding
                to the expression, if applicable
        """
        expr = orig_expr

        # TODO: split conjunctions and evaluate them separately
        # Try to quickly evaluate trivially true/false comparisons
        # using var_to_range, before calling expensive _maybe_evaluate_static.
        fast_result = self._maybe_fast_eval_comparison(expr)
        if fast_result is not None:
            return bool(fast_result)

        if self._should_skip_static_eval(expr):
            new_expr = expr
        else:
            static_expr = self._maybe_evaluate_static(expr)
            if static_expr is not None:
                self.log.debug(
                    "runtime_assert %s == %s [statically known]", orig_expr, static_expr
                )
                # TODO: assert bool(static_expr)
                return bool(static_expr)

            # Attempt to eliminate the unbacked SymInt
            new_expr = self._maybe_evaluate_static(expr, unbacked_only=True)
            if new_expr is None:
                raise AssertionError("new_expr must not be None")
        if (
            not self.prefer_deferred_runtime_asserts_over_guards
            and new_expr.free_symbols <= self.backed_var_to_val.keys()
        ):
            # Do a normal guard
            return self.evaluate_expr(new_expr, fx_node=fx_node)
        # NB: Don't use new_expr as expr; it could contain gunk like shape0
        # which we don't want to guard on

        if (
            self._translation_validation_enabled
            and fx_node is not None
            and not self._suppress_guards_tls()
        ):
            node, fresh = self._create_fx_call_function(torch._assert, (fx_node,))
            if node is None:
                raise AssertionError("node must not be None")
            if fresh:
                self._add_fx_node_metadata(node)

        if not self._suppress_guards_tls():
            self._log_guard("runtime_assert", orig_expr, forcing_spec=False)
            # If you're here because of this assert, read Note [Backwards runtime asserts]
            # in torch/_inductor/graph.py
            if self.runtime_asserts_frozen:
                log.debug("runtime_asserts_frozen but then got %s", expr)
            self._check_frozen(expr, sympy.true)
            # eliminate symbols on equality tests / refine ranges
            self._maybe_guard_rel(expr)

            # canonicalise to remove equations that are trivially equal
            orig_expr = expr
            expr = canonicalize_bool_expr(expr)
            stack = CapturedTraceback.extract(skip=1)
            ra = RuntimeAssert(expr, msg, stack)

            # TODO: Do this in a way that is less janky than int(s.name[1:])
            cands = sorted(
                (s for s in expr.free_symbols if symbol_is_type(s, SymT.UNBACKED_INT)),
                key=lambda s: int(s.name[1:]),
            )
            # Is None when prefer_deferred_runtime_asserts_over_guards=True
            # and the guard in question has no unbacked SymInts in front
            ix = cands[-1] if cands else None
            self.deferred_runtime_asserts.setdefault(ix, []).append(ra)
            self.axioms.update(dict(self.get_implications(self.simplify(expr))))
            self.num_deferred_runtime_asserts += 1
            self._update_version_counter()
        else:
            self._log_guard(
                "runtime_assert [guard suppressed]", orig_expr, forcing_spec=False
            )

        return True

    # Refines the ranges of the variables present in 'guard'.
    #
    # This function tries to refine the range of the variables inside
    # 'guard' by reasoning about it. Specifically, when 'guard' is a
    # 'sympy.Relational' operation.
    #
    # It does mainly 3 things:
    #   1. Tries to isolate a variable in the left-hand side
    #   2. Compute the value range of the right-hand side
    #   3. Update the value range of the variable, if better
    def _refine_ranges(self, expr: SympyBoolean) -> None:
        expr = self.simplify(expr)

        for symbol in expr.free_symbols:
            if not isinstance(symbol, sympy.Symbol):
                raise AssertionError(f"Expected sympy.Symbol, got {type(symbol)}")

            if isinstance(self.backed_var_to_val.get(symbol, None), SingletonInt):
                # Skip var_to_range logic for SingletonInt which is only used
                # for jagged layout NestedTensors today
                continue

            r = try_solve(expr, symbol)

            if r is None or not (symbol.is_integer and r[1].is_integer):
                # Range refinement only supports integer symbols for now.
                # There are lots of SymPy bugs when it comes to comparing
                # reals and integers, so we skip that for now.
                continue

            r_expr, rhs = r
            vr = self.var_to_range[symbol]
            lower, upper = vr.lower, vr.upper

            rhs_vr = bound_sympy(rhs, self.var_to_range)

            # Let's suppose that we have a preexisting range for x [0, 100].
            # Now, we issue a guard x > y, where the range for y is [50, 150].
            # Then, lower = 0, rhs_vr.lower = 50 and therefore refinement can happen,
            # refining x to [51, 100], since x must be greater than y, but the lowest
            # y could be is 50.
            #
            # sympy.Eq may update both lower and upper bounds.
            # sympy.G{t,e} may update the lower bound, only.
            # sympy.L{t,e} may update the upper bound, only.
            if lower <= rhs_vr.lower and isinstance(
                r_expr, (sympy.Eq, sympy.Ge, sympy.Gt)
            ):
                # Strictly greater relations allow us to refine a bit more, since
                # x < y implies that the lower bound for x is: y + 1.
                lower = rhs_vr.lower + int(isinstance(r_expr, sympy.Gt))
            if upper >= rhs_vr.upper and isinstance(
                r_expr, (sympy.Eq, sympy.Le, sympy.Lt)
            ):
                upper = rhs_vr.upper - int(isinstance(r_expr, sympy.Lt))

            # Do nothing if the new value range is no better than what we already have.
            if vr == ValueRanges(lower, upper):
                continue

            # Updates the range and the guards corresponding to each bound of the symbol.
            self._update_var_to_range(symbol, ValueRanges(lower, upper))
            # If the range is refined to singleton, set replacement
            if self.var_to_range[symbol].is_singleton():
                self._set_replacement(
                    symbol,
                    self.var_to_range[symbol].lower,
                    "range_refined_to_singleton",
                )

            # Clears the cache, since this update can change the result.
            self._maybe_evaluate_static.cache_clear()

    @lru_cache(maxsize=None)
    @record_shapeenv_event()
    def constrain_symbol_range(
        self, s: sympy.Symbol, compiler_min: int, compiler_max: int
    ) -> None:
        upd_vr = ValueRanges(compiler_min, compiler_max)
        old_vr = self.var_to_range.get(s, ValueRanges.unknown())
        self._update_var_to_range(s, upd_vr)
        if (new_vr := self.var_to_range[s]) != old_vr:
            log.info(
                "constrain_symbol_range %s [%s, %s]", s, new_vr.lower, new_vr.upper
            )



__all__ = [
    "IndicatorTypes",
    "_expandsums",
    "_fast_expand",
    "safe_expand",
    "_SymbolInfo",
    "_maybe_evaluate_static_worker",
    "RuntimeAssert",
    "SymExprPrinter",
    "_ShapeGuardPrinter",
    "ShapeGuardPythonPrinter",
    "ShapeGuardPrinter",
    "_ShapeGuardCppPrinter",
    "_ShapeGuardsHelper",
    "_CppShapeGuardsHelper",
    "LoggingShapeGuardPrinter",
    "TLS",
    "ShapeEnvSettings",
    "ValueRangesSLoc",
    "_suppress_guards",
    "_FrameLocalResult",
    "ShapeEnv",
]
