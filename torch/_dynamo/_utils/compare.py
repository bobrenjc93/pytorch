from __future__ import annotations

import collections
import logging
import math
from typing import Any, TypeGuard

import torch


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


log = logging.getLogger("torch._dynamo.utils")


def is_numpy_int_type(value: Any) -> bool:
    if not np:
        return False

    return isinstance(
        value,
        (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    )


def is_numpy_float_type(value: Any) -> bool:
    if not np:
        return False

    return isinstance(
        value,
        (
            np.float16,
            np.float32,
            np.float64,
        ),
    )


def is_numpy_ndarray(value: Any) -> TypeGuard[np.ndarray]:  # type: ignore[type-arg]
    if not np:
        return False

    return isinstance(value, np.ndarray)


def rmse(ref: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    """
    Calculate root mean squared error
    """
    return torch.sqrt(torch.mean(torch.square(ref - res)))


def bitwise_same(ref: Any, res: Any, equal_nan: bool = False) -> bool:
    return same(
        ref,
        res,
        tol=0.0,
        equal_nan=equal_nan,
    )


def same(
    ref: Any,
    res: Any,
    fp64_ref: Any = None,
    cos_similarity: bool = False,
    tol: float = 1e-4,
    equal_nan: bool = False,
    exact_dtype: bool = True,
    relax_numpy_equality: bool = False,
    ignore_non_fp: bool = False,
    log_error: Any = log.error,
    use_larger_multiplier_for_smaller_tensor: bool = False,
    force_max_multiplier: bool = False,
    use_iou_for_bool: bool = False,
    iou_threshold: float = 0.99,
) -> bool:
    """Check correctness to see if ref and res match"""
    if fp64_ref is None:
        fp64_ref = ref
    if isinstance(
        ref, (list, tuple, collections.deque, torch.nn.ParameterList, torch.Size)
    ):
        assert isinstance(res, (list, tuple, collections.deque)), (
            f"type mismatch {type(ref)} {type(res)}"
        )
        if len(ref) != len(res):
            log_error("Length mismatch")
            return False
        return len(ref) == len(res) and all(
            same(
                ai,
                bi,
                fp64_refi,
                cos_similarity,
                tol,
                equal_nan,
                exact_dtype,
                relax_numpy_equality,
                ignore_non_fp,
                log_error=log_error,
                use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
                force_max_multiplier=force_max_multiplier,
                use_iou_for_bool=use_iou_for_bool,
                iou_threshold=iou_threshold,
            )
            for ai, bi, fp64_refi in zip(ref, res, fp64_ref)
        )
    elif type(ref).__name__ == "QuestionAnsweringModelOutput":
        # This skips checking accuracy for start_logits/end_logits.
        # Tentatively, start_logits/end_logits appear to be very prone to
        # inaccuracies and is somewhat subsumed by checking the loss.
        return same(
            ref.loss,
            res.loss,
            fp64_ref.loss,
            cos_similarity,
            tol,
            equal_nan,
            exact_dtype,
            relax_numpy_equality,
            ignore_non_fp,
            log_error=log_error,
            use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
            force_max_multiplier=force_max_multiplier,
            use_iou_for_bool=use_iou_for_bool,
            iou_threshold=iou_threshold,
        )
    elif isinstance(ref, dict):
        assert isinstance(res, dict)
        assert set(ref.keys()) == set(res.keys()), (
            f"keys mismatch {set(ref.keys())} == {set(res.keys())}"
        )
        for k in sorted(ref.keys()):
            if not (
                same(
                    ref[k],
                    res[k],
                    fp64_ref[k],
                    cos_similarity=cos_similarity,
                    tol=tol,
                    equal_nan=equal_nan,
                    exact_dtype=exact_dtype,
                    relax_numpy_equality=relax_numpy_equality,
                    ignore_non_fp=ignore_non_fp,
                    log_error=log_error,
                    use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
                    force_max_multiplier=force_max_multiplier,
                    use_iou_for_bool=use_iou_for_bool,
                    iou_threshold=iou_threshold,
                )
            ):
                log_error("Accuracy failed for key name %s", k)
                return False
        return True
    elif isinstance(ref, set):
        assert isinstance(res, set)
        assert set(ref) == set(res), f"elements mismatch {set(ref)} == {set(res)}"
        return True
    elif isinstance(ref, (torch.Tensor, float)):
        assert not isinstance(ref, torch._subclasses.FakeTensor)
        assert not isinstance(res, torch._subclasses.FakeTensor)

        def to_tensor(t: Any) -> torch.Tensor:
            return t if isinstance(t, torch.Tensor) else torch.tensor(t)

        ref, res, fp64_ref = (to_tensor(val) for val in (ref, res, fp64_ref))

        if ref.is_sparse:
            assert res.is_sparse
            ref = ref.to_dense()
            res = res.to_dense()
        assert isinstance(res, torch.Tensor), f"type mismatch {type(ref)} {type(res)}"
        if exact_dtype:
            if ref.dtype != res.dtype:
                log_error("dtype mismatch %s, %s", ref.dtype, res.dtype)
                return False
            if ref.dtype == torch.bool:
                if ignore_non_fp:
                    return True
                if use_iou_for_bool:
                    # Use IoU (Intersection over Union) metric for boolean mask comparison.
                    # This is useful for segmentation models where small floating-point
                    # differences get thresholded into boolean masks.
                    intersection = (ref & res).sum().float()
                    union = (ref | res).sum().float()
                    if union == 0:
                        # Both masks are empty
                        return bool(intersection == 0)
                    iou = (intersection / union).item()
                    if iou < iou_threshold:
                        log_error(
                            "IoU accuracy failed: %.4f < %.2f (intersection=%d, union=%d, ref_sum=%d, res_sum=%d, shape=%s)",
                            iou,
                            iou_threshold,
                            int(intersection.item()),
                            int(union.item()),
                            int(ref.sum().item()),
                            int(res.sum().item()),
                            list(ref.shape),
                        )
                        return False
                    return True
                # triton stores bool as int8, so add this for more accurate checking
                r = torch.allclose(
                    ref.to(dtype=torch.uint8),
                    res.to(dtype=torch.uint8),
                    atol=tol,
                    rtol=tol,
                    equal_nan=equal_nan,
                )
                if not r:
                    log_error("Accuracy failed: uint8 tensor did not match")
                return r

        if cos_similarity:
            ref = ref.flatten().to(torch.float32)
            res = res.flatten().to(torch.float32)
            if torch.allclose(ref, res, atol=tol, rtol=tol, equal_nan=True):
                # early exit that handles zero/nan better
                # cosine_similarity(zeros(10), zeros(10), dim=0) is 0
                return True
            score = torch.nn.functional.cosine_similarity(ref, res, dim=0, eps=1e-6)
            if score < 0.99:
                log.warning("Similarity score=%s", score.detach().cpu().item())
            return bool(score >= 0.99)
        else:
            if not exact_dtype:
                ref = ref.to(res.dtype)

            # First try usual allclose
            if torch.allclose(ref, res, atol=tol, rtol=tol, equal_nan=equal_nan):
                return True

            # Check error from fp64 version
            if fp64_ref.dtype == torch.float64:
                # Fix a corner case that res and fp64_ref does not contains NaN and match (with loose tolerance)
                # while the ref contains NaN. In this case, RMSE should not match any ways.
                # But res is 'BETTER' than ref so we count it pass.
                #
                # This happens for Super_SloMo when loop ordering after fusion is enabled:
                # https://gist.github.com/shunting314/11f235c70f7db0d52718d26f4a701cab
                loose_tol = 1e-2 * 4
                if (
                    not fp64_ref.isnan().any()
                    and not res.isnan().any()
                    and ref.isnan().any()
                    and torch.allclose(
                        fp64_ref.to(dtype=res.dtype),
                        res,
                        atol=loose_tol,
                        rtol=loose_tol,
                        equal_nan=equal_nan,
                    )
                ):
                    return True
                ref_error = rmse(fp64_ref, ref).item()
                # ref unable to produce this with stable numerics in this precision, ignore
                if math.isnan(ref_error):
                    log.warning(
                        "Found nan in reference. Consider running in higher precision."
                    )

                res_error = rmse(fp64_ref, res).item()

                def get_multiplier() -> float:
                    # In some particular cases, we expect high difference in results.
                    # At the moment one of this cases is inductor freezing bfloat16 convolution const folding.
                    # In case of it the res_error is at least one order of magnitude higher.
                    if force_max_multiplier:
                        return 10.0
                    # In the case of using AMP (Automatic Mixed Precision), certain models have
                    # failed the benchmark's correctness check. However, the end-to-end model's
                    # accuracy when comparing AMP with FP32 is within a difference of less than 0.1%.
                    # Thus, it's possible that the correctness check failures for these models are
                    # false alarms. We use multiplier of 3 instead of 2 to avoid these false alarms.
                    multiplier = (
                        3.0 if res.dtype in (torch.float16, torch.bfloat16) else 2.0
                    )

                    if use_larger_multiplier_for_smaller_tensor and (
                        fp64_ref.numel() <= 10
                    ):
                        multiplier = 10.0
                    elif use_larger_multiplier_for_smaller_tensor and (
                        fp64_ref.numel() <= 500
                    ):
                        multiplier = 8.0
                    elif (
                        fp64_ref.numel() < 1000
                        or (ref.ndim == 4 and ref.shape[-1] == ref.shape[-2] == 1)
                        # large tol means a benchmark has been specified as REQUIRE_HIGHER_TOLERANCE
                        or tol >= 2 * 1e-2
                    ):
                        # In the presence of noise, noise might dominate our error
                        # metric for smaller tensors.
                        # Similarly, for 1x1 kernels, there seems to be high noise with amp.
                        multiplier = 3.0
                    return multiplier

                multiplier = get_multiplier()

                passes_test = res_error <= (multiplier * ref_error + tol / 10.0)
                if (
                    not passes_test
                    and equal_nan
                    and math.isnan(ref_error)
                    and math.isnan(res_error)
                    # Some unit test for the accuracy minifier relies on
                    # returning false in this case.
                    and not torch._inductor.config.cpp.inject_relu_bug_TESTING_ONLY
                ):
                    passes_test = True
                if not passes_test:
                    log_error(
                        "RMSE (res-fp64): %.5f, (ref-fp64): %.5f and shape=%s. res.dtype: %s, multiplier: %f, tol: %f"
                        ", use_larger_multiplier_for_smaller_tensor: %d",
                        res_error,
                        ref_error,
                        res.size(),
                        res.dtype,
                        multiplier,
                        tol,
                        use_larger_multiplier_for_smaller_tensor,
                    )
                return passes_test

            if ignore_non_fp:
                return True

            log_error("Accuracy failed: allclose not within tol=%s", tol)
            return False
    elif isinstance(ref, (str, int, type(None), bool, torch.device)):
        if ignore_non_fp:
            return True
        r = ref == res
        if not r:
            log_error("Accuracy failed (%s): %s != %s", type(ref), ref, res)
        return r
    elif is_numpy_int_type(ref) or is_numpy_float_type(ref):
        if relax_numpy_equality and not (
            is_numpy_int_type(res) or is_numpy_float_type(res)
        ):
            ref = ref.item()
        r = (type(ref) is type(res)) and (ref == res)
        if not r:
            log_error("Accuracy failed (numpy): %s != %s", ref, res)
        return r
    elif is_numpy_ndarray(ref):
        return (type(ref) is type(res)) and same(
            torch.as_tensor(ref),
            torch.as_tensor(res),
            fp64_ref,
            cos_similarity=cos_similarity,
            tol=tol,
            equal_nan=equal_nan,
            exact_dtype=exact_dtype,
            relax_numpy_equality=relax_numpy_equality,
            ignore_non_fp=ignore_non_fp,
            log_error=log_error,
            use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
        )
    elif type(ref).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
        "LongformerMaskedLMOutput",
        "Instances",
        "SquashedNormal",
        "Boxes",
        "Normal",
        "TanhTransform",
        "Foo",
        "Variable",
    ):
        assert type(ref) is type(res)
        return all(
            same(
                getattr(ref, key),
                getattr(res, key),
                getattr(fp64_ref, key),
                cos_similarity=cos_similarity,
                tol=tol,
                equal_nan=equal_nan,
                exact_dtype=exact_dtype,
                relax_numpy_equality=relax_numpy_equality,
                ignore_non_fp=ignore_non_fp,
                log_error=log_error,
                use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
            )
            for key in ref.__dict__
        )
    else:
        raise RuntimeError(f"unsupported type: {type(ref).__name__}")
