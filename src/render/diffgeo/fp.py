"""Arithmetic modes of the differentiable chain.

All tensors are float64.  Two modes:

* ``'exact'`` -- float64 compute / float32 round: ``R(x)`` rounds to binary32 at exactly the
  points where the game rounds (after every native SSE float operation; at Mono stores,
  float32 arguments and returns).  Double rounding through binary64 is innocuous for one
  +, -, *, /, sqrt of binary32 operands (53 >= 2*24 + 2), so ``R(a op b)`` equals the binary32
  operation; a Mono double expression is evaluated in float64 and rounded once.  The CRT
  ``sinf`` / ``cosf`` of UnityPlayer.dll are executed from the installed binary
  (``src.render.hs2pipe.unity_crt``, Unicorn) -- they are not correctly rounded, so no torch
  function reproduces them -- and their backward is the analytic derivative of sin / cos.
* ``'f64'`` -- every ``R`` is the identity and sin / cos are torch's float64 functions: the same
  operation graph without rounding, for float64 finite-difference gradient checks.

Autograd flows through ``R`` (a cast; its derivative is taken as 1, the derivative of the
unrounded operation).  ``S(x)`` = copysign(1, x) (sign-of-scale products of the native
hierarchy walk) has zero derivative.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

DT = torch.float64


@lru_cache(maxsize=1 << 16)
def _crt(fn, bits):
    from src.render.hs2pipe import unity_crt
    x = np.frombuffer(np.uint32(bits).tobytes(), np.float32)[0]
    return float(getattr(unity_crt, fn)(x))


def _crt_eval(fn, x):
    xs = x.detach().to(torch.float32).cpu().numpy().ravel()
    if not np.array_equal(xs.astype(np.float64), x.detach().cpu().numpy().ravel()):
        raise ValueError('CRT %s argument is not a binary32 value (exact mode requires rounded inputs)' % fn)
    out = [_crt(fn, int(b)) for b in xs.view(np.uint32)]
    return torch.tensor(out, dtype=DT).reshape(x.shape)


class _CrtSin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _crt_eval('sinf', x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * torch.cos(x)


class _CrtCos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _crt_eval('cosf', x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return -g * torch.sin(x)


class _ForwardFrom(torch.autograd.Function):
    """Value of ``taken`` (bit for bit), gradient routed to ``alt``."""

    @staticmethod
    def forward(ctx, taken, alt):
        return taken.clone()

    @staticmethod
    def backward(ctx, g):
        return None, g


def forward_from(taken, alt):
    return _ForwardFrom.apply(taken, alt)


class Arith:
    def __init__(self, mode='exact', identity_gradient='branch'):
        """identity_gradient: 'branch' (default; autograd of the taken branch: zero for an ABMX
        channel at exactly its identity value) or 'one_sided' (``abmx._one_sided``)."""
        if mode not in ('exact', 'f64'):
            raise ValueError("mode must be 'exact' or 'f64'")
        if identity_gradient not in ('branch', 'one_sided'):
            raise ValueError("identity_gradient must be 'branch' or 'one_sided'")
        self.mode = mode
        self.exact = mode == 'exact'
        self.one_sided = identity_gradient == 'one_sided'

    def R(self, x):
        if not self.exact:
            return x
        if isinstance(x, torch.Tensor):
            return x.to(torch.float32).to(DT)
        return float(np.float32(x))

    @staticmethod
    def S(x):
        if isinstance(x, torch.Tensor):
            return torch.copysign(torch.ones_like(x), x.detach())
        return float(np.copysign(1.0, x))

    @staticmethod
    def sqrt(x):
        return torch.sqrt(x)

    def sinf(self, x):
        return _CrtSin.apply(x) if self.exact else torch.sin(x)

    def cosf(self, x):
        return _CrtCos.apply(x) if self.exact else torch.cos(x)

    def f32(self, value):
        """A binary32 constant/input as a float64 tensor (rounded in both modes: game data is binary32)."""
        return torch.as_tensor(np.asarray(value, np.float32).astype(np.float64), dtype=DT)

    def lanes(self, value):
        t = value if isinstance(value, torch.Tensor) else self.f32(value)
        return [t[..., i] for i in range(t.shape[-1])]
