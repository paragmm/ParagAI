"""Verification of scalar autograd engine against PyTorch autograd."""

import torch

from paragai.engine.nn import MLP
from paragai.engine.value import Value


def math_close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


def test_autograd_basic_ops():
    x1 = Value(2.0)
    x2 = Value(3.0)
    w1 = Value(-1.5)
    w2 = Value(2.5)
    b = Value(0.5)

    n = (x1 * w1) + (x2 * w2) + b
    out = n.relu()
    out.backward()

    px1 = torch.tensor(2.0, requires_grad=True)
    px2 = torch.tensor(3.0, requires_grad=True)
    pw1 = torch.tensor(-1.5, requires_grad=True)
    pw2 = torch.tensor(2.5, requires_grad=True)
    pb = torch.tensor(0.5, requires_grad=True)

    pn = (px1 * pw1) + (px2 * pw2) + pb
    pout = torch.relu(pn)
    pout.backward()

    assert math_close(out.data, pout.item())
    assert math_close(x1.grad, px1.grad.item())
    assert math_close(x2.grad, px2.grad.item())
    assert math_close(w1.grad, pw1.grad.item())
    assert math_close(w2.grad, pw2.grad.item())
    assert math_close(b.grad, pb.grad.item())


def test_autograd_complex_ops():
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()

    pa = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    pb = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    pc = pa + pb
    pd = pa * pb + pb**3
    pc = pc + pc + 1
    pc = pc + 1 + pc + (-pa)
    pd = pd + pd * 2 + torch.relu(pb + pa)
    pd = pd + 3 * pd + torch.relu(pb - pa)
    pe = pc - pd
    pf = pe**2
    pg = pf / 2.0
    pg = pg + 10.0 / pf
    pg.backward()

    assert math_close(g.data, pg.item())
    assert math_close(a.grad, pa.grad.item())
    assert math_close(b.grad, pb.grad.item())


def test_mlp_forward_and_backward():
    mlp = MLP(3, [4, 4, 1])
    x = [Value(1.0), Value(2.0), Value(-1.0)]
    out = mlp(x)
    assert len(out) == 1
    out[0].backward()

    # At least some parameters must have non-zero gradients
    grads = [p.grad for p in mlp.parameters()]
    assert any(g != 0.0 for g in grads)
