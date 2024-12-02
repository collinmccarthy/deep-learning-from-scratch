"""
Modified from https://github.com/karpathy/micrograd/blob/master/test/test_engine.py
    - Added on-the-fly tests from YouTube/Notebook
"""

import torch
from torch import nn
from dl_scratch.micrograd.engine import Value
from dl_scratch.micrograd.nn import Neuron, Layer, MLP


def run_value_example_tanh() -> list[Value]:
    # From https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_second_half_roughly.ipynb

    # -------------
    # Forward pass
    # -------------
    # inputs x1,x2
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")

    # weights w1,w2
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")

    # bias of the neuron (so tanh grad is nice number)
    b = Value(6.8813735870195432, label="b")

    # n = x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"
    n = x1w1x2w2 + b
    n.label = "n"

    # o = tanh(n)
    o = n.tanh()
    o.label = "o"

    # -------------
    # Backward pass
    # -------------
    o.backward()

    # Can call `draw_dot(o)` on return value in jupyter notebook
    return [o, x1, w1, x2, w2]


def run_value_example_tanh_with_exp() -> list[Value]:
    # From https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_second_half_roughly.ipynb

    # -------------
    # Forward pass
    # -------------
    # inputs x1,x2
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")

    # weights w1,w2
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")

    # bias of the neuron (so tanh grad is nice number)
    b = Value(6.8813735870195432, label="b")

    # n = x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"
    n = x1w1x2w2 + b
    n.label = "n"

    # o = tanh(n) == (exp(2n) - 1) / (exp(2n) + 1)
    #   - Uses e = exp(2n); o = (e-1)/(e+1)
    #   - See https://en.wikipedia.org/wiki/Hyperbolic_functions#Exponential_definitions
    e = (2 * n).exp()
    o = (e - 1) / (e + 1)
    o.label = "o"

    # -------------
    # Backward pass
    # -------------
    o.backward()

    # Can call `draw_dot(o)` on return value in jupyter notebook
    return [o, x1, w1, x2, w2]


def run_value_example_tanh_torch() -> list[torch.Tensor]:
    x1 = torch.Tensor([2.0]).double()
    x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double()
    x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double()
    w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double()
    w2.requires_grad = True
    b = torch.Tensor([6.8813735870195432]).double()
    b.requires_grad = True
    n = x1 * w1 + x2 * w2 + b
    o = torch.tanh(n)

    o.backward()
    return [o, x1, w1, x2, w2]


def run_neuron_example() -> tuple[Value, list[Value], list[Value]]:
    # Mirrors run_value_example_tanh() above but using ReLU (via nonlin=True)

    # Input data
    x = [Value(2.0, label="x1"), Value(0.0, label="x2")]

    # Network with manual weights/biases
    n = Neuron(nin=2, nonlin=True)
    n.w = [Value(-3.0, label="w1"), Value(1.0, label="w2")]
    n.b = Value(6.8813735870195432, label="b")

    # Forward pass
    o = n(x)
    o.label = "o"

    # Backward pass
    o.backward()

    # Can call `draw_dot(o)` on return value in jupyter notebook
    return (o, x, n.w)


def run_neuron_example_torch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Mirrors run_value_example_tanh_torch() above but using ReLU

    # Input data
    x = torch.tensor([2.0, 0.0], dtype=torch.double)
    x.requires_grad = True

    # Network with manual weights/biases
    n = nn.Linear(in_features=2, out_features=1, bias=True, dtype=torch.double)
    n.weight.data = torch.tensor([[-3.0, 1.0]], dtype=torch.double)  # Shape 1x2 (out x in feat)
    n.bias.data = torch.tensor([6.8813735870195432], dtype=torch.double)
    n.weight.requires_grad = True
    n.bias.requires_grad = True

    # Forward pass (using nonlin=True in run_neuron_example() so need relu here)
    o = n(x).relu()

    # Backward pass
    o.backward()
    return (o, x, n.weight)


def run_layer_example() -> tuple[Value, list[Value], list[Value], list[Value], list[Value]]:
    # Mirrors run_neuron_example() but uses two output neurons
    # Dropping labels, but can still visualize with draw_dot()

    # Input data
    x = [Value(2.0), Value(0.0)]

    # Network with manual weights/biases
    n = Layer(nin=2, nout=3, nonlin=True)
    n.neurons[0].w = [Value(-3.0), Value(1.0)]
    n.neurons[1].w = [Value(-2.0), Value(2.0)]
    n.neurons[2].w = [Value(-1.0), Value(3.0)]
    n.neurons[0].b = Value(1.0)
    n.neurons[1].b = Value(2.0)
    n.neurons[2].b = Value(3.0)

    # Forward pass
    o: Value = sum(n(x))  # pyright: ignore[reportAssignmentType]

    # Backward pass
    o.backward()

    # Can call `draw_dot(o)` on return value in jupyter notebook
    return (o, x, n.neurons[0].w, n.neurons[1].w, n.neurons[2].w)


def run_layer_example_torch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Mirrors run_value_example_tanh_torch() above but using ReLU

    # Input data
    x = torch.tensor([2.0, 0.0], dtype=torch.double)
    x.requires_grad = True

    # Network with manual weights/biases
    n = nn.Linear(in_features=2, out_features=3, bias=True, dtype=torch.double)
    n.weight.data = torch.tensor(
        [
            [-3.0, 1.0],
            [-2.0, 2.0],
            [-1.0, 3.0],
        ],
        dtype=torch.double,
    )
    n.bias.data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
    n.weight.requires_grad = True
    n.bias.requires_grad = True

    # Forward pass (using nonlin=True in run_layer_example() so need relu here)
    o = n(x).relu().sum()

    # Backward pass
    o.backward()
    return (o, x, n.weight)


def test_micrograd_value_tanh():
    o_mg, x1_mg, w1_mg, x2_mg, w2_mg = run_value_example_tanh()
    o_pt, x1_pt, w1_pt, x2_pt, w2_pt = run_value_example_tanh_torch()

    tol = 1e-6
    # forward pass went well
    assert abs(o_mg.data - o_pt.data.item()) < tol
    # backward pass went well
    assert abs(x1_mg.grad - x1_pt.grad.item()) < tol
    assert abs(w1_mg.grad - w1_pt.grad.item()) < tol
    assert abs(x2_mg.grad - x2_pt.grad.item()) < tol
    assert abs(w2_mg.grad - w2_pt.grad.item()) < tol


def test_micrograd_value_exp():
    o_mg, x1_mg, w1_mg, x2_mg, w2_mg = run_value_example_tanh_with_exp()
    o_pt, x1_pt, w1_pt, x2_pt, w2_pt = run_value_example_tanh_torch()

    tol = 1e-6
    # forward pass went well
    assert abs(o_mg.data - o_pt.data.item()) < tol
    # backward pass went well
    assert abs(x1_mg.grad - x1_pt.grad.item()) < tol
    assert abs(w1_mg.grad - w1_pt.grad.item()) < tol
    assert abs(x2_mg.grad - x2_pt.grad.item()) < tol
    assert abs(w2_mg.grad - w2_pt.grad.item()) < tol


def test_micrograd_neuron():
    o_mg, x_mg, w_mg = run_neuron_example()
    o_pt, x_pt, w_pt = run_neuron_example_torch()

    tol = 1e-6
    # forward pass went well
    assert abs(o_mg.data - o_pt.data.item()) < tol
    # backward pass went well
    # grad attribute for pytorch only exists for whole tensor, not slice, so:
    #   x_pt.grad[0] is valid, x_pt[0].grad is None
    # weight tensor from pytorch is 2D (out feat x in feat), so need to select first row
    assert abs(x_mg[0].grad - x_pt.grad[0].item()) < tol
    assert abs(w_mg[0].grad - w_pt.grad[0, 0].item()) < tol
    assert abs(x_mg[1].grad - x_pt.grad[1].item()) < tol
    assert abs(w_mg[1].grad - w_pt.grad[0, 1].item()) < tol


def test_micrograd_layer():
    o_mg, x_mg, n1w_mg, n2w_mg, n3w_mg = run_layer_example()
    o_pt, x_pt, w_pt = run_layer_example_torch()

    tol = 1e-6
    # forward pass went well
    assert abs(o_mg.data - o_pt.data.item()) < tol
    # backward pass went well
    # grad attribute for pytorch only exists for whole tensor, not slice, so:
    #   x_pt.grad[0] is valid, x_pt[0].grad is None
    assert abs(x_mg[0].grad - x_pt.grad[0].item()) < tol
    assert abs(x_mg[1].grad - x_pt.grad[1].item()) < tol
    assert abs(n1w_mg[0].grad - w_pt.grad[0, 0].item()) < tol
    assert abs(n1w_mg[1].grad - w_pt.grad[0, 1].item()) < tol
    assert abs(n2w_mg[0].grad - w_pt.grad[1, 0].item()) < tol
    assert abs(n2w_mg[1].grad - w_pt.grad[1, 1].item()) < tol
    assert abs(n3w_mg[0].grad - w_pt.grad[2, 0].item()) < tol
    assert abs(n3w_mg[1].grad - w_pt.grad[2, 1].item()) < tol


def test_micrograd_simple():  # Originally test_sanity_check()
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()


def test_micrograd_detailed():  # Originally test_more_ops()
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
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
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
