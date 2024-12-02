# Modified from https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py

import random
from dl_scratch.micrograd.engine import Value


class Module:
    """Base class for all modules with parameters."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    """Single neuron that performs `wx + b` or `ReLU(wx+b)`.

    w and x are a list of Values, e.g. 1D tensor.
    b is a single Value.
    """

    def __init__(self, nin: int, nonlin: bool = True):
        self.w: list[Value] = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b: Value = Value(0)
        self.nonlin: bool = nonlin

    def __call__(self, x: list[Value]) -> Value:
        act: Value = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """Single layer that performs Wx or ReLU(Wx) where W == self.neurons

    W is a list of Neurons == list of list of Values, e.g. 2D tensor.
    x is a list of Values.

    Pass in nonlin=True to use ReLU for all neurons.
    """

    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons: list[Neuron] = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """Multi-layer perceptron that stacks Layer objects.

    layers is a list of Layers with non-linearities (ReLU) for all intermediate layers.
    """

    def __init__(self, nin: int, nouts: list[int]):
        sz = [nin] + nouts
        self.layers: list[Layer] = [
            # Non-linearity for all intermediate layers
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x: list[Value]) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
