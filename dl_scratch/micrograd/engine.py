# Modified from https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
#   - Add types
#   - Add / clean up comments
#   - Add tanh from https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_first_half_roughly.ipynb
import math
from typing import Callable, Union


class Value:
    """Stores a single scalar value and its gradient, with support for .backward() on final output.

    Big picture:
    - This Value is like torch.Tensor (but only supports scalars, not tensors / numpy arrays)
    - The input is stored in `data`, the gradient will be computed in `grad`

    Public members:
    - self.data: input scalar value
    - self.grad: calculated scalar derivative, e.g. dL/dValue if calling L.backward() on loss L
    - self.label: node string for debugging only (e.g. graphviz)

    Private members (for autograd):
    - self._backward: track backward function based on the operation
    - self._prev: track children nodes / previous nodes to propagate gradients
    - self._op: op string for debugging only (e.g. graphviz)

    Backward basics:
    - Let final node output = `L` (for loss)
        - Assume we called L.backward()
        - We want to compute dL/d<node> for every node in compute graph
    - Let current node output = `z`, e.g. z = a+b for __add__
        - Need to apply chain rule, e.g. dL/da = dL/dz * dz/da
        - dL/dz is incoming gradient (passed down from parent)
        - Use local derivative dz/da and dz/db to calculate dL/da, dL/db
    - Need to sum gradient so if value appears twice we add grad not overwrite
        - See https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=4948s
        - Or https://en.wikipedia.org/wiki/Chain_rule#General_rule:_Vector-valued_functions_with_multiple_inputs
    """

    def __init__(
        self, data: int | float, _children: tuple["Value", ...] = (), _op: str = "", label: str = ""
    ):
        self.data: int | float = data
        self.grad: float = 0.0

        # Internal variables used for autograd graph construction
        self._backward: Callable = lambda: None
        self._prev: set[Value] = set(_children)

        # Strings for labeling nodes for visualizing compute graph w/ `draw_dot()`
        self._op: str = _op
        self.label: str = label

    def __add__(self, other: Union[int, float, "Value"]):
        # Cast other to support expr like `x + 2` versus only `x + Value(2)`
        other = other if isinstance(other, Value) else Value(other)

        # Compute z = a + b
        out = Value(data=self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            # dL/dz = out.grad
            # dz/da = self.grad = 1
            # dz/db = other.grad = 1
            self.grad += out.grad  # dz/da * dL/dz
            other.grad += out.grad  # dz/db * dL/dz

        out._backward = _backward
        return out

    def __mul__(self, other: Union[int, float, "Value"]):
        # Cast other to support expr like `x * 2` versus only `x * Value(2)`
        other = other if isinstance(other, Value) else Value(other)

        # Compute z = a * b
        out = Value(data=self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            # dL/dz = out.grad
            # dz/da = self.grad = b = other.data
            # dz/db = other.grad = a = self.data
            self.grad += other.data * out.grad  # dz/da * dL/dz
            other.grad += self.data * out.grad  # dz/db * dL/dz

        out._backward = _backward
        return out

    def __pow__(self, other: Union[int, float, "Value"]):
        # Other must be int/float, e.g. x**2.5, not x**y where y is Value (req. different gradient)
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        # Compute z = a**b where b is int/float
        out = Value(data=self.data**other, _children=(self,), _op=f"**{other}")

        def _backward():
            # dL/dz = out.grad
            # dz/da = b * a**(b-1) = other * self.data**(other-1)  where other=int/float not Value
            self.grad += (other * self.data ** (other - 1)) * out.grad  # dz/da * dL/dz

        out._backward = _backward
        return out

    def tanh(self):
        # Compute z = tanh(x) directly, using (e^(2x) - 1) / (e^(2x) + 1)
        #   - See https://en.wikipedia.org/wiki/Hyperbolic_functions#Exponential_definitions
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(data=t, _children=(self,), _op="tanh")

        def _backward():
            # dL/dz = out.grad
            # dz/dx = 1 - tanh(x)**2
            #   - See https://en.wikipedia.org/wiki/Hyperbolic_functions#Derivatives
            self.grad += (1 - t**2) * out.grad  # dz/dx * dL/dz

        out._backward = _backward
        return out

    def exp(self):
        # Compute z = exp(x) directly
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            # dL/dz = out.grad
            # dz/dx = exp(x) = out.data
            # NOTE: Video incorrectly uses = instead of += (fixed here)
            self.grad += out.data * out.grad  # dz/dx * dL/dz

        out._backward = _backward
        return out

    def relu(self):
        # Compute z = ReLU(a)
        out = Value(data=0 if self.data < 0 else self.data, _children=(self,), _op="ReLU")

        def _backward():
            # dL/dz = out.grad
            # dz/da = 1 if out.data > 0 else 0
            self.grad += (out.data > 0) * out.grad  # dz/da * dL/dz

        out._backward = _backward
        return out

    def backward(self):

        # Topological sort
        # - Start with current node (e.g. L for L.backward())
        # - Call build_topo() on all children to append children to topo list first
        # - Then append current node to topo list after children
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Iterate backwards through topological sort to apply chain rule and update gradients
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other: Union[int, float, "Value"]):  # other + self
        # Necessary to support `2 + Value(1)` versus only `Value(1) + 2`
        return self + other

    def __sub__(self, other: Union[int, float, "Value"]):  # self - other
        # Implement with add and negate to re-use (could have slight performance penalty)
        return self + (-other)

    def __rsub__(self, other: Union[int, float, "Value"]):  # other - self
        # Necessary to support `2 - Value(1)` versus only `Value(1) - 2`
        return other + (-self)

    def __rmul__(self, other: Union[int, float, "Value"]):  # other * self
        # Necessary to support `2 * Value(1)`` versus only `Value(1) * 2`
        return self * other

    def __truediv__(self, other: Union[int, float, "Value"]):  # self / other
        # Implement with pow to be more general (could have slight performance penalty)
        return self * other**-1

    def __rtruediv__(self, other: Union[int, float, "Value"]):  # other / self
        # Necessary to support `2/Value(1)` versus only `Value(1)/2`
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
