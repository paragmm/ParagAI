"""Neural Network modules built entirely from scalar Value autograd nodes."""

import random
from typing import List
from paragai.engine.value import Value


class Module:
    """Base neural network module."""

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> List[Value]:
        return []


class Neuron(Module):
    """A single artificial neuron with weights and bias."""

    def __init__(self, nin: int, nonlin: bool = True):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: List[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer(Module):
    """A fully connected dense layer of neurons."""

    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: List[Value]) -> List[Value]:
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """Multi-Layer Perceptron neural network."""

    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x: List[Value]) -> List[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

