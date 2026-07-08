# micrograd

## What I built
A micro implementation of PyTorch's autograd from scratch in pure Python

Followed [tutorial from Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0).

Value class(scalar equivalent of PyTorch) implemented, backpropagation using topological sort, a small MLP trained with it.

## What I learned
- Basics of how neural networks work and how they are trained
- How backprop works. More in [`notes/`](../notes/01-backprop-explained.md).
- Why it is important to use nonlinear functions as activation functions, otherwise it is not possible to train a neural net because linear functions will not be able to represent most real patterns and stacking layers will tell us nothing.
- How a loss function works and how to minimize it to get close to ideal values.
- ReLU vs tanh. More in [`notes/`](../notes/02-tanh-vs-relu.md).
- Non-leaf of Tensor in PyTorch need retain_grad() call to retain their gradient.

## How to run
### Dependencies
- Numpy
- Matplotlib
- PyTorch
- Graphviz

### Commands
For the notebook:  
`open micrograd.ipynb`

For the pytorch autograd verification script:  
`python verify_pytorch.py`

## Backlog / known limitations
- `Value ** Value` currently unsupported (asserts on non-numeric exponent).
  Full support needs a `log()` op, since d(a^b)/db = a^b · ln(a).
