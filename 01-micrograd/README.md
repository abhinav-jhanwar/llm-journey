# micrograd

## What I built
A micro implementation of PyTorch's autograd from scratch in pure Python.

Followed [tutorial from Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0).

Value class (the scalar equivalent of PyTorch's Tensor), backpropagation using topological sort, and a small MLP trained with it.

## What I learned
- Basics of how neural networks work and how they are trained
- How backprop works. More in [`notes/`](../notes/01-backprop-explained.md).
- Why activation functions must be nonlinear: linear functions cannot represent most real patterns, and stacking linear layers adds nothing because they just compose into another linear function.
- How a loss function works and how to minimize it to bring the outputs close to the targets.
- ReLU vs tanh. More in [`notes/`](../notes/02-tanh-vs-relu.md).
- Non-leaf tensors in PyTorch need a `retain_grad()` call to keep their gradient.
- Symmetric Derivative: `(f(x + h) - f(x - h)) / (2h)` - more accurate than the one-sided version (error shrinks with h² instead of h), which is why it's used for gradient checking

## How to run
### Dependencies
- NumPy
- Matplotlib
- PyTorch
- Graphviz

### Commands
For the notebook:  
`open micrograd.ipynb`

For the PyTorch autograd verification script:  
`python verify_pytorch.py`

## Backlog / known limitations
- `Value ** Value` currently unsupported (asserts on non-numeric exponent).
  Full support needs a `log()` op, since d(a^b)/db = a^b * ln(a).
