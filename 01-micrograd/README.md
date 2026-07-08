# micrograd

## What I built
A "micro" implementation of autograd in PyTorch from scratch.

Followed [tutorial from Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0).

Value class(scalar equivalent of PyTorch) implemented, backpropagation using topological sort, a small MLP trained with it.

## What I learned
- Basics of how neural networks work and how they are trained
- How backprop works. More in [`notes/`](../notes/01-backprop-explained.md).
- Why it is important to use nonlinear functions as activation functions, otherwise it is not possible to train a neural net because linear functions will not have patters and stacking layers will tell us nothing.
- How a loss function works and how to minimize it to get close to ideal values.
- ReLU vs tanh. More in [`notes/`](../notes).
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
