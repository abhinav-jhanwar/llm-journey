# tanh vs ReLU explained

*Month 1, Week 1 · micrograd*

## The idea in one paragraph
Activation functions need to be nonlinear. Linear functions are not adequate to represent real patterns, and stacking layers would give us nothing because linear functions just compose into another linear function, so we would not be able to train the neural network.

## What confused me / what clicked
- Initially I did not understand why linear activations would not work, but it made sense after looking at the graphs and how they behave: training would just be dead with linear functions.
- Why tanh() and ReLU are so popular: they are simple nonlinear functions, cheap to compute, with very simple derivatives. Modern networks mostly use smooth ReLU relatives like GELU/SiLU, but the idea is the same: the cheapest nonlinearity that trains well.
- The hinge in ReLU is what makes it a nonlinear function.
- tanh seems to be more reliable at minimizing the loss function. With ReLU, if you are not careful with the update steps, a neuron's weights and bias can end up where its output is 0 for every input. Its gradient is then 0 as well, so it never updates again: a dead ReLU, contributing no further activation or loss minimization.

## The math (minimal)
- Derivative of tanh: `1 - tanh**2`
- ReLU is 0 if the input is less than 0, otherwise the input itself.

## Where it shows up later
- These are common nonlinear functions that will keep coming up as activation functions in later training work.
