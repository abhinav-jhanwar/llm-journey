# tanh vs ReLU explained

*Month 1, Week 1 · micrograd*

## The idea in one paragraph
For activation functions, we need to use nonlinear functions as linear functions are not adequate to represent real patterns and stacking layers would give us nothing as linear functions will just compose into linear functions so we would not be able to train the neural network.

## What confused me / what clicked
- Initially I did not understand why nonlinear functions would not work but it made sense looking at the graphs and how they would work that the training would just be dead on linear functions.
- Why tanh() and ReLU are so popular: Turns out they are simple linear functions so they are easy to use but complex neural networks use more complex functions as well depending on their usecase.
- The hinge in ReLU makes it a non linear function.
- tanh seems to be more reliable in minimizing loss function. With ReLU, if you are not careful with the steps, it would just end up in a dead ReLU with the input always being 0 leading to no further activation or loss minimization.

## The math (minimal)
- Derivative of tanh(`1 - tanh**2`)
- ReLU is 0 if input is less than 0, otherwise it is the input itself.

## Where it shows up later
- These are common nonlinear functions that could come up later in training as activation functions.
