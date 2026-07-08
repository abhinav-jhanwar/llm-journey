# Backprop explained

*Month 1, Week 1 · micrograd*

## The idea in one paragraph
Backpropagation walks backwards through all the nodes of a neural net, starting from the output node, and sets the gradient of each one. The gradient of a node tells us how much that node affects the output, which is why it is the derivative of the output with respect to that node. This is exactly the chain rule from calculus. Training then uses these gradients to change the weights and biases toward the desired output.

## What confused me / what clicked
- You do not need to worry about the sign of a value when deciding whether to increase or decrease it to change the output; the sign of the gradient takes care of that.
- During backprop, gradients must be accumulated (`+=`) so that when the same variable is used multiple times, its gradient is adjusted accordingly.
- After backprop, if we are doing another forward pass, we should zero the gradients first.

## The math (minimal)
- Basic rules of calculus are good to know.
- It is just the chain rule for derivatives, so that is the important one.

## Where it shows up later
- In every kind of neural network, whenever it is trained to minimize a loss function.
