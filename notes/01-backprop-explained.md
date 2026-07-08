# Backprop explained

*Month 1, Week 1 · micrograd*

## The idea in one paragraph
Backpropagation in neural net is propagating backwards across all the nodes starting the output node to set the gradient for them. The gradient decides how much it effects the output which is why the gradients are derivatives of the output in relation to them. This is exactly the same concept as chain rule in derivatives. This helps in training a neural network to change it's weights and biases to get the desired output.

## What confused me / what clicked
- You do not need to worry about the sign of the value when deciding to increase or decrease it to change the output as the sign of gradient will take care of it.
- During backprop, we need to add the gradients to itself so that when we work multiple times with the same variable, the gradient is adjusted accordingly.
- After backprop, if we are doing another forward pass, we should zero the gradients again.

## The math (minimal)
- Basic rules of calculus would be good to know.
- It is just the chain rule for derivatives, so it's important to know this.

## Where it shows up later
- In all kinds of neural networks when it comes to training them to minimize loss functions.

