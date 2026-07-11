# Softmax and negative log likelihood

*Month 1, Week 2 · makemore-bigram*

## The idea in one paragraph
Softmax turns raw scores (logits) into a probability distribution: exponentiate, then normalize. It is a soft version of max, since a hard max would put probability 1 on the biggest logit and 0 elsewhere. Negative log likelihood is the loss on top: take the probability the model assigned to the correct answer, take its log, negate, average over examples. The log turns a product of probabilities into a sum and avoids underflow. In practice both are fused into one op, F.cross_entropy, which takes logits directly and applies the subtract-max trick in log-space so exp() cannot overflow.

## What confused me / what clicked
In the Week 2 exercises, after loss.backward() on the softmax + NLL example, the four logit gradients printed as [0.0418, 0.8390, 0.0057, -0.8865]. At the time they looked arbitrary, except that they summed to 0. After learning softmax, I computed softmax([0.0, 3.0, -2.0, 1.0]) and got [0.0418, 0.8390, 0.0057, 0.1135], which sums to 1 as a probability distribution should. The gradients are exactly those probabilities, except at dim 3, the correct label, where 1 was subtracted: 0.1135 - 1 = -0.8865. That subtraction is also why the gradients sum to 0 while the probabilities sum to 1.

The rule: gradient = p - 1 for the correct class, p for every other class (probs - onehot).

Why this is fair: when the correct class has low probability, p - 1 is a large negative number, so the update pushes it up strongly. When the correct class already has high probability, p - 1 is near 0 and it barely moves. A wrong class the model is confident about gets a large positive gradient and is pushed down hard by loss minimization. Blame is distributed in proportion to confidence.

## The math (minimal)
- softmax(logits) = exp(logits) / sum(exp(logits))
- loss = -log(probs[correct]), averaged over examples
- Gradient of the loss w.r.t. logits: probs - onehot(correct label)

## Where it shows up later
- Cross-entropy is the pretraining loss for every language model through Month 20; softmax also reappears inside attention.
- Month 2 (backprop ninja): derive by hand why the softmax-then-NLL chain collapses to probs - onehot. This note verifies the pattern; the derivation proves it.
