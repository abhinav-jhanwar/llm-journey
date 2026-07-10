# makemore-bigram

## What I built
A bigram model that generates new names by sampling from bigram probabilities, evaluated with negative log likelihood. Built twice: first by counting bigrams, then as a one-layer neural net trained by gradient descent - both converge to the same model.

Followed [tutorial from Andrej Karpathy](https://www.youtube.com/watch?v=PaCmpygFfXo).

## What I learned
- backward() frees the graph's intermediate buffers after use, hence you can only call it once after a forward pass. In real training you never notice - each step runs a fresh forward pass, building a fresh graph. retain_graph=True is for the rare case of backpropping twice through the same forward.
- For inference (forward pass only), wrap the computation in `with torch.no_grad():` - operations inside build no autograd graph, saving memory and time. Useful when running pre-trained networks without training them. (.detach() is the per-tensor tool instead: it removes one tensor from the graph so gradients stop flowing through that path.)
- Autograd graphs are built fresh during every forward pass and consumed by backward(). Avoid in-place ops on tensors autograd cares about - PyTorch tracks tensor versions and will usually fail loudly with a RuntimeError if an in-place mutation breaks the graph.
- backward(v) computes vᵀJ; scalar loss is v=1.
- torch.multinomial() samples indices according to a probability distribution.
- Tensor broadcasting: align dimensions from the right, then each dimension pair must be equal, or one of them is 1, or one of them is missing.
- Broadcasting can succeed and still be wrong: `P /= P.sum(1)` without `keepdim=True` gives a row vector that broadcasts incorrectly and silently normalizes the wrong way - no error raised. This is a worse failure class than a loud RuntimeError.
- Smoothing: adding a constant to the counts (`N + 1`) avoids zero probabilities and therefore infinite negative log likelihood. The higher the constant, the smoother the distribution; the lower, the more peaked.
- For maximizing likelihood, we take the product of the probabilities and try to maximize it. Since a product of such small values can underflow, we use the log instead, which turns the product into a sum: log(a*b*c) = log(a) + log(b) + log(c). In practice we minimize the average negative log likelihood.
- torch.tensor and torch.Tensor both exist. The difference is that torch.tensor infers the dtype automatically while torch.Tensor returns torch.FloatTensor. Usually advised to use torch.tensor as it also has an option to specify the data type if needed.
- One hot encoding in pytorch can be used to set one value to 1 and all the others to 0.
- tensor matmul operations do not support automatic promotion when working on int and float. So int inputs need to be manually cast to float first.
- softmax is exponentiating then normalizing logits: softmax(logits) = exp(logits) / sum(exp(logits)). Note: it's a soft version of max. A hard max would put probability 1 on the biggest logit and 0 elsewhere.
- We can add penalties to the loss (e.g. `0.01 * (W**2).mean()`) to regularize extreme weights.
- Why counting and gradient descent give the same model: the network's logits are log-counts, so W learns the log of the count matrix. With a one-hot input, `xenc @ W` just selects row x of W - the "network" is the count table, learned instead of counted. Same seed, same sampled names from both versions.
- W² regularization is the gradient-descent twin of count smoothing: smoothing pushes counts toward uniform by adding a constant, regularization pushes W toward zero (uniform logits) by penalizing magnitude. Same knob, two frameworks.
- `W.grad = None` before backward() is the efficient way to zero gradients.

## How to run
### Dependencies
- Matplotlib
- PyTorch

### Data
`names.txt` (committed in this folder) - one name per line, from Karpathy's makemore repo.

### Commands
For the notebook:  
`open bigram.ipynb`

## Backlog / known limitations
None yet.
