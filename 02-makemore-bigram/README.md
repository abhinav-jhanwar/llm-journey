# makemore-bigram

## What I built
A bigram model that generates new names by sampling from bigram probabilities, evaluated with negative log likelihood.

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
