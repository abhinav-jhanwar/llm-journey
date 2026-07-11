# makemore-bigram

## What I built
A bigram model that generates new names by sampling from bigram probabilities, evaluated with negative log likelihood. Built twice: first by counting bigrams, then as a one-layer neural net trained by gradient descent - both converge to the same model. Extended in `exercises.ipynb`: a trigram model (context = previous two characters) and proper train/dev/test evaluation.

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
- Train/dev/test split (80/10/10, shuffled with a fixed seed): train fits the weights, dev is for comparing models and tuning hyperparameters, test is touched once at the end. Evaluate without the regularization term (and under no_grad) so losses are comparable across models.
- A model with far fewer parameters than examples (bigram: 729 params vs 180k examples) cannot meaningfully overfit - its train/dev gap is sampling noise, and the sign of a tiny gap carries no meaning. The trigram's small gap (0.019) is real but mild, from sparse context rows.
- Hyperparameter selection discipline: sweep on dev, touch test once with the winner. Margins smaller than your measured noise floor are ties, not wins - report them as such. Regularization only "fights the data" once it's strong enough; below that it mostly smooths rows with little data, which is why it's nearly free here (the model barely overfits to begin with).
- Control one variable at a time: my first sweep drew a different random init per strength (`randn(generator=g)` consumes the generator - it advances state rather than resetting), confounding init with strength. Reseeding inside the loop made dev nll cleanly monotonic and dissolved the apparent 0.0003 winner - the 0.003 "margin" was init variance, which turned out to be the sweep's dominant noise source. Seed-once-outside = reproducible variety; seed-per-iteration = controlled comparison. Same API, opposite semantics.
- One-hot @ W is row selection performed the expensive way: the matmul materializes a huge mostly-zero matrix and does 729 multiply-adds per example just to copy out one row. `W[xs]` does the same selection directly. Same seed gives an identical training trajectory up to float dust in the last digits (the summation order of all those zero-products rounds differently than a direct copy). This lookup is called an embedding: one-hot is the conceptual explanation, indexing is the actual implementation (E04).
- `F.cross_entropy(logits, ys)` fuses softmax + pick-target + log + negate + mean into one op. Beyond convenience: numerical safety (subtract-max trick, works in log-space, so raw exp() can't overflow) and efficiency (fused kernel, no intermediate counts/probs tensors, simpler backward). This is why real losses take logits, not probabilities (E05).
- Trigram context trick: encode the previous two characters as a single index (27*27 = 729 rows) and keep predicting one of 27 next characters - the same one-layer machinery scales to longer context by growing the input vocabulary. Cost: one-hot over 729 classes is huge and mostly zeros, which is exactly what indexing into W directly (exercise E04) removes.

## Results

| Model | Train | Dev | Test |
|---|---|---|---|
| Bigram (1-layer NN) | 2.460 | 2.459 | 2.464 |
| Trigram (1-layer NN) | 2.234 | 2.254 | 2.254 |

The trigram's ~0.2 improvement holds on dev and test, so it's real learning, not memorization.

Regularization sweep (E03, trigram, inits pinned per strength): dev nll is monotonic in strength - 0 through 0.001 are identical to ~0.00003, 0.1+ clearly hurts. The unpinned sweep's apparent winner (0.0003, by 0.003) evaporated once inits were controlled: it was init luck. Test nll 2.2548, consistent with dev.

## How to run
### Dependencies
- Matplotlib
- PyTorch

### Data
`names.txt` (committed in this folder) - one name per line, from Karpathy's makemore repo.

### Commands
For the notebook:  
`open bigram.ipynb`

For the trigram + train/dev/test exercises:  
`open exercises.ipynb`

## Backlog / known limitations
None yet.
