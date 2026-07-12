# makemore-mlp

## What I built
A multi-layer perceptron language model (the Bengio 2003 architecture) that predicts the next character from the previous three. Unlike the bigram and trigram count models, each character is mapped to a learned embedding vector, so similar characters can share statistical strength instead of every context being an independent row. This is the first model in the series that learns a representation rather than just a table.

Followed [tutorial from Andrej Karpathy](https://www.youtube.com/watch?v=TCH_1BHY58I).

## Results

| Split | Loss |
|---|---|
| Train | 2.042 |
| Dev | 2.141 |

Dev loss of 2.141 beats the video's ~2.17, from a larger embedding (10 dims) and hidden layer (300 units) with a two-stage learning rate.

## What I learned
- Embeddings are learned lookup vectors: C[X] fetches a small vector per character, and the network trains those vectors along with everything else. Characters that behave similarly drift to nearby vectors, which is the mechanism by which this model beats the disjoint rows of a count-based model.
- A tensor's values are stored as one flat array regardless of shape, and torch.Tensor.view() just reinterprets the layout with no copy, which is why view(-1, 6) is far more efficient than torch.cat() or torch.unbind() for flattening the embedded context: cat allocates a new tensor, view does not.
- Bug I hit and fixed by watching the loss floor: I had `@ W1 + b1` outside the tanh instead of inside it. The model still trained, but its loss plateaued much higher. The lesson is that a misplaced nonlinearity fails silently (no error, just a worse floor), so an unexpectedly high plateau is a signal to check the architecture, not just the learning rate.
- F.cross_entropy is beneficial because it does not create the intermediate exp/probability tensors in memory, so it is much more efficient than manually doing exp then normalize then average negative log likelihood. Its backward pass is also simpler, not just because it is a fused kernel but because analytically the backward pass is simpler to compute.
- Once the loss plateaus at a given learning rate, dropping the rate (learning rate decay) lets it settle further. I used 0.1 for the first half of training and 0.01 for the second.

## How to run
### Dependencies
- PyTorch
- Matplotlib

### Data
`names.txt` (committed in this folder), one name per line, from Karpathy's makemore repo.

### Commands
For the notebook:
`open mlp.ipynb`

## Backlog / known limitations
- Learning rate was set by hand (0.1 then 0.01). The LR-finder sweep is the principled way to pick it; wire it in and record the elbow.
