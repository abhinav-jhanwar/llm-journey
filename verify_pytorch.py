import math
import torch
from micrograd import Value

def get_micrograd_values():
    x1 = Value(2.0)
    x2 = Value(0.0)
    w1 = Value(-3.0)
    w2 = Value(1.0)
    b = Value(6.8813735870194532)
    n = (x1 * w1) + (x2 * w2) + b
    o = n.tanh()
    o.backward()
    return [x1, x2, w1, w2, b, n, o]

def get_pytorch_values():
    x1 = torch.Tensor([2.0]).double()                ; x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double()                ; x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double()               ; w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double()                ; w2.requires_grad = True
    b = torch.Tensor([6.8813735870194532]).double()  ; b.requires_grad = True
    n = (x1 * w1) + (x2 * w2) + b                    ; n.retain_grad() # pytorch does not retain grad for non leaf nodes by default and requires grad cannot be used
    o = torch.tanh(n)                                ; o.retain_grad()
    o.backward()
    return [x1, x2, w1, w2, b, n, o]

def verify_micrograd_pytorch():
    for m, t in zip(get_micrograd_values(), get_pytorch_values()):
        # Comparing float64 for Value(python default) and float64 for Tensor(pytorch default)
        assert math.isclose(m.data, t.data.item(), rel_tol=1e-6)
        assert math.isclose(m.grad, t.grad.item(), rel_tol=1e-6)
    print("Values match!")

verify_micrograd_pytorch()
