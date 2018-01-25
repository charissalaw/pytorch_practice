from __future__ import print_function
import torch

# ## uninitialized 5x3 matrix
x = torch.Tensor(5, 3)
print(x)

## randomly initialized matrix
x = torch.rand(5, 3)
print(x)

## print it's size
print(x.size())

## different syntax using y
y = torch.rand(5, 3)
print(x + y)

## giving an output Tensor
## NOTE: following doesn't work
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)
