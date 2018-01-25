import torch
from torch.autograd import Variable

# create a variable
x = Variable(torch.ones(2, 2), requires_grad=True)
print x

# do an operation of variable
y = x + 2
print y

# y was created as a result of an operation, so it has a `grad_fn`
print y.grad_fn

# more operations
z = y * y * 3
out = z.mean()

print z, out

# Gxradients
out.backward()

# print gradients d(out)/dx

print x.grad
