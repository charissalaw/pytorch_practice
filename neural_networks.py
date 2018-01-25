import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

## Define the network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


print "Neural Network"
net = Net()
print net

print "Parameters"
params = list(net.parameters())
print len(params)
print params[0].size()   # conv1's .weight
# print params[0]

'''
The input to the forward is an autograd.Variable, and so is the output. Note: Expected input size to this net(LeNet) is 32x32.
To use this net on MNIST dataset,please resize the images from the dataset to 32x32.
'''

print "Input/Output"
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print out

## Computing Loss (how far the output was from the target)
output = net(input)
target = Variable(torch.arange(1, 11))   # a dummy target, for example
criterion = nn.MSELoss()

print "Loss"
loss = criterion(output, target)
print loss

'''
So, when we call loss.backward(), the whole graph is differentiated w.r.t. the loss, and all Variables
in the graph will have their .grad Variable accumulated with the gradient.
'''
# For illustration, let us follow a few steps backward:
print loss.grad_fn   # MSELoss
print loss.grad_fn.next_functions[0][0]   #Linear
print loss.grad_fn.next_functions[0][0].next_functions[0][0]    # ReLU

print "Backprop"
# have a look at conv1's bias gradients before and after the backward
net.zero_grad()     # zeroes teh gradient buffers of all parameters

print 'conv1.bias.grad before backward'
print net.conv1.bias.grad

loss.backward()

print 'conv1.bias.grad after backward'
print net.conv1.bias.grad


print "Update the weights"
## using the Stochastic Gradient Descent(SGD)
learning_rate = 0.01  ##NOTE is this a number determined by SGD or is it arbitrary?
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
    print f.data.sub_(f.grad.data * learning_rate)

## or use torch.optim package which has various update rules such as SGD
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
