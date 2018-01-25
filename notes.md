
**Tensor**

A tensor is a matrix that user defines. Numpy arrays can easily be converted to a Tensor

**Autograd**

provides automatic differentiation (see definition) for all operations on Tensors. It is define-by-run
framework, which means that every single iteration can be different.

Automatic differentiation is really just a jumped-up chain rule.
When you implement a function on a computer, you only have a small number of primitive operations available
(e.g. addition, multiplication, logarithm). Any complicated function, like (log2 log⁡2x, over x squared) is just a combination of these simple functions.

In other words, any complicated function ff can be rewritten as the composition of a sequence of primitive functions

**Gradients**

(autograd)
if you perform an operation on the values in your tensor, it gives it a gradient

**Neural Networks**

Neural networks (`nn`) depends on `autograd` (gradients) to define models and differentiate them. An `nn.Module`
contains layers, and a method `foward(input)` that returns the `output`

convnet
It is a simple feed-forward network. It takes the input, feeds it through several layers one after the other,
and then finally gives the output.
?? are input and gradient and parameter synonymous? or what is the difference?

A typical training procedure for a neural network is as follows:
* Define the neural network that has some learnable parameters (or weights)
* Iterate over a dataset of inputs
* Process input through the network
* Compute the loss (how far is the output from being correct)
* Propagate gradients back into the network's parameters
* Update the weights of the network, typically using a simple update rule:
    `weight = weight - learning_rate * gradient`

Programmers may choose to apply a pooling layer (also referred to as downsampling layer). Maxpooling is the most popular option.
This basically takes a filter (normally of size 2x2) and a stride of the same length. It then applies it to the input volume and outputs
the maximum number in every subregion that the filter convolves around.
1 0 2 3
4 6 6 8   (max pooling takes highest #'s from each 2x2 grid)
3 1 1 0    ->  6 8
1 2 2 4        3 4

Define a net
class Net(nn.Module):

in that class you just have to define a forward function. The backward function (where gradients are computed) are
automatically defined using `autograd`. You can use any of the Tensor operations in the forward function
The learnable parameters of a model are returned by `net.parameters()`
?? How does the nn determine learnable parameters??

Recap:
* `torch.Tensor` - A multi-dimensional array.
* `autograd.Variable` - Wraps a Tensor and records the history of operations applied to it. Has the same API as a `Tensor`, with some additions like `backward()`. Also holds the gradient w.r.t. the tensor.
* `nn.Module` - Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading, etc.
* `nn.Parameter` - A kind of Variable, that is automatically registered as a parameter when assigned as an attribute to a `Module`.
* `autograd.Function` - Implements forward and backward definitions of an autograd operation. Every `Variable` operation, creates at least a single `Function` node, that connects to functions that created a `Variable` and encodes its history.

At this point, we covered:
Defining a neural network
Processing inputs and calling backward.
Still Left:
Computing the loss
Updating the weights of the network

**Loss Function**

A loss function takes the (output, target) pair of inputs, and computes a value that estimates
how far away the output is from the target.
A simple loss function is: `nn.MSELoss` which computes the mean-squared error between the input and the target.

**Backprop**

To backpropagate the error all we have to do is to `loss.backward()`. You need to clear the existing gradients through, else gradients will be accumulated to existing gradients.
?? does this mean iterations of gradients can't be combined? only adjusted?

The only thing left to learn is updating the weights of the network

**Update the weights**

The simplest update ruly used in parctice is the Stochastic Gradient Descent(SGD):
`weight = weight - learning_rate * gradient`

`torch.optim` is a package that implements several update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc

**What about data?**

Generally, when you have to deal with image, text, audio or video data, you can use standard python packages that load data into a numpy array. Then you can convert this array into a `torch.*Tensor`

* For images, packages such as Pillow, OpenCV are useful
* For audio, packages such as scipy and librosa
* For text, either raw Python or Cython based loading, or NLTK and SpaCy are useful.

Specifically for `vision`, we have created a package called `torchvision`, that has data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz., `torchvision.datasets` and `torch.utils.data.DataLoader`.

This provides a huge convenience and avoids writing boilerplate code.


To run on amazon:
Best option is instance with G2
there is an instance built for deep learning
https://aws.amazon.com/marketplace/pp/B077GCH38C

rui-tao notes
convert image to numpy array
im_np = np.array
