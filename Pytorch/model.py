import torch # For all things Pytorch
# contains the neural network layers that your'e going to compose  into our model as well as the parent
# class of the model itself 
import torch.nn as nn # for torch.nn.Module, the parent object for Pytorch models.
# Give us activation functions and max pooling funcs that we'll use to connect the layers
import torch.nn.functional as F # for the activation functions

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black and white), 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
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

"""
Looking over this code, you should be able to spot some structural similarities with the diagram above (not here)
This demonstrates the structure of a typical PyTorch model:
    * It inheritcs from torch.nn.Module - modules may be nested - in fact, even the Conv2d and Linear layer classes
    inherit from torch.nn.Module.
    * A model will have an __init__() function, where it instantiates its layers, and loads any data artifacts it might need
    (e.g., an NLP model might load a vocabulary)
    * A model will have a forward() function. This is where the actual computation happens: An input is passed through the
    network layers and various functions to generate an output / prediction
    * Other than that, you can build your own model class like any other Python class, adding whatever properties and methods
    you need to support your model's computation.

Let's instantiate this object and run a sample input though it
"""


net = LeNet()
print(net) # What does the object tell us about itself?

input = torch.rand(1, 1, 32, 32) # stand-in for a 32x32 black and white image
print("\nImage batch shape:")
print(input.shape)
print("\nRaw Input:")
print(input)

output = net(input) # we don't call forward() directly
print("\nRaw Output:")
print(output)
print(output.shape)

"""
There are a few important things happening above.

First, we instantiate the LeNet class, and we print the net object. A sublass of torch.nn.Module will report the layers
it has created and hteir shapes and parameters. This can provide a handy overview of a model if you want to get the gist
of its processing.

Below that, we create a dummy input representing a 32x32 image with 1 color channel. Normally, you would load an image tile 
and convert it to a tensor of this shape.

You may have notices an extra dimension to our tensor - the batch dimension - Pytorch models assume they are working on
batches of data - for example, a batch of 16 of our image tiles would have the shape (16, 1, 32, 32). Since we are only using 
1 image, we create a batch of 1 with shape (1, 1, 32, 32).

We ask the model for an inferencec by calling it like a function net(input). The output of this call represents the model's
confidence that the input represents a particular digit. (Since this instance of the model hasn't learned anything yet, we shouldn't
expect to see any signal in the output)) Looking at the shape of the output we can see it also has a batch dimension the size of which
should always match the input batch dimension. Had we passed in an input batch of 16 instances, output would have a shape of 16 by 10.

We built a simple model, but our outputs don't mean anything yet. For that, we need to train out data. 
"""