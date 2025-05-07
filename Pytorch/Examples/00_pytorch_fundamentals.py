DEBUG = 0
print("Hello World! I'm excited to learn PyTorch!")

## 00. PyTorch Fundamentals

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if (DEBUG): print(torch.__version__)

# Introduction to Tensors
# Main building block for ML or deep learning

# Scalar
# Tensors are created using torch.Tensor()
scalar = torch.tensor(7)
print(scalar)

# Some attributes about a scalar.
if DEBUG: print("python integer:", scalar.item())

# Vector
vector = torch.tensor([7, 7])
if DEBUG: print("This is a vector:", vector)
if DEBUG: print("vector shape:", vector.shape)

# MATRIX
MATRIX = torch.tensor([[7, 8], 
                       [9, 10]])

if DEBUG: print(MATRIX)
if DEBUG: print(MATRIX.ndim) # number of dimensions (2 sets of brackets)
if DEBUG: print(MATRIX[0])
if DEBUG: print(MATRIX[1])

if DEBUG: print(MATRIX.shape) # shape of the matrix (2x2)

# TENSOR
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])

if DEBUG: 
    print(TENSOR)
    print(TENSOR.ndim)
    print(TENSOR.shape) # returns 1, 3, 3 (one 3x3 array)
    print(TENSOR[0])

TENSOR2 = torch.tensor([[[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]],
                         
                         [[13, 14, 15],
                         [17, 19, 20],
                         [20, 21, 22]]])
if DEBUG:
    print(TENSOR2)
    print(TENSOR2.ndim)
    print(TENSOR2.shape)

# Random TENSORS
"""
Why random tensors?
Random tensors are important because the way many neural networks learn
is that they start with tensors full of random numbers
and then adjust those random numbers to better represent the data.

Start with random numbers -> Look at data -> Update random numbers -> Look at data
-> Update Random numbers
"""

# Create a random tensor (size of 2x10x4)
random = torch.rand(2, 10, 4)
if DEBUG:
    print(random)
    print(random.ndim)
    print(random.shape)

# Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(224, 224, 3)) # height, width, color channels (R, G, B)
if DEBUG:
    print(random_image_size_tensor)
    print(random_image_size_tensor.shape)
    print(random_image_size_tensor.ndim)

# Create a Tensor of Zeroes and Ones
zeros = torch.zeros(10, 1)

if DEBUG: print(zeros)

# Create a tensor of all ones
ones = torch.ones(3, 4)
if DEBUG:
    print(ones) 
    print(ones.dtype)

# Create a range of tensors and tensors-like
# use torch.range()
if DEBUG: print(torch.arange(0, 11))
one_to_thousand = torch.arange(start=0, end=1001, step=10)
if DEBUG: print(one_to_thousand)

# Creating tensors like replicate shape of another tensor but not define what that shape might be
hundred_zeros = torch.zeros_like(input=one_to_thousand)
if DEBUG: print(hundred_zeros)

hundred_ones = torch.ones_like(one_to_thousand)
if DEBUG: print(hundred_ones)


float_32_tensor = torch.tensor([3.0, 6.0, 9.0], 
                               dtype=None, # What datatype the tensor is (e.g. float32, int16, etc.)
                               device=None, # What device is your tensor on (Cuda vs CPU, etc.)
                               requires_grad=False) # Whether or not to track the gradients of a tensor when it'sd going through numerical calculations
if DEBUG:
    print(float_32_tensor)
    print(float_32_tensor.dtype)

""" 
Tensor datatypes is one of the 3 big issues with PyTorch and deep learning:
1. Tensors not right datatype (int vs floating point number) - to get datatype from tensor, can use tensor.dtype
2. Tensors not right shape (If the shape doesn't align with multiple tensors) - to get shape from tensor, can use tensor.shape
3. Tensors not on the right device (If one tensor lives on a GPU and another tensor lives on the CPU) - to get device from a tensor, can use tensor.device
"""

float_16_tensor = float_32_tensor.type(torch.float16)
if DEBUG:
    print(float_16_tensor)
    print(float_16_tensor.dtype)

multiplied_tensors = float_16_tensor * float_32_tensor
if DEBUG:
    print(multiplied_tensors)
    print(multiplied_tensors.dtype)


int_32_tensor = torch.tensor([3, 6, 9], 
                             dtype=torch.int32)

if DEBUG: print(int_32_tensor)

float_and_int_tensor = float_32_tensor * int_32_tensor
if DEBUG:
    print(float_and_int_tensor)
    print(float_and_int_tensor.dtype)


# Creating a new trensor
some_tensor = torch.rand(size=(3, 4))
print(some_tensor)
print(f"the type of some_tensor: {some_tensor.dtype}")
print(f"The shape of some_tensor: {some_tensor.shape}") # can also use tensor.size()
print(f"The device of some_tensor: {some_tensor.device}")

# Maninpulating tensors
"""
NN are comprised with a lot of mathematical operations
Tensor operations include:
* Addition
* Subtraction
* Multiplication (element-wise)
* Division
* Matrix Multiplication
"""

tensor = torch.tensor([1, 2, 3])
if DEBUG:
    print("add 10: ", tensor + 10)
    print("add 100: ", tensor + 100)

    print("multiply 10:", tensor * 10)
    print("multiply 100:", tensor * 100)

    print("subtract 10:", tensor - 10)

    # Try the built in torch operations
    print("torch mult", torch.mul(tensor, 10))

# matrix multiplication
tensor2 = torch.tensor([4, 5, 6])
print("matrix multiplication", torch.matmul(tensor, tensor2))

"""
Two main ways of performing multiplication in NN and deep learning
* Element-wise multiplication
* Matrix multiplication (dot product) - possibly the most common tensor operation in NN
"""

if DEBUG:

    print(tensor,  "*", tensor)
    print(f"Equals: {tensor * tensor}")

"""
torch.matmul
"""

print(torch.matmul(tensor, tensor)) # why 14 instead of [1, 4, 9]?

# matrix multiplication by hand:
# 1 * 1 + 2 * 2 + 3 * 3 = 14
value = 0
for i in range(len(tensor)):
    value += tensor[i] * tensor[i]
if DEBUG: print("by hand:", value)

# However, built in matmul function is MUCH more efficient

"""
One of the most common errors in deep learning are shape errors.
There are two main rules that perform matrix multiplication needs to satisfy:
1. The inner dimension must match
(3, 2) @ (3, 2) won't work
(2, 3) @ (3, 2) WILL work
(3, 2) @ (2, 3) will work

The resulting matrix has the shape of the outer dimensions
"""

if DEBUG: print(tensor @ tensor)

rand_tensor = torch.rand(5, 10)
rand_tensor2 = torch.rand(10, 9)

if DEBUG: print(torch.matmul(rand_tensor, rand_tensor2))

# Shapes for MM
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])
if DEBUG:
    print("shape of tensor_A", tensor_A.shape)
    print("shape of tensor_B", tensor_B.shape)
    #print("tensor_A and tensor_b matmul:", torch.mm(tensor_A, tensor_B)) # torch.mm is the same as torch.matmul (it's an alais for matmul)

    # reshapeing the tensors
    """
    To fix the tensor shape issue, we can manipulate the shape of one of our tensors using transpose

    Transpose switchs the axes or dimensions of a given tensors
    """

    print("NOW mm the two:", torch.mm(tensor_A, tensor_B.T))
    print("Is this different?", torch.mm(tensor_A.T, tensor_B))

    """
    Finding the min, max, mean, sum, etc (tensor aggregation)
    """

    x = torch.arange(0, 101, 10)
    print("x", x)

    print("min:", torch.min(x))
    print("max:", torch.max(x))
    print("mean:", torch.mean(x.type(torch.float32))) # dtype error requires tensor of float32 datatype to work
    print("sum:", torch.sum(x))

    """
    find positional min max or argmin or argmax (which index does the min and max values occur at?)

    Useful when you want to use the softmax function
    """

    print("argmin:", torch.argmin(x)) # OR
    print(x.argmin())
    print("argmax:", torch.argmax(x)) # OR
    print(x.argmax())


"""
Reshaping, stacking, squeezing, and unsqueezing tensors
Reshaping - fixes common shape mismatch error for tensors. Reshapes input tensor to defined shape
View - Return a view of an input tensor of certain shape but keep the same memory ???
Stacking - combine multiple tensors on top of each other (vstack) or side by stack (hstack)
Squeeze - removes all `1` dimensions from a tensor
Unsqueeze - adds a `1` dimension to a target tensor
Permute - Return a view of the input with dimensions permuted (Swapped) in a certain way
"""
if DEBUG:
    x = torch.arange(1, 11)
    print(x, x.shape)

    # add an extra dimension
    x_reshaped = x.reshape(5, 2)  # has to be compatible with original size
    print(x_reshaped)

    # Change the view
    z = x.view(1, 10)
    print(z, z.shape)

    # Changing z changes x (because a view of a tensor shares the same memory as the original)
    z[:, 0] = 5
    print(z, x)

# Stack tensors on top of each other
if DEBUG:
    x_stacked = torch.stack([x, x, x, x], dim=0)
    print(x_stacked)
    x_stacked = torch.stack([x, x, x, x], dim=1)
    print(x_stacked)
    print("hstack", torch.hstack((x, x, x, x)))
    print("vstack:", torch.vstack((x, x, x, x)))


    # Squeeze and unsqueeze
    x_reshaped = x_reshaped.reshape(1, 1, 10)
if DEBUG:

    print("original x_reshaped:", x_reshaped)
    print("shape of original reshaped:", x_reshaped.shape)

    print("squeeze:", torch.squeeze(x_reshaped)) # removes all single dimensions from a target tensor
    print("shape of squeezed:", torch.squeeze(x_reshaped).shape)

"""
torch.unsqueeze - adds a single dimension to a target tensor at a specific dimension
"""

if DEBUG:
    print(x_reshaped.unsqueeze(dim=3))
    print(x_reshaped.unsqueeze(dim=3).shape)


"""
torch.permute - rearranges the dimensions of a target tensor in a specified order
"""

if DEBUG:
    print("permuted:", torch.permute(x_reshaped, (2, 1, 0))) # rearrange dimensions
x_original = torch.rand(size=(224, 224, 3)) # height, width, color channels
if DEBUG:
    print("x_original", x_original, x_original.shape)

# permute the original tensor to rearrange the axis (or dim) order
x_permuted = x_original.permute(2, 0, 1) # color, height, width
if DEBUG:
    print("x_permuted:", x_permuted, x_permuted.shape)

x_original[0, 0, 0] = 99999
if DEBUG:
    print("x_original after permuting:", x_original[0, 0, 0])
    print("x_permuted after permuting:", x_permuted[0, 0, 0]) # They share memory with the tensor. Same as view. Same value gets copied to x_permuted



# Indexing (selecting data from tensors)
"""
Indexing with PyTorch is similar to indexing with NumPy
"""
if DEBUG:
    # Creating a tensor
    x = torch.arange(1, 10).reshape(1, 3, 3)
    print("x:", x, x.shape)

    """
    Let's index on our new tensor
    """
    print("x[0]:", x[0], x.shape)
    print("x[0][0]:", x[0][0])
    print("x[0][0][2]", x[0][0][2])
    print("9:", x[0][2][2])

    # You can also use ":" to select "all" of a target dimension:
    print(x[:, 0]) # ALL of the 0'th dimension but index 0

    print(x[:, :, 1]) # All elements in 0th and 1st dimension, and only element at index 1 in 2nd dimension

    # Get all values of the 0th dimension but only the 1 index value of the 1st and 2nd dimension
    print(x[:, 1, 1])

    # Get index 0 of 0th and 1st dimension and all values of 2nd dimension 
    print(x[0, 0, :])

    # Index on x to return 9
    print(x[:, 2, 2]) 

    print(x[:, :, 2]) # index on x to return 3, 6, 9

## PyTorch rensors and NumPy
"""
NumPy  is a popular scientific Python numerical computing library

Because of this, PyTorch has functionality to interact with it.

* Data in NumPy array, want in PyTorch tensor -> "torch.from_numpy(ndarray) -> torch tensor
* PyTorch tensor -> NumPy -> "torch.Tensor.numpy()"
"""
if DEBUG:
    # NumPy array to Tensor
    array = np.arange(1.0, 8.0)
    print("array:", array)

    torch_tensor = torch.from_numpy(array).type(torch.float32)
    print("numpy to torch tensor:", torch_tensor, torch_tensor.dtype) # dtype = torch.float64 because numpy's default datatype is float64, but PyTorch's default datatype is float32
    # Warning when converting from numpy -> PyTorch, pytorch reflects numpy's default datatype of float64 unless specified otherwise.

    # array = torch.Tensor.numpy(torch_tensor)
    # print("Back to a numpy array:", array)
    array += 1

    # What happens if we change the value of an array? What will this do to the tensor?

    print("array + 1", array)
    print("tensor", torch_tensor)



    tensor_of_ones = torch.ones(7)
    print(tensor_of_ones)
    numpy_tensor = tensor_of_ones.numpy()
    print("ones tensor:", tensor_of_ones.dtype)
    print("numpy tensor:", numpy_tensor, numpy_tensor.dtype) # float 32 because you switched from tensor -> numpy

    # Change the tensor, what happens to numpy_tensor?
    tensor_of_ones += 1
    print("torch tensor:", tensor_of_ones)
    print("array:", numpy_tensor)



# Reproducability Trying to take the randomness out of random

"""
In short, how a neural network learns:

Start with random nums -> tensor operations -> update random numbers to try and make 
them better representations of the data -> again -> again -> again...

However, when you're trying to reproduce, you need to get rid of the randomness
"""

print(torch.rand(3, 3))


"""
To reduce the randomness in NN and PyTorch, comes the concept of a random seed

Essentially what the random seed does is flavor the randomness.
Random is not true randomness. Computers are fundamentally deterministic. They run the same steps over and over again. 
"""

random_tensor_A = torch.rand(5, 5)
random_tensor_B = torch.rand(5, 5)
print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

# Let's make some random but reproducable tensors
# Set the random set
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(10, 10)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(10, 10)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)

