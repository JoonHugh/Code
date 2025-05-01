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

print("shape of tensor_A", tensor_A.shape)
print("shape of tensor_B", tensor_B.shape)
#print("tensor_A and tensor_b matmul:", torch.mm(tensor_A, tensor_B)) # torch.mm is the same as torch.matmul (it's an alais for matmul)

# reshapeing the tensors
"""
To fix the tensor shape issue, we can manipulate the shape of one of our tensors using transpose

Transpose switchs the axes or dimensions of a given tensors
"""

tensor_B = tensor_B.T
print("NOW mm the two:", torch.mm(tensor_A, tensor_B))

