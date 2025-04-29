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
