DEBUG = 0

import torch
import numpy as np

"""
1. Documentation reading
A big part of deep learning (and learning to code in general) is getting familiar with the documentation of a certain framework you're using. We'll be using the PyTorch documentation a lot throughout the rest of this course. So I'd recommend spending 10-minutes reading the following (it's okay if you don't get some things for now, the focus is not yet full understanding, it's awareness):

The documentation on torch.Tensor.
The documentation on torch.cuda.
"""

"""
2. Create a random tensor with shape (7, 7).
"""
rand_tensor = torch.rand(7, 7)
if DEBUG: print(rand_tensor)


"""
3. Perform a matrix multiplication on the tensor from 2 with another random tensor with shape (1, 7)
(hint: you may have to transpose the second tensor).
"""

rand_tensor2 = torch.rand(1, 7)
rand_tensor2 = rand_tensor2.T
res = rand_tensor.matmul(rand_tensor2)
if DEBUG: print(res)


"""
4. Set the random seed to 0 and do 2 & 3 over again.
The output should be:

(tensor([[1.8542],
         [1.9611],
         [2.2884],
         [3.0481],
         [1.7067],
         [2.5290],
         [1.7989]]), torch.Size([7, 1]))
"""

SEED = 0
torch.manual_seed(SEED)
rand_tensor = torch.rand(7, 7)

torch.manual_seed(SEED)
rand_tensor2 = torch.rand(1, 7)
rand_tensor2 = rand_tensor2.T

res = rand_tensor.matmul(rand_tensor2)
if DEBUG: print("res", res, res.shape)


"""
5. Speaking of random seeds, we saw how to set it with torch.manual_seed() 
but is there a GPU equivalent? 

(hint: you'll need to look into the documentation for torch.cuda for this one)
"""

device = "gpu" if torch.cuda.is_available() else "cpu"
RANDOMSEED = 1234
torch.cuda.manual_seed(RANDOMSEED)
tensor_gpu = torch.rand(5, 5).to(device)
if DEBUG: print(tensor_gpu, tensor_gpu.device, tensor_gpu.shape)


"""
6. Create two random tensors of shape (2, 3) and send them both to the GPU 
(you'll need access to a GPU for this). 

Set torch.manual_seed(1234) when creating the tensors 
(this doesn't have to be the GPU random seed). The output should be something like:

Device: cuda
(tensor([[0.0290, 0.4019, 0.2598],
         [0.3666, 0.0583, 0.7006]], device='cuda:0'),
 tensor([[0.0518, 0.4681, 0.6738],
         [0.3315, 0.7837, 0.5631]], device='cuda:0'))
"""

RANDOM_SEED = 1234
torch.manual_seed(RANDOM_SEED)
rand1 = torch.rand(2, 3)
rand1.to(device)

# torch.manual_seed(RANDOM_SEED)
rand2 = torch.rand(2, 3)
rand2.to(device)
if DEBUG:
    print(rand1)
    print(rand2)

"""
7. Perform a matrix multiplication on the tensors you created in 6 
(again, you may have to adjust the shapes of one of the tensors).

The output should look like:

(tensor([[0.3647, 0.4709],
         [0.5184, 0.5617]], device='cuda:0'), torch.Size([2, 2]))
"""

rand2 = rand2.T
res = rand1.matmul(rand2)
if DEBUG: print(res, res.device, res.shape)

"""
8. Find the maximum and minimum values of the output of 7.

"""

max = torch.max(res)
if DEBUG: print("max", max)

min = torch.min(res)
if DEBUG: print("min", min)


"""
9. Find the maximum and minimum index values of the output of 7.

"""

argmax = torch.argmax(res)

argmin = torch.argmin(res)

if DEBUG: print("argmax:", argmax)
if DEBUG: print("argmin:", argmin)


"""
10. Make a random tensor with shape (1, 1, 1, 10) 
and then create a new tensor with all the 1 dimensions removed to be left with a tensor of shape (10). 
Set the seed to 7 when you create it and print out the first tensor and it's shape as well as the second tensor and it's shape.

The output should look like:

tensor([[[[0.5349, 0.1988, 0.6592, 0.6569, 0.2328, 0.4251, 0.2071, 0.6297,
           0.3653, 0.8513]]]]) torch.Size([1, 1, 1, 10])
tensor([0.5349, 0.1988, 0.6592, 0.6569, 0.2328, 0.4251, 0.2071, 0.6297, 0.3653,
        0.8513]) torch.Size([10])
"""

SEED = 7
torch.manual_seed(SEED)
res2 = torch.rand(1, 1, 1, 10)
print(res2, res2.shape)

res2 = torch.squeeze(res2)
print(res2, res2.shape)
