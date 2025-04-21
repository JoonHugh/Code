import torch

# create a 2d tensor with 5 rows and 3 columns that's filled with 0s
z = torch.zeros(5, 3)
print(z)
print(z.dtype) # We can see that the default datatype of a tensor is torch.float32

# What if you wanted integers? You can override

i = torch.ones((5, 3), dtype=torch.int16)
print(i)

# Pytorch tells me the dtype without being asked because it's different from the default. 

# It's common to initialize learning weights randomly, often with a specific seed for the random number generated (PRNG)
# so you can reproduce the same results on subsequent runs.
torch.manual_seed(1729)
r1 = torch.rand(2, 2)
print("A random tensor:")
print(r1)

r2 = torch.rand(2, 2)
print("A different random tensor:")
print(r2) # new values

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print("should be the same as r1, because you reset the seed back to 1729, and you called for the first time")
print(r3) # repeats values of r1 because of re-seed

# Tensor operations are intuitive. You can add, multiply, etc. Operations with scalars are distribued over the tensor.

ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2 # every element is multiplied by 2
print(twos)

threes = ones + twos # addition allowed because shapes are similar
print(threes)
print(threes.shape)

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
#r3 = r1 + r2 # Not allowed because the shapes are not the same

# different types of operations you can do with tensors:
r = torch.rand(2,2) - 0.5 * 2 # values between -1 and 1
print(r)

# common mathematical operations are supported:
print("\nAbsolute value of r:")
print(torch.abs(r))

print("\nDeterminant of r:")
print(torch.det(r))

print("\nSingular value decomposition of r:")
print(torch.svd(r))

print("\nAverage and standard deviation of r:")
print(torch.std_mean(r)) # std, mean

print("\nMaximum value of r:")
print(torch.max(r))



