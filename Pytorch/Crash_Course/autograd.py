import torch

# Simple Recurrent Neural Network or RNN
# 4 tensors:
x = torch.randn(1, 10) # The input
prev_h = torch.randn(1, 20) # Hidden state of the RNN that gives it its memory
# Two sets of learning weights, one for the input and one for the hidden state
W_h = torch.randn(20, 20) # set of learning wts. for hidden state
W_x = torch.randn(20, 10) # set of learning wts. for input

# multiply weights by their respective tensors. mm stands for matrix multiplication
i2h = torch.mm(W_x, x.t()) 
h2h = torch.mm(W_h, prev_h.t())
next_h = i2h + h2h # add the outputs of the two matrix multiplications
next_h = next_h.tanh() # pass result through an activation function in this case, the hyperbolic tangent

loss = next_h.sum() # Finally, compute the loss of this output.

# A loss is the difference between the correct putout and the actual prediciton of our model
print("input:", x)
print("hidden state of RNN:", prev_h)
print("wt for hidden state:", W_h)
print("wt for input:", W_x)

print("\nCalcuated loss:", loss)

"""
We've taken the training input, run it through a model, gotten an output, and determined the loss
This is the point in the training loop where we have to compute the derivatives of that loss w.r.t.
every parameter in the model and use the gradients over the learning weights to decide how to adjust those
weights in a way that reduces the loss

You can do this in one line of code.

Each tensor generated by this computation knows how it came to be. ec. i2h has metadata, indivcating that it 
came from the mm of W_x and x.t(), and so it continues down the rest of the graph.
This history tracking enables the backward method to rapidly calculate the gradients your model needs for learning.

This history tracking is one of the things that enables flexibility and rapid iteration in your models. Even in complex
models with decision branches and loops, the computation history will track the particular paht thorugh the model that 
a particular input took and compute the backward derivatives correctly. 
"""



