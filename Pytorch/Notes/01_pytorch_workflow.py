DEBUG = 0
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import numpy
import matplotlib.pyplot as plt

# Pytorch Workflow
"""
Let's explore a an example PyTorch end-to-end workflow.

Resources:
* Ground truth notebook
* Book version of notebook
* Ask a question
"""

what_were_covering = {1: "data (prepare and load)",
                      2: "build movel",
                      3: "fitting the model to date (training)",
                      4: "making predictions and evaluating a model (inference)",
                      5: "saving and loading a model", 
                      6: "putting it all together"}

print(what_were_covering)
print(torch.__version__)

# 1. Data (preparing and loading)

"""
Data can be almost anything... in machine learning.

* Excel spreadsheets
* Images of any kind
* Videos (YouTube has lots of data...)
* VAudio like songs or podcasts
* DNA
* Text

Machine learning is a game of two parts:
1. Get data into a numerical representation
2. Build a model to learn patterns in that numerical representation
"""