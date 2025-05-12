DEBUG = 0
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

To showcase this, let's create some *known* data using the linear regression formula

We'll use a linear regression formula to make a straight line with known parameters***

w^Tx + b
"""

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
stepsize = 0.02
X = torch.arange(start, end, stepsize).unsqueeze(dim=1)
y = weight * X + bias
print("First 10 elems of X:", X[:10], "First 10 elems of y:", y[:10])
 
print("length of x:", len(X), "length of y:", len(y))


# Split data into training and test sets (one of the most important concepts in machine learning in general)
"""
Course materials (training set) - learns patterns from here
Practice exam (validation set) - tune model patterns 
Final exam (test set) - see if model is ready for the wild

Generalization - the ability for a ML model to perform well on data it hasn't seen before 

Let's create a training and test set with our data
"""  

# Create a train/test split in this case it's 80, 20 %
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split] # everything up until the train split
X_test, y_test = X[train_split:], y[train_split:] # from train split to end

print("lengths of X_train, y_train, X_test, y_test:", len(X_train), len(y_train), len(X_test), len(y_test))

"""
How might we better visualize our data? 

This is where the data explorer's motto comes in!
"Visualize, visualize, visualize!"
"""

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_label=y_test,
                     predictions=None):
    """
    Plots training data, test data, and compares predictions
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Traning data")

    # Plot test data in greeen
    plt.scatter(test_data, test_label, c="g", s=4, label="Testing data")

    # Are there predictions
    if predictions is not None:
        # Plot predictions if they exist
        plt.scatter(test_data, predictions, c="r", label="Prediction")
        
    # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()

if DEBUG: plot_predictions()


# Building our first PyTorch model

"""
Our first PyTorch model!

Create a linear regression model
Because we're going to be building classes throughout the course,
I'd recommend getting familiar with OOP in Python, to do so you can follow resource from real python

Information about nn.Module
* Base class for all neural network modules
* Your models should also subclass this class
* Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign
the submodules as regular attirbutes

requires_grad = if it requires gradient

Start with random parameters (weights, bias), then looking at our training data, will update these
models to represent the pattern. Ideally if the model is learning correctly, it will take a weight, a random value, run it 
through the forward calculation and update weight and bias to best fit.

What our model does: 
* Start with random values (weight & bias)
* Look at training data and adjust the random values to better represent (or get closer to) the ideal values
(the weight & bias values we used to create the data)

How does it do so?
Through 2 main algorithms
1. Gradient descent (requires_grad=True)
2. Backpropagation
"""

# Create a linear regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch inherits from nn.Module
    def __init__(self): 
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))
        
        # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data
            return self.weights * x + self.bias # this is the linear regression formula 
        
# PyTorch model building essentials
"""
* torch.nn contains all of the buildings for computational graphs (a NN can be considered a computational graph_
* torch.nn.Parameter - what paramteers should our model try and learn, often a PyTorch layer from torch.nn will set these for us
* torch.nn.Module - the base class for all neural network modules, if you subclass it, you should override forward()

* torch.optim - this is where the optimizers in PyTorch livem they will help with gradient descent.
* def forward() - all nn.Module subclasses require you to override forward(). THis method defines what happens in the forward
function.

* torch.utils.data.Dataset - represents a map between key (label) and sample (features) pairs of your data.
Such as images and their associated labels
* torch.utils.data.DataLoader - creates a Python iterable over a torch Dataset (allows you to iterate over your data)

"""
        
# Checking the contents of our PyTorch model
"""
Now we've created a model. Now let's see what's inside...
So we can check out our model params or what's inside our model using ".parameters"
"""
# Create a random seed bc you're generating your parameters using randomizer
torch.manual_seed(42)

# Create an instancce of the model (this is a subclass of nn.Module)
model_0 = LinearRegressionModel()

if DEBUG: print(list(model_0.parameters())) 

# List named parameters
print(model_0.state_dict())

# Now get the model_0 parameters as close to as our known parameters (w = 0.7, b = 0.3.
#  The closer our values, the better we can predict our data)

# Making predictions using "torch.inference_mode()"
"""
To check our model's predictive power, let's see how well it predicts y_test based on x_test

When we pass data through our model, it's going to run it through the forward() method. 

"""
# make predictions with model
print("X_test:", X_test)

with torch.inference_mode(): # a context manager. turns off gradient tracking. When we're doing inference, we're not doing gradient, so we can disable. PyTorch will not be keeping track of gradient data. Runs faster.
    y_preds = model_0(X_test)

"""
similar to: However, inference_mode() is preferred
with torch.no_grad():
     y_preds = model_0(X_test)
"""

print("y_preds:", y_preds)
print("y_test:", y_test)

plot_predictions(predictions=y_preds)


# Training model
"""
The whole idea of training is for a model to move from some *unknown* parameters (these may be random)
to some known values

Or in other words, fro ma poor representation of the data to a better representation of the data.
One way to measure how poor or how wrong model predictions are, is to use a Loss Function
(Also called criterion or cost function in different areas).

* loss function: A function to measure how wrnog your model's predictions are to the ideal output, lower is better
* optimizer: takes into account the loss of a model and adjusts the model's parameters (e.g. weights and bias)
in our case to improce the loss function
    Inside the optimizer you'll often have to set two parameters:
    1. params - the model parameters you'd like to optimize, for example params=model_0.parameters()
    2. lr (learning rate) - the elarning rate is a hyperparameter that defines how big/small the optimizer changes the parameters
       with each step (a small lr results in small changes, a large lr results in large changes)

Specifically for PyTorch, we need:
* Training loop
* Testling loop
"""

# Setup a loss function. Measures error
loss_fn = nn.L1Loss()
print(loss_fn)

# Setup an optimizer (stochastic gradient descent) adjusts our model parameters to reduce the loss
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr = 0.01) # lr - learning rate - possible the most important hyperparameter (parameter you can set)

"""
Which loss function and optimizer should I use?

This is problem specific. But with experience, you'll get an idea of what works and what doesn't with your particular
problem set

For example, for a refression problem (like ours), a loss function of nn.L1Loss() and an optimizer like torch.optim.SGD() will
suffice

For a classification problem like classifiying whether a photo is a dog or a cat, you'll likely want to use
a loss function of nn.BCELoss() (Binary Cross Entropy Loss)
"""

# Building a training loop (and a testing loop) in PyTorch

"""
A couple of things we need in a training loop:
0. Loop through the data and do...
1. forward pass (this involves data moving through our model's "foward()" functions) to make predictions on data - also called forward propogation
2. calculate the loss (compare forward pass predictions to ground truth labels)
3. optimizer zero grad
4. loss backward - move backwards through the network to calculate the gradients of each of the parameters of our model with respect to the loss (backpropogation)
5. optimizer step - use the optimizer to adjust the model's parameters to try and improve the loss (gradient descent)
"""

# an epoch is 1 loop through the data... (this is a hyperparameter because we've set it to 0)
epochs = 200

# Training
"""
Rewriting for practice:
# 0. pass the data through the model for a number of epochs (hyperparameter) (e.g. 100 for 100 passes of the data)
for epoch in range(epoch):

    model.train() # Put model in training mode (this is the default state of a model)

    y_pred = model_0(X_test) # 1. Forward pass on train data using the forward() method located inside model object
    
    loss = loss_fn(y_pred, y_true) # 2 Calculate the loss (how different the model's prediction ot the true dataset)
    
    optimizer.zero_grad() # 3. Zero the gradients of the optimizer (they accumulate by default)
    
    loss.backwards() # 4 Perform backpropogation on the loss function (compute the gradient )
    
    optimizer.step() # 5 Progress / step the optimizer (gradient descent)

    model_0.eval()
"""
# Track different values to compare future experiments to past experiments
epoch_count = [] # Create empty lists for storing useful values (helpful for tracking model progress) 
loss_values = []
test_loss_values = []

# Loop through the data
for epoch in range(epochs): 
    # set the model to training mode
    model_0.train() # train mode in PyTorch sets all parameters that require gradients to require gradients
    
    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train) # predictions first and then actual labels (input, target)
    if epoch % 100 == 0:
        print("Previous Loss:", loss)

    # 3. Optimizer zero grad
    optimizer.zero_grad()
    
    # 4. Perform backpropogation on the loss with respect to the parameters of the model
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step() # by default how the optimizer changes will accumulate through the loop... So we need have to zero them above in step 3 for the next iteration of the loop

    #############################################

    """
    Testing loop

    """

    # Testing Loop
    # Tell the model we want to evaluate rather than train (this turns off functionality used for training but not evaluation)
    model_0.eval() # turns off diffrent settings in the model not needed for evaluation/testing (dropout/batchnorm layers)

    # turn on torch.inference_mod() context manager to disable functionality such as grad tracking for inference (not needed)
    with torch.inference_mode(): # turns off gradient tracking and a couple of more things behind the scenes
        # 1. Do the forward pass
        test_pred = model_0(X_test) # pass test data through model (this call's model's forward method)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test) # test data with loss

    # print out values to keep track of values 
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}") 
        # print out model state dictionary
        print(model_0.state_dict())

with torch.inference_mode():
     y_preds_new = model_0(X_test)

print("weight:", weight, "bias:", bias)

plot_predictions(predictions=y_preds_new)

# Check and visualize the training loss curve and test loss curve. If there is a decreasing istance, model is converging
print(epoch_count, np.array(torch.tensor(loss_values).numpy()), test_loss_values)
plt.plot(epoch_count, np.array(torch.tensor(loss_values).cpu().numpy()), label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show() 


# Saving a model in PyTorch (to use it later or to send it to someone)
"""
There are 3 main methods for saving and loading models in PyTorch.
1. torch.save() - allows you to save a PyTorch object in Python's pickle format
2. torch.load() - allows you to load a saved PyTorch object
3. torch.nn.Module.load_state_dict() - this allows to load a model's saved state dictionary
model_0.state_dict()

Usually just save model's statedict, but you can also save and load the entire model
"""

# 1. Create model directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_0.pth" # use .pt or .pth most commonly
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

MODEL_SAVE_PATH

# 3. save the model save_dict()
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
print("model_0.state_dict()", model_0.state_dict())

# Loading a PyTorch model
"""
Since we saved our model's state_dict() rather than the entire model, we'll create a new instance of our
model class and load the saved state_dict() into that

"""

# To load in a saved state_dict, we need to instantiate a new instance of our model class
loaded_model_0 = LinearRegressionModel()

# load the saved state_dict of model_0 (this will update the new instance with updated parameters)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH)) 

print("loaded_model_0:", loaded_model_0.state_dict())


# Make some predictions with our loaded model
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)

    print(loaded_model_preds)



# Make some model preds
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)

# Compare laoded model preds with original model preds
print("equal?", y_preds  == loaded_model_preds) # all true


# 6. Now put everything together
"""
Let's go back through the steps above and see it all in one place
"""

# 6.1 Data 

# Create device agnostic code. This means if we've got access to a GPU, our code will use it (for potentially faster computing)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

# Create some data using the linear regression formula of y = weight * X + bias
weight = 0.2
bias = 0.1

# Create range values
start = 0
end = 1
stepsize = 0.02

X = torch.arange(start, end, stepsize, device=device).unsqueeze(dim=1) # features, without unsqueeze, errors will pop up
y = weight * X + bias # labels

print("X:", X[:10])
print("y:", y[:10])

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split] # 80%

X_test, y_test = X[train_split:], y[train_split:] # 20%

print(len(X_train), len(y_train), len(X_test), len(y_test))

# plot the data
# Note: if you don't have the plot_predictions() function loaded, this will error
plot_predictions(X_train, y_train, X_test, y_test)

# 6.2 Building a PyTorch Linear model
# Create a linear model by subclassing nn.Module

"""
swap creating a random weight and random bias with creating a linear layer and having that
define our weights and bias

We can also get rid of the formula y = weight * X + bias and replace it with self.linear_layer(x).
Pass it through the linear layer and have it compute some predefined forward function
"""
    
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        # Use nn.Linear() for creating the model parameters / also called: linear transform, probing layer, fully connected layer, dense layer
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1) # one feature of x map to one label of y
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x) # perform linear regression formula behind the scenes
    
# Set the manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print(model_1, model_1.state_dict())

# Check the model current device
print(next(model_1.parameters()).device)

# Set the model to use the target device
model_1.to(device)
print(next(model_1.parameters()).device)

# 6.3 Training 
"""
For training we need 
a loss function 
optimizer
training loop
testing loop
"""

# Setup loss function
loss_fn = nn.L1Loss() # same as MAE

# setup our optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01) # passing in model parameters, and the learning rate into your optimizer

# Let's write a training loop
torch.manual_seed(42)

epochs = 2000

# Put data on the target device (device agnostic code for data)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):

    model_1.train()

    # 1. do forward pass
    y_pred = model_1(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zewro grad
    optimizer.zero_grad()

    # 4. perform backpropogation
    loss.backward()

    # 5. Calculate the gradient with respect to each parameter in the model
    optimizer.step()

    #################
    # Testing
    model_1.eval()
    
    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, y_test)

    # Print out what's happening
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss} | Test_loss: {test_loss}")
        print(f"State_dict:, {model_1.state_dict()}")


# 6.4 Making and evaluating predicitons
# Turn model into evaluation mode (everytime you want to evalute or make predictions)
model_1.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model_1(X_test).cpu()


print("y_preds:", y_preds)
plot_predictions(predictions=y_preds) # Different because you changed the weight to = 0.2 and bias to 0.1

# 6.5 Saving and loading a trained model
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)
print("model_1.state_dict()", model_1.state_dict())

# loading model into new model

# Create a new instance of LRMV2
loaded_model_1 = LinearRegressionModelV2()
# load state dict of saved model
loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH)) 

# Send this to the device you want to work with, but in my case since I have a Mac, I only have cpu
loaded_model_1.to(device)
print("loaded_model_1.state_dict():", loaded_model_1.state_dict())


# Evaluate the loaded model
loaded_model_1.eval()

with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)

print(y_preds == loaded_model_1_preds)