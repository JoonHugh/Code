"""
You've seen how a model is built, and how to give it a batch of input and examine the output.
The model didn't do very much because it hasn't been trained yet. For that, we'll need to feed it a bunch of data. 

In order to train our model, we're going to need a way to feed it data in bulk. 
This is where the pytorch dataset and dataloader classes come into play. Let's see them in action.
"""

import matplotlib.pyplot as plt # Because we'll be rendering some images 
import torch
# Give us our datasets and some transforms tnat we eed to apply to the images 
# to make them digestable by our pytorch model
import torchvision
import torchvision.transforms as transforms
import numpy as np

"""
Below, we're going to demonstrate using one of the read-to-download, open-access datasets from TorchVision,
how to transform the images for consumption by your mode, and how to use the DataLoader to feed batches of data 
to your model

The first thing we need to do is transform our incoming images into a PyTorch tensor:
"""

transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

"""
Here we specify two transformations for our input:
    * transforms.ToTensor() converts images loaded by Pillow into PyTorch tensors
    * transforms.Normalize() adjusts the values of the tensor so that their average is 0 and their std_dev 
    is 0.5. Most activation funcs have their strongest gradients around x = 0, so centering our data there can speed learning

There are many more transforms available, including cropping, centering, rotation, and reflection.

Next we'll create an instance of the CIFAR10 dataset. This is a set of 32x32 color image tiles representing 10 classes
of objects: 6 animals (bird, cat, deer, dog, frog, horse) and 4 vehicles (airplane, automobile, ship, truck):
"""

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

"""
It might take a little while to download the dataset.

This is an example of creating a dataset object in PyTorch. Downloadable datasets (like CIFAR-10 above) are subclasses
of torch.utils.data.Dataset. Dataset classes in PyTorch include the downloadable datasets in TorchVision, TorchText, and 
TorchAudio, as well as utility dataset classes such as torchvision.datasets.ImageFolder, which will read a folder of labeled
images. You can also create your own subclasses od Dataset

When we instantiate our dataset, we need to tell it a few things:
    * The filesystem path to where we want the data to go,
    * Whether or not we are using this set for training: most datasets will be split into training and test subsets
    * Whether we would like to download the dataset if we haven't already.
    * The transformations we want to apply to the data

Once your dataset is ready, you can give it to the DataLoader:
"""
if __name__ == "__main__":

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    """
    A Dataset subclass wraps access to the data, and is specialized to the type of data it's serving, The DataLoader knows
    nothing about the data, but organizes the input tensors served by the Dataset into batches with the parameters you specify.

    In the ex. above, we've asked a DataLoader to give us batches of 4 images from trainset, randomizing their order (shuffle=True),
    and we told it to spin up two workers to load data from disk.

    It's good practice to visualize the batches your DataLoader serves:
    """

    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    def imshow(img):
        img = img / 2 + 0.5 # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

        plt.show()


    # get some random training images
    images, labels = next(iter(trainloader))

    # print labels
    print(" ".join("%5s" % classes[labels[j]] for j in range(4)))
    
    # Show images
    imshow(torchvision.utils.make_grid(images))
 

