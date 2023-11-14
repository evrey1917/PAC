import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch

def encode_label(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def shape_data(data):
    features = [np.reshape(x[0][0].numpy(), (784,1)) for x in data]
    labels = [encode_label(y[1]) for y in data]
    return zip(features, labels)

def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1]) == digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis=0)

# Initializing the transform for the dataset
transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5), (0.5))
                                            ])

# Downloading the MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root="./MNIST/train", train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False)

# test_dataset = torchvision.datasets.MNIST(
#     root="./MNIST/test", train=False,
#     transform=torchvision.transforms.ToTensor(),
#     download=False)

train = list(shape_data(train_dataset))

av_dig = []
for i in range(0,10):
    av_dig.append(average_digit(train, i))