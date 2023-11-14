import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch

def give_me(av_dig, image):
    weight = -10000
    index = 0
    for i in range(0,10):
        local_weight = np.dot(av_dig[i], image)[0][0]
        if (weight < local_weight):
            weight = local_weight
            index = i
    vector_weight = np.zeros((10,1))
    vector_weight[index] = 1
    return vector_weight

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

test_dataset = torchvision.datasets.MNIST(
    root="./MNIST/test", train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False)

train = list(shape_data(train_dataset))
test = list(shape_data(test_dataset))

av_dig = []
for i in range(0,10):
    filtered_array = average_digit(train, i)
    av_dig.append(np.transpose(filtered_array))


print("Write:")
a = input()

image = test[a][0]
answer = test[a][1]

vector_weight = give_me(av_dig, image)

print(vector_weight)
print(test[a][1])