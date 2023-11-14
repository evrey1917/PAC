import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

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
    features = np.empty(shape=(1000, 784))
    labels = np.empty(shape=(1000, 10))
    for i in range(1000):
        features[i] = np.reshape((data[i])[0][0].numpy(), (1, 784))
        labels[i] = encode_label((data[i])[1]).T
    return features, labels

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

trainX, trainY = shape_data(train_dataset)

X = []
y = []
number = np.zeros((10,1))
for i in range(len(trainX)):
    if (number[np.argmax(trainY[i])] < 30):
        number[np.argmax(trainY[i])] = number[np.argmax(trainY[i])] + 1
        X.append(trainX[i])
        y.append(np.argmax(trainY[i]))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# # Perform t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualize the results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", alpha=0.6)
plt.colorbar(scatter, ticks=range(10), label="Digits")
plt.title("MNIST t-SNE Visualization")
plt.xlabel("t-SNE 1st Component")
plt.ylabel("t-SNE 2nd Component")
plt.show()