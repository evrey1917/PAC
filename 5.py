import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def my_scale(av_dig, X):
    scaled = []
    for i in range(len(X)):
        scaled.append(give_me(av_dig, X[i]))
    return scaled

def give_me(av_dig, image):
    weight = -10000
    index = 0
    for i in range(0,10):
        local_weight = np.dot(av_dig[i], image)
        if (weight < local_weight):
            weight = local_weight
            index = i
    vector_weight = [0 for i in range(10)]
    vector_weight[index] = 1
    return vector_weight

def encode_label(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def shape_data(data):
    features = np.empty(shape=(len(data), 784))
    labels = np.empty(shape=(len(data), 10))
    for i in range(len(data)):
        features[i] = np.reshape((data[i])[0][0].numpy(), (1, 784))
        labels[i] = encode_label((data[i])[1]).T
    return features, labels

def average_digit(X, Y, digit):
    data = zip(X, Y)
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

trainX, trainY = shape_data(train_dataset)
testX,  testY = shape_data(test_dataset)

av_dig = []
for i in range(0,10):
    av_dig.append(np.transpose(average_digit(trainX, trainY, i)))

X = []
y = []
number = np.zeros((10,1))

testX = my_scale(av_dig, testX)
for i in range(len(testX)):
    X.append(testX[i])
    y.append(np.argmax(testX[i]))
X = np.array(X)

# # More accurate result on the plot:
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, alpha=0.6)
plt.colorbar(scatter, ticks=range(10), label="Digits")
plt.title("MNIST t-SNE Visualization")
plt.xlabel("t-SNE 1st Component")
plt.ylabel("t-SNE 2nd Component")
plt.show()