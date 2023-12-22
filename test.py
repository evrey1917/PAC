import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
batch_size = 8

# PATH = './cifar_mein.pth'
PATH = './cifar_anot.pth'

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print("0:", x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        print("1:", x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print("2:", x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        print("3:", x.shape)
        x = F.relu(self.fc1(x))
        print("4:", x.shape)
        x = F.relu(self.fc2(x))
        print("5:", x.shape)
        x = self.fc3(x)
        print("6:", x.shape)
        return x

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 4, 1)
        self.conv3 = nn.Conv2d(4, 4, 3, padding=1)
        self.conv4 = nn.Conv2d(4, 16, 1)

        self.conv5 = nn.Conv2d(16, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def fir_conv(self, h):
        h1 = self.conv2(h)
        h1 = self.conv3(h1)
        h1 = F.relu(self.conv4(h1))

        h1 = self.conv2(h1)
        h1 = self.conv3(h1)
        h1 = self.conv4(h1)
        h1 = h + h1

        h1 = F.relu(h1)
        return h1

    def forward(self, x):
        # Не меняем, входной ствол
        h = F.relu(self.conv1(x))
        h = self.pool(h)
        #

        h1 = self.fir_conv(h)
        # h1 = self.fir_conv(h1)
        # h1 = self.fir_conv(h1)

        h1 = self.conv5(h1)
        h1 = self.pool(h1)
        
        h = torch.flatten(h1, 1) # flatten all dimensions except batch
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h

def imshower(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def pre_load():
    global trainloader
    global testloader

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

def print_data():
    # show images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshower(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

def train():
    net = MyNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.925)

    enumerate(trainloader, 0)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), PATH)

    print('Saved on path:', PATH)

def all_accuracy():
    net = MyNet()
    net.load_state_dict(torch.load(PATH))
    print("Loaded")

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def class_accuracy():
    net = MyNet()
    net.load_state_dict(torch.load(PATH))
    print("Loaded")

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def main():
    pre_load()
    # train()
    # print_data()
    all_accuracy()
    class_accuracy()

if __name__ == "__main__":
	main()