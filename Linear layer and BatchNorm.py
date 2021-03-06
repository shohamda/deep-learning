# -*- coding: utf-8 -*-
"""
# Assignment 3
In this assignment, you will be coding two new (nn.Module)s that implement a Linear layer and BatchNorm. Then you will use them to train a conv net over multiple datasets.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

"""**Datasets**
In this assignment you will only use training sets. Here, I only load CIFAR10, but you should load the following datasets, which you will use later:


*   CIFAR10
*   Fashion-MNIST
*  KMNIST



"""

cifar10_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
cifar10_trainloader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=64,
                                          shuffle=True)

# TODO - load more datasets
fashion_mnist_trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
fashion_mnist_trainloader = torch.utils.data.DataLoader(fashion_mnist_trainset, batch_size=64,
                                          shuffle=True)

kmnist_trainset = torchvision.datasets.KMNIST(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
kmnist_trainloader = torch.utils.data.DataLoader(kmnist_trainset, batch_size=64,
                                          shuffle=True)

"""# Neural net
This is our vanilla CNN for this experiment. The sizes of the Linear layers were designed for CIFAR10, the first dataset you should experiment with. Later, you will have to change those numbers to fit the other datasets.
"""

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # now a few fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNNFMnist(nn.Module):

    def __init__(self):
        super(CNNFMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 5, 3)
        self.conv3 = nn.Conv2d(5, 16, 3)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNFMnist_1(nn.Module):

    def __init__(self):
        super(CNNFMnist_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 5, 3)
        self.conv3 = nn.Conv2d(5, 16, 3)

        self.my_b_norm = MyBatchNorm(128)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.my_b_norm(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Creating the network 

net = CNN().cuda()     # -- For GPU
print(net)

# define loss function

criterion = nn.CrossEntropyLoss()

# define the optimizer

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training loop
def training_loop(net,trainloader):
  linear_training = []
  for epoch in range(20):  # 100

      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
          # get the inputs
          inputs, labels = data
          
          inputs = inputs.cuda() # -- For GPU
          labels = labels.cuda() # -- For GPU

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          if (i+1) % 200 == 0:    
              print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
              linear_training.append(running_loss/200)
              running_loss = 0.0

  print('Finished Training')
  return linear_training

"""# Task 1
Implement a nn.Module that immitates nn.Linear, with additional support for Dropout. In other words, this module should take three arguments: input dimension, output dimension and the keep_prop probability for dropout. Make sure to wrap your parameter tensors with nn.Parameter.
"""

# TODO: Implement the following Module. Size is the length of the input vectors.

class MyLinear(nn.Module):
  def __init__(self, input_dim, output_dim, keep_prob=0.):
    super(MyLinear, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.keep_prob = keep_prob
    self.weight = nn.Parameter(torch.zeros(output_dim, input_dim))
    self.bias = nn.Parameter(torch.zeros(output_dim))
    self.randomize()

  def randomize(self): 
    nn.init.xavier_uniform_(self.weight)
    a = 1 / math.sqrt(self.input_dim)
    nn.init.uniform_(self.bias, -a, a)
    
  def forward(self, x):
    i = torch.zeros(self.input_dim, device=torch.device('cuda:0'))
    i = i + self.keep_prob
    dropout = torch.bernoulli(i)
    x = x * dropout
    return x @ torch.transpose(self.weight, 0, 1) + self.bias

"""# Task 2
You should add your new MyLinear Module to our CNN. Simply replace fc1, fc2, and fc3 with your new module, this time with keep_prob=1. Then, train the network over CIFAR10 with and without your Module, and compare the loss curves (plot both of them onto the same figure, with two different colors).


"""

without_MyLinear = training_loop(net, cifar10_trainloader)

class CNN_1(nn.Module):

    def __init__(self):
        super(CNN_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # now a few fully connected layers
        self.fc1 = MyLinear(16 * 5 * 5, 120, 1)
        self.fc2 = MyLinear(120, 84, 1)
        self.fc3 = MyLinear(84, 10, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net1 = CNN_1().cuda()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net1.parameters(), lr=0.001, momentum=0.9)

with_MyLinear = training_loop(net1, cifar10_trainloader)

plt.plot(without_MyLinear, color = "red")
plt.plot(with_MyLinear, color = "blue")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

"""# Task 3

Use your MyLinear module like before, but this time compare the loss curves of two runs: keep_prob = 1 and keep_prob = 0.5.
"""

# You code for Task 3 goes here. Note - you don't have to copy the entire network here, just modify everything in place, run it, and collect the losses. Then, here write your plots.

class CNN_2(nn.Module):

    def __init__(self):
        super(CNN_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # now a few fully connected layers
        self.fc1 = MyLinear(16 * 5 * 5, 120, 0.5)
        self.fc2 = MyLinear(120, 84, 0.5)
        self.fc3 = MyLinear(84, 10, 0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net2 = CNN_2().cuda()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net2.parameters(), lr=0.001, momentum=0.9)

keep_prob_05 = training_loop(net2, cifar10_trainloader)

plt.plot(keep_prob_05, color = "red")
plt.plot(with_MyLinear, color = "blue")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

"""# Task 4

Implement a Module that performs Batch Normalization for the output of a Linear Module. In a nutshel, in it's forward procedure, this module should standartize the input (assumed to be of the same shape you use with nn.Linear, that is, (N, L) = N is the number of vectors in the batch, and L is the input vector), and then multiply it by gamma and add beta. Gamma and beta should be learnable, i.e., of nn.Parameter type.
Use running average to calculate the relevant information for testing time, and store them accordingly.
"""



# TODO: Implement the following Module. Size is the length of the input vectors.
 
class MyBatchNorm(nn.Module):
  def __init__(self, size, epsilon=1e-05):
    super(MyBatchNorm, self).__init__()
    self.gamma = nn.Parameter(torch.ones(size))
    self.beta = nn.Parameter(torch.zeros(size))
    self.epsilon = epsilon

    
  def forward(self, x):
    var, meu = torch.var_mean(x, axis=0)
    zed_prime = (x - meu)/(torch.sqrt(var + self.epsilon))
    zed_norm = self.gamma*zed_prime + self.beta
    return zed_norm

"""# Task 5
You should add your new MyBatchNorm Module to our CNN, right after fc1. Then, train the network over CIFAR10 with and without your Module, and compare the loss curves (plot both of them onto the same figure, with two different colors).
"""

# You code for Task 5 goes here. Note - you don't have to copy the entire network here, just modify everything in place, run it, and collect the losses. Then, here write your plots.

class CNN_3(nn.Module):

    def __init__(self):
        super(CNN_3, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # now a few fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.my_b_norm = MyBatchNorm(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.my_b_norm(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net3 = CNN_3().cuda()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net3.parameters(), lr=0.001, momentum=0.9)

with_MyBatchNorm = training_loop(net3, cifar10_trainloader)

plt.plot(without_MyLinear, color = "red")
plt.plot(with_MyBatchNorm, color = "blue")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

"""# Task 6
Run your network over all training sets, with and without batch norm, as designed in Task 4 and 5.
Present 6 curves, two for each dataset.
"""

net4 = CNN().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net4.parameters(), lr=0.001, momentum=0.9)
CIFAR10_without_MyBatchNorm = training_loop(net4, cifar10_trainloader)

net5 = CNN_3().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net5.parameters(), lr=0.001, momentum=0.9)
CIFAR10_with_MyBatchNorm = training_loop(net5, cifar10_trainloader)

net6 = CNNFMnist().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net6.parameters(), lr=0.001, momentum=0.9)
Fashion_MNIST_without_MyBatchNorm = training_loop(net6, fashion_mnist_trainloader)

net7 = CNNFMnist_1().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net7.parameters(), lr=0.001, momentum=0.9)
Fashion_MNIST_with_MyBatchNorm = training_loop(net7, fashion_mnist_trainloader)

net8 = CNNFMnist().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net8.parameters(), lr=0.001, momentum=0.9)
KMNIST_without_MyBatchNorm = training_loop(net8, kmnist_trainloader)

net9 = CNNFMnist_1().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net9.parameters(), lr=0.001, momentum=0.9)
KMNIST_with_MyBatchNorm = training_loop(net9, kmnist_trainloader)

# You code for Task 6 goes here. Note - you don't have to copy the entire network here, just modify everything in place, run it, and collect the losses. Then, here write your plots.

plt.plot(CIFAR10_with_MyBatchNorm, color = "green", label ='CIFAR10_WITH')
plt.plot(CIFAR10_without_MyBatchNorm, color = "limegreen", label ='CIFAR10_WITHOUT')
plt.plot(Fashion_MNIST_with_MyBatchNorm, color = "blue",  label ='FMIST_WITH')
plt.plot(Fashion_MNIST_without_MyBatchNorm, color = "dodgerblue",  label ='FMIST_WITHOUT')
plt.plot(KMNIST_with_MyBatchNorm, color = "red",  label ='KMIST_WITH')
plt.plot(KMNIST_without_MyBatchNorm, color = "tomato",  label ='KMIST_WITHOUT')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training graphs of datasets with and without Batch Normalization')
plt.legend()
plt.show()
