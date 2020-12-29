# -*- coding: utf-8 -*-
"""Shoham Danino 204287635 Yehonatan Moshkovitz 314767674 - DL Assignment 2

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hUnIpK7OVBXyQ1r-V6VKu_XqQLcy9UdX

**Assignment # 2, CNN over Fasion MNIST**

In this assignment you are requested to build a convolutional network and train it over the Fasion MNIST data, which is a collection of 28X28 back and white images, classified into 10 different classes of clothing items. For more information about Fashion MNIST you may refer to: 
https://github.com/zalandoresearch/fashion-mnist
"""

# Loading Fashion MNIST

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())

testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transforms.ToTensor())

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# Use dataloaders for train and test (batch size is 4)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

# The images are of 1, 28, 28 size (only one black-white channel)

trainset[0][0].shape

"""# **Part 1**: Implementing a CNN network for Fashion MNIST
Here is what you need to do; you are encoureged to look at notebook "DL Notebook 9 - CIFAR CNN" when trying to complete the next steps.


Write a network CNNFMnist, that has the following architecture:

* Convolution with 10 3X3 filters
* Relu
* Max pool with 2X2
* Convolution with 5 3X3 filters
* Relu
* Convolution with 16 3X3 filters
* Relu
* Max pool with 2X2
* Liner, output size 128
* Relu
* Liner, output size 64
* Relu
* Liner, output size 10
"""

trainset[0]

class CNNFMnist(nn.Module):

    def __init__(self):
        super(CNNFMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 5, 3)
        self.conv3 = nn.Conv2d(5, 16, 3)

        #self.fc1 = nn.Linear(16 * 3 * 3, 128)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        #x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        h = x
        x = self.fc3(x)
        return h, x

"""Write a code that trains the network with FashionMNIST train dataset, for classification (use cross entropy, and SGD).
Run the network for at least 10 epochs, over the entire dataset. Make sure to print the loss over the train set as well as the **test set** over time (say, every 1000 batches, but it's up to you), so you will know where you are during training. 

Note, measuring loss of test is similar to measuring loss over the train test. However, make sure not to run the test images in back propagation. Use them only in forward and calulate the average loss over the entire test set. Since it will make the training process run slower, you should measure loss for the test set only at the end of an epoch (so overall you get 10 loss values for the test set). You are encoureged to write a different function for claculating the loss of the test set, and then call it from the training procedure.


You should collect the loss values in an array, so you can plot then into two curves, one for train and one for test.

In addition, you should measure the time it takes you to train the network completely.


"""

net = CNNFMnist()            # -- For CPU

print(net)

# define loss function

criterion = nn.CrossEntropyLoss()

# define the optimizer

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training loop
def test_net():
    test_loss = []
    for data in testloader:
        inputs, labels = data
        #inputs = inputs.cuda() # -- for GPU
        #labels = labels.cuda() # -- for GPU
        
        _, outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss.append(loss.item())

    # return average loss over all test set
    return sum(test_loss) / len(test_loss)

train_loss = []
test_loss = []
interval_tuples = []

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for epoch in range(10):

    running_train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        _, outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        if (i+1) % 1000 == 0:    
            interval_tuples.append(str((epoch + 1, i + 1)))
            train_loss.append(running_train_loss / 1000)
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_train_loss / 1000))
            running_train_loss = 0.0

    net.eval()
    with torch.no_grad():   
        running_test_loss = test_net()
        print("epoch {}, test loss: {}\n".format(epoch + 1, running_test_loss))
        test_loss.append(running_test_loss)
    net.train()

print('Finished Training')
end.record()
# Waits for everything to finish running
torch.cuda.synchronize()

mnist_cpu=(start.elapsed_time(end)/1000)

# print train loss graph per batch
plt.figure(figsize=(25,10))
plt.plot(interval_tuples, train_loss)
plt.xlabel('(epoch, batch)')
plt.ylabel('loss')
plt.title('train-set loss per epochs')
plt.show()

# Visualization of train and test loss
plt.plot(range(1,11), train_loss[::15], color='blue')
plt.plot(range(1,11),test_loss, color='red')
plt.legend(["train", "test"], loc ="best") 
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss over train and test sets')
plt.show()

"""Write a function that evaluates the resulted model over the entire test data of FashionMNIST. Provide a single accuracy number."""

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data

        _, outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
acc_cpu = (100 * correct / total)

"""# **Part 2**: Training with a GPU 
You are requested to change your code to use the GPU instead of the CPU.
This can be easily done bu converting every torch.tensor to torch.cuda.tensor. 

Specific instructions:
* Change the hardware equipent of your colab notebook. To do that, go to the "Runtime" menu, and then to "Change runtime type". In the dialog box, change "Hardware accelerator" to GPU.
* Please follow the lines that were commented out with the comment    # -- For GPU
* Also, remove the lines that have the comment # -- For CPU

Train your network again and compare training time.
"""

#copy the same code with GPU insted of CPU

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())

testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transforms.ToTensor())

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# Use dataloaders for train and test (batch size is 4)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

# The images are of 1, 28, 28 size (only one black-white channel)

trainset[0][0].shape

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
        h = x
        x = self.fc3(x)
        return h, x

net = CNNFMnist().cuda()     # -- For GPU

print(net)

# define loss function

criterion = nn.CrossEntropyLoss()

# define the optimizer

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training loop
def test_net():
    test_loss = []
    for data in testloader:
        inputs, labels = data
        inputs = inputs.cuda() # -- for GPU
        labels = labels.cuda() # -- for GPU
        
        _, outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss.append(loss.item())

    # return average loss over all test set
    return sum(test_loss) / len(test_loss)

train_loss = []
test_loss = []
interval_tuples = []

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for epoch in range(10):

    running_train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda() # -- For GPU
        labels = labels.cuda() # -- For GPU

        optimizer.zero_grad()

        _, outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        if (i+1) % 1000 == 0:    
            interval_tuples.append(str((epoch + 1, i + 1)))
            train_loss.append(running_train_loss / 1000)
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_train_loss / 1000))
            running_train_loss = 0.0

    net.eval()
    with torch.no_grad():   
        running_test_loss = test_net()
        print("epoch {}, test loss: {}\n".format(epoch + 1, running_test_loss))
        test_loss.append(running_test_loss)
    net.train()

print('Finished Training')
end.record()
# Waits for everything to finish running
torch.cuda.synchronize()

mnist_gpu=(start.elapsed_time(end)/1000)

# print train loss graph per batch
plt.figure(figsize=(25,10))
plt.plot(interval_tuples, train_loss)
plt.xlabel('(epoch, batch)')
plt.ylabel('loss')
plt.title('train-set loss per epochs')
plt.show()

# Visualization of train and test loss
plt.plot(range(1,11), train_loss[::15], color='blue')
plt.plot(range(1,11),test_loss, color='red')
plt.legend(["train", "test"], loc ="best") 
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss over train and test sets')
plt.show()

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.cuda()  # -- for GPU
        labels = labels.cuda()  # -- for GPU

        _, outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
acc_gpu = (100 * correct / total)

#comparing GPU and CPU


#print time graph
plt.bar(('cpu', 'gpu'), (mnist_cpu, mnist_gpu))
plt.title('time (in seconds)')
plt.show()

#print accuracy graph that should be the same
plt.bar(('cpu', 'gpu'), (acc_cpu,acc_gpu))
plt.title('accuracy')

plt.show()

"""# **Part 3**: Transfer Learning
Traininng data is a valuable resource, and sometimes there is not enough of it for traiing a neural netowrk at scale. To handle this situation, one approach is transfer learning, where we train our network on a different related task, and then switch to train it on the downstream task that we focus on. In this last part of the assignment, you are requested to pretrain your network on CIFAR-10, then train it on Fashion-MNIST, and measure its contribution to the results. To do that, please follow the steps:

**Step 1**

Modify your CNNFMnist implementation to return the output of the layer one before last after Relu (Linear layer of size 64, above) in addition to the final output. For example:

```
def forward(self, x):
  ...
  return h, out
```

 and train it on the training-set part of CIFAR-10. Use batch size of 4, and train it for at least 10 epochs. Note that CIFAR-10 images are of different shapes (3X32X32), therefore a conversion into 1X28X28 is needed. To do that, when you load CIFAR-10 using a torchvision Dataset, you can use the transformer torchvision.transforms.Grayscale(num_output_channels=1) in order to convert the images to a 1X32X32 grayscale volume:

```
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor()]))
```
Then, from each 1X32X32 image, sample 10 1X28X28 images at random positions, and use them for training (*optional* - for data augmentation, if you want, you can also generate the reflection of each of the 10 images and add them the training set).

**Setp 2**

Once done, write a new Module CNNFMnist2, which uses CNNFMnist as one of its sub modules, followed by some additional layers. The output of CNNFMnist that goes into the next layer, should be the output of the 64 neuron one-before-last layer, as described above. CNNFMnist2 should have the following architecture:

* CNNFMnist
* Liner, output size 32
* Relu
* Liner, output size 16
* Relu
* Liner, output size 10

Make sure to allow the user to assign a pre-trained version CNNFMnist as a member of the module. For example:

```
class CNNFMnist2(nn.Module):
    def __init__(self, trained_cnnfmnist_model):
        super(CNNFMnist2, self).__init__()
        self.trained_cnnfmnist_model = trained_cnnfmnist_model
        self.fc1 = nn.Linear(64, 32)
        ...
```

**Step 3**

Train and eval CNNFMnist2 on Fashion-MNIST a few times:
- Using the pre-trained version of CNNFMnist.
- Using a fresh CNNFMnist instance (without training it).
- (Optional) Using the pre-trained version of CNNFMnist, after freezing its weights using the .eval() function.

Report on evaluation results (accuracy) for all of those cases.
"""

train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                                                                                  torchvision.transforms.CenterCrop((28,28)),
                                    torchvision.transforms.ToTensor()]))


for i in range(9):
    train_data2 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                                                                                  torchvision.transforms.CenterCrop((28,28)),
                                    torchvision.transforms.ToTensor()]))
    train_data = train_data + train_data2

trainsetcifar = torch.utils.data.ConcatDataset(train_data)

testsetcifar = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                                                                                 torchvision.transforms.CenterCrop((28,28)),
                                    torchvision.transforms.ToTensor()]))

classescifar = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Use dataloaders for train and test (batch size is 64)

trainloadercifar = torch.utils.data.DataLoader(train_data, batch_size=64,
                                          shuffle=True)

testloadercifar = torch.utils.data.DataLoader(testsetcifar, batch_size=64,
                                         shuffle=False)

net_cifar = CNNFMnist().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net_cifar.parameters(), lr = 0.001, momentum=0.9)

for epoch in range(10):

    running_train_loss = 0.0
    for i, data in enumerate(trainloadercifar, 0):
        inputs, labels = data
        inputs = inputs.cuda() # -- For GPU
        labels = labels.cuda() # -- For GPU

        optimizer.zero_grad()

        _, outputs = net_cifar(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        if (i+1) % 1000 == 0:    
            #interval_tuples.append(str((epoch + 1, i + 1)))
            #train_loss.append(running_train_loss / 1000)
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_train_loss / 1000))
            running_train_loss = 0.0

class CNNFMnist2(nn.Module):
    def __init__(self, trained_cnnfmnist2_model):
        super(CNNFMnist2, self).__init__()
        self.trained_cnnfmnist2_model = trained_cnnfmnist2_model
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x,_ = self.trained_cnnfmnist2_model(x)
        #x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def accuracy(net):
  correct = 0
  total = 0
  with torch.no_grad():
      for data in testloader:
          images, labels = data
          images = images.cuda()  # -- for GPU
          labels = labels.cuda()  # -- for GPU

          outputs = net(images)
          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
  return (100 * correct / total)

#pretarined
pretarined = CNNFMnist2(net_cifar).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(pretarined.parameters(), lr = 0.001, momentum=0.9)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

for epoch in range(10):

    running_train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda() # -- For GPU
        labels = labels.cuda() # -- For GPU

        optimizer.zero_grad()

        outputs = pretarined(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        if (i+1) % 1000 == 0:    
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_train_loss / 1000))
            running_train_loss = 0.0

end.record()
# Waits for everything to finish running
torch.cuda.synchronize()

pretrained_seconds=(start.elapsed_time(end)/1000)

a = accuracy(pretarined)

#untrained
untrained_cifar = CNNFMnist().cuda()
untrained_net_cifar = CNNFMnist2(untrained_cifar).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(untrained_net_cifar.parameters(), lr = 0.001, momentum=0.9)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for epoch in range(10):

    running_train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda() # -- For GPU
        labels = labels.cuda() # -- For GPU

        optimizer.zero_grad()

        outputs = untrained_net_cifar(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        if (i+1) % 1000 == 0:    
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_train_loss / 1000))
            running_train_loss = 0.0

end.record()
# Waits for everything to finish running
torch.cuda.synchronize()

untrained_seconds=(start.elapsed_time(end)/1000)

b = accuracy(untrained_net_cifar)

#pretrained-freeze
net_cifar.eval()
pretrained_net_cifar = CNNFMnist2(net_cifar).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(pretrained_net_cifar.parameters(), lr = 0.001, momentum=0.9)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for epoch in range(10):

    running_train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda() # -- For GPU
        labels = labels.cuda() # -- For GPU

        optimizer.zero_grad()

        outputs = pretrained_net_cifar(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        if (i+1) % 1000 == 0:    
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_train_loss / 1000))
            running_train_loss = 0.0
end.record()
# Waits for everything to finish running
torch.cuda.synchronize()

pretrained_freeze_seconds=(start.elapsed_time(end)/1000)

c = accuracy(pretrained_net_cifar)

plt.bar(('pretrained', 'untrained', 'pretrained_freeze'), (pretrained_seconds, untrained_seconds, pretrained_freeze_seconds))
plt.title('time (in seconds)')
plt.show()

plt.bar(('pretrained', 'untrained', 'pretrained_freeze'), (a, b, c))
plt.title('accuracy')

plt.show()

"""# Submission instructions

You should submit a pdf file with the following items:

CPU Experiment:
*   Plot of loss curves (train in blue, test in red)
*   Training time

GPU Experiment:
*   Plot of loss curves (train in blue, test in red)
*   Training time

Transfer Learning Experiment:
* Accuracy results on test set for the 2-3 implemeted settings (see above)

Link for your collab notebook.
ID and names of submitters.


Good luck!
"""