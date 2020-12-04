import numpy as np
np.random.seed(42)

class MyNN:
  def __init__(self, learning_rate, layer_sizes):
    self.learning_rate = learning_rate
    self.layer_sizes = layer_sizes
    self.model_params = {}
    self.memory = {}
    self.grads = {}
    
    # Initializing weights
    for layer_index in range(len(layer_sizes) - 1):
      W_input = layer_sizes[layer_index + 1]
      W_output = layer_sizes[layer_index]
      self.model_params['W_' + str(layer_index + 1)] = np.random.randn(W_input, W_output) * 0.1
      self.model_params['b_' + str(layer_index + 1)] = np.random.randn(W_input) * 0.1
      
  def forward_single_instance(self, x):    
    a_i_1 = x
    self.memory['a_0'] = x
    for layer_index in range(len(self.layer_sizes) - 1):
      W_i = self.model_params['W_' + str(layer_index + 1)]
      b_i = self.model_params['b_' + str(layer_index + 1)]
      z_i = np.dot(W_i, a_i_1) + b_i
      a_i = 1/(1+np.exp(-z_i)) #sigmoid
      self.memory['a_' + str(layer_index + 1)] = a_i
      a_i_1 = a_i
    return a_i_1
  
  
  def log_loss(self, y_hat, y):
    m = y_hat[0]
    cost = -y[0]*np.log(y_hat[0]) - (1 - y[0])*np.log(1 - y_hat[0])
    return cost
  
  
  def backward_single_instance(self, y):
    a_output = self.memory['a_' + str(len(self.layer_sizes) - 1)]
    dz = a_output - y
     
    for layer_index in range(len(self.layer_sizes) - 1, 0, -1):
      print(layer_index)
      a_l_1 = self.memory['a_' + str(layer_index - 1)]
      dW = np.dot(dz.reshape(-1, 1), a_l_1.reshape(1, -1))
      self.grads['dW_' + str(layer_index)] = dW
      W_l = self.model_params['W_' + str(layer_index)]
      db = dz
      self.grads['db_' + str(layer_index)] = db
      dz = (a_l_1 * (1 - a_l_1)).reshape(-1, 1) * np.dot(W_l.T, dz.reshape(-1, 1))
         
  # TODO: update weights with grads
  def update(self):
    for layer_index in range(len(self.layer_sizes) - 1, 0, -1):
        self.model_params['W_' + str(layer_index)] -= self.learning_rate * self.grads['dW_' + str(layer_index)]
        self.model_params['b_' + str(layer_index)] -= self.learning_rate * self.grads['db_' + str(layer_index)]        

  # TODO: implement forward for a batch X.shape = (network_input_size, number_of_instance)
  def forward_batch(self, X):
    a_i_1 = X
    self.memory['a_0'] = X
    for layer_index in range(len(self.layer_sizes) - 1):
      W_i = self.model_params['W_' + str(layer_index + 1)]
      b_i = self.model_params['b_' + str(layer_index + 1)]
      z_i = np.dot(W_i, a_i_1) + b_i.reshape(W_i.shape[0],-1)
      a_i = 1/(1+np.exp(-z_i)) #sigmoid
      self.memory['a_' + str(layer_index + 1)] = a_i
      a_i_1 = a_i
    return a_i_1
  
  # TODO: implement backward for a batch y.shape = (1, number_of_instance)
  def backward_batch(self, y):
    a_output = self.memory['a_' + str(len(self.layer_sizes) - 1)]
    dz = a_output - y
     
    for layer_index in range(len(self.layer_sizes) - 1, 0, -1):
      #print(layer_index)
      a_l_1 = self.memory['a_' + str(layer_index - 1)]
      dW = np.divide(np.dot(dz, a_l_1.T), y.shape[1])
      self.grads['dW_' + str(layer_index)] = dW
      W_l = self.model_params['W_' + str(layer_index)]
      db = np.sum(dz, axis=1) / y.shape[1]
      self.grads['db_' + str(layer_index)] = db
      dz = (a_l_1 * (1 - a_l_1))* np.dot(W_l.T, dz) 
  
  # TODO: implement log_loss_batch, for a batch of instances
  def log_loss(self, y_hat, y):
    m = y_hat.shape[1]
    cost = -np.mean(np.multiply(y, np.log(y_hat)) + np.multiply((1-y), np.log(1-y_hat)))
    return cost

nn = MyNN(0.01, [3, 2, 1])

nn.model_params

x = np.random.randn(3)
y = np.random.randn(1)

y_hat = nn.forward_single_instance(x)
print(y_hat)

nn.backward_single_instance(y)

def train(X, y, epochs, batch_size):

  X_tmp = X
  y_tmp = y
  loss = []

  for e in range(1, epochs + 1):
    epoch_loss = 0
    np.random.shuffle(X) #shuffle X
    np.random.shuffle(y) #shuffle y
    num_epoch = X.shape[1] - (X.shape[1] % batch_size) #suppose to be 96 in our case
    X = np.split(X[:,:num_epoch], X.shape[1] // batch_size, axis=1)
    y = np.split(y[:,:num_epoch], y.shape[1] // batch_size, axis=1)
    batches = zip(X, y) # got the idea from the student Whatsapp group
    for X_b, y_b in batches:
      y_hat = nn.forward_batch(X_b)
      epoch_loss += nn.log_loss(y_hat, y_b)
      nn.backward_batch(y_b)
      nn.update()
    loss.append(epoch_loss/len(X))
    print(f'Epoch {e}, loss={epoch_loss/len(X)}')
    X = X_tmp
    y = y_tmp
  return loss

# TODO: Make sure the following network trains properly

nn = MyNN(0.001, [6, 4, 3, 1])

X = np.random.randn(6, 100)
y = np.random.randn(1, 100)
batch_size = 8
epochs = 2

train(X, y, epochs, batch_size)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

!git clone https://github.com/kfirbar/course-ml-data.git

raw = pd.read_csv('course-ml-data/Bike-Sharing-Dataset 2/day.csv')

raw["success"] = raw["cnt"] > (raw["cnt"].describe()["mean"])

x = raw[["temp", "atemp", "hum", "windspeed", "weekday"]].values.reshape(-1,5)
y = raw[["success"]].values.reshape(1,-1)

x_train, x_test, y_train, y_test = train_test_split(x, y.T, test_size=0.25)

x_train = x_train.T
y_train = y_train.T

nn = MyNN(0.001, [5, 40, 30, 10, 7, 5, 3, 1])

batch_size = 8
epochs = 100

loss = train(x_train, y_train, epochs, batch_size)

plt.plot(list(range(0, epochs)), loss)
plt.title('loss per epoch')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()