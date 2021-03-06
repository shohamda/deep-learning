# -*- coding: utf-8 -*-

#In this assignment you will be working with a character based LSTM language model, which you will turn into a text classifier for sentiment analysis using **Attention**. For that, you will need to develop the Attention mechanism that aggregates the hidden output vectors that you get per character into a single vector, which you will use as an input for a final linear classifier.

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from random import sample

!git clone https://github.com/kfirbar/course-ml-data.git

"""The data that you will be working with is SST-2, which is a collection of reviews, each is classified into 0/1 reflecting the overall sentiment of the author (0 = negative, 1 = positive). In the next piece of code, we load the data and create a dictionary (named Vocab) that assigns a unique ID per character, similar to what have done in DL Notebook 12. Finally, each one of *train* and *dev* is a list of tuples, with the first item being the text encoded as character indices, and the second item is the label (0, 1)."""

# We will work only with texts of size < 50 characters
MAX_SEQ_LEN = 50

class Vocab:
    def __init__(self):
        self.char2id = {}
        self.id2char = {}
        self.n_chars = 1
        
    def index_text(self, text):
      indexes = [self.index_char(c) for c in text]
      return indexes
    
    def index_char(self, c):
        if c not in self.char2id:
            self.char2id[c] = self.n_chars
            self.id2char[self.n_chars] = c
            self.n_chars += 1
        return self.char2id[c]
            
            
def load_data(data, vocab):
  data_sequences = []
  for text in data.iterrows():
    if len(text[1]["sentence"]) <= MAX_SEQ_LEN:
      indexes = vocab.index_text(text[1]["sentence"])
      data_sequences.append((indexes, text[1]["label"]))
  return data_sequences

vocab = Vocab()
train = load_data(pd.read_csv('/content/course-ml-data/SST2_train.tsv', sep='\t'), vocab)
dev = load_data(pd.read_csv('/content/course-ml-data/SST2_dev.tsv', sep='\t'), vocab)
print(f'Train size {len(train)}, Dev size {len(dev)}, vocab size {vocab.n_chars}')

"""# Task 1
The following RNN architectures takes a single sentence as an input (formatted as a 1D tensor of character ids), and returns a distribution over the labels. In our case the number of labels is 2 (negative, positive). 

I basically copied the same architecture from Notebook 12, where each input character gets an output vector from the LSTM module, which are used to precdict the next character in line. However, here, we are not really interested in predicting the next character, but in aggregating all those output vectors into a single "context" vector, which will be sent to a Linear layer for the final classification step.

Therefore, you are requested to add the relevant code for aggregating the output vectors using the **additive attention** approach, following presentation *DL 14*. Note that some of what you need to add should be parameters, which you need to define under the __init__ function.

"""

class SeqModel(torch.nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, output_size):
    super(SeqModel, self).__init__()
    self.embedding = torch.nn.Embedding(input_size, embedding_size).cuda()
    self.rnn = torch.nn.LSTM(embedding_size, hidden_size).cuda()
    
    # TODO: add the relevant initialization code for the Attention mechanism
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.embedding_size = embedding_size
    self.out1 = torch.nn.Linear(hidden_size, output_size)
    self.v = torch.randn(1,hidden_size).cuda()
    self.new_linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    self.my_tanh = nn.Tanh()
    self.my_softmax = torch.nn.Softmax(dim=0)

  def forward(self, single_sentence):
    # single_sentence is a 1D tensor containing indices of the sentence characters
    embedded = self.embedding(single_sentence)
    embedded = embedded.view(len(single_sentence), 1, -1)
    out, hidden = self.rnn(embedded) # Passing the concatenated vector as input to the RNN cell
    
    # TODO: calculate the context vector, which is a weighted average of the out vectors, with weights learned automatically
    o = self.my_tanh(self.new_linear(out)) # Calculating Alignment Scores
    o = o.view(o.shape[0], o.shape[2])

    c = torch.mm(self.v,o.transpose(0,1))
    alpha = self.my_softmax(c) 
    context = out.squeeze(dim=1) * alpha.transpose(0,1)
    context = torch.sum(context, dim=0)
    
    return self.out1(context)

  def init_hidden(self):
    return (torch.zeros(self.hidden_size))

"""# Task 2
Once completed, you are now requested to write some code for training the model using the following configuration. Make sure to print the training loss every 100 sentences so you can follow. Train your code for 4 epochs, and use cuda + GPU.
"""

model = SeqModel(vocab.n_chars, 64, 300, 2).cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
print(model)

# Your training code goes here

device = torch.device('cuda')

n_epochs = 4
print_every = 100
total_loss = []

for e in range(1, n_epochs + 1):
    train_shuff = sample(train, len(train))
    sentence_loss = 0.0
    epoch_loss = 0
    
    for counter, sequence in enumerate(train_shuff):
      optimizer.zero_grad()
      seq_len = len(sequence)
      sequence_tensor = torch.LongTensor(sequence[0]).to(device) 
      label = torch.empty(1).to(device) 
      label.fill_(sequence[1])
      label = label.type(torch.long)
      #print(sequence_tensor, label)

      hidden = model.init_hidden() #maybe add to SeqModel
      output = model(sequence_tensor)
      tmp_loss = criterion(output.view(1,2), label)
      tmp_loss.backward()
      optimizer.step()

      loss = tmp_loss.item()
      sentence_loss += loss
      epoch_loss += loss

      if counter % print_every == 0:
          new_loss = sentence_loss / print_every
          print('Epoch %d/%d, Current Loss = %.4f' % (e, counter, new_loss))
          #print('Epoch %d, %d/%d, Current Loss = %.4f' % (e, counter, len(data_sequences_shuff), loss))
          sentence_loss = 0
    
    total_loss.append(epoch_loss / counter)

"""# Task 3
Write some code for evaluating your model on the dev set. Since the data is almost balanced (there are 52 positives in the dev set), let's print accuracy (i.e., the number of correctly classified instances).
"""

# Your evalutation code goes here
n_correct = 0
n_samples = 0
with torch.no_grad():
    for counter, sequence in enumerate(dev):
        sequence_tensor = torch.LongTensor(sequence[0]).cuda()
        outputs = model(sequence_tensor)
        if sequence[1] == torch.argmax(outputs.data).item():
            n_correct += 1
        n_samples += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the dev set: {acc} %')
