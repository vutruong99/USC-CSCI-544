#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import csv
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 


# In[2]:


if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
    
device = torch.device(dev)  


# ## Task 1: Simple Bidirectional LSTM model

# In[319]:


# Read train data.

train_indexes = []
train_words = []
train_tags = []
with open("data/train") as f:
    for l in f:
        if l != "\n":
            l = l.strip()
            index = l.split(" ")[0]
            word = l.split(" ")[1]
            tag = l.split(" ")[2]
            
            train_indexes.append(index)
            train_words.append(word)
            train_tags.append(tag)


# In[320]:


# Read dev data.

dev_indexes = []
dev_words = []
dev_tags = []
with open("data/dev") as f:
    for l in f:
        if l != "\n":
            l = l.strip()
            index = l.split(" ")[0]
            word = l.split(" ")[1]
            tag = l.split(" ")[2]
            
            dev_indexes.append(index)
            dev_words.append(word)
            dev_tags.append(tag)


# In[334]:


# Read test data.

test_indexes = []
test_words = []
with open("data/test") as f:
    for l in f:
        if l != "\n":
            l = l.strip()
            index = l.split(" ")[0]
            word = l.split(" ")[1]
            
            test_indexes.append(index)
            test_words.append(word)


# In[335]:


# Convert data to dataframes.

train_data = pd.DataFrame(zip(train_indexes, train_words, train_tags), columns = ["index", "word", "tag"])
dev_data = pd.DataFrame(zip(dev_indexes, dev_words, dev_tags), columns = ["index", "word", "tag"])
test_data = pd.DataFrame(zip(test_indexes, test_words), columns = ["index", "word"])


# In[323]:


train_words = train_data["word"].values
train_indexes = train_data["index"].values
train_pos_tags = train_data["tag"].values
train_words_dict = dict()

# Count words in the training set.

for word in train_words:
    train_words_dict[word] = train_words_dict.get(word,0) + 1


# In[324]:


common_vocab = dict()
unknown_words = []

# Find words with less than 3 occurences and store the counts.

for word, count in train_words_dict.items():
    if count <= 3:
        unknown_words.append(word)
    else:
        common_vocab[word] = count


# In[325]:


# Create the vocabulary.

final_words = ["< pad >", "< unk >"]
for word, count in sorted(common_vocab.items(), key = lambda item: item[1], reverse = True):
    final_words.append(word)

index = range(0,len(final_words))
final_vocabulary = pd.DataFrame(zip(final_words, index), columns = ["word", "index"])
final_vocabulary.index = final_vocabulary.index + 1  # shifting index
final_vocabulary = final_vocabulary.sort_index() 


# In[327]:


# Map word to index.

final_vocabulary_dictionary = final_vocabulary[["word","index"]].set_index("word").T.to_dict("list")


# In[329]:


# Get vocab size.

vocab_size = len(final_vocabulary_dictionary)


# In[330]:


# Create tags dictionary.

tags_dict = dict()
tags = list(train_data["tag"].unique())
for i, tag in enumerate(tags):
    tags_dict[tag] = i+1


# In[331]:


# Create inverse tags dictionary.

tags_dict_inv = {v: k for k, v in tags_dict.items()}


# In[332]:


# Convert tags to index.

new_tags = []
for tag in train_tags:
    new_tags.append(tags_dict[tag])
    
dev_new_tags = []
for tag in dev_tags:
    dev_new_tags.append(tags_dict[tag])


# In[333]:


# Map word to index (train).
# Unknown word = 1.
# Else, map to dictionary.

word_indexes = []
true_ners = []
sentence = []
temp_ner = []
for i in range(0, len(train_words)):
    ner = new_tags[i]
    try:
        word = final_vocabulary_dictionary[train_words[i]][0]
        
    except:
        word = 1

    if i == len(train_words) - 1:
        temp_ner.append(ner)
        true_ners.append(temp_ner)
        sentence.append(word)
        word_indexes.append(sentence)
        break

    sentence.append(word)
    temp_ner.append(ner)
    if train_indexes[i+1] == "1":
        word_indexes.append(sentence)
        true_ners.append(temp_ner)
        sentence = []
        temp_ner = []


# In[240]:


# Map word to index (dev).

dev_word_indexes = []
dev_true_ners = []
sentence = []
temp_ner = []
for i in range(0, len(dev_words)):
    ner = dev_new_tags[i]
    try:
        word = final_vocabulary_dictionary[dev_words[i]][0]
        
    except:
        word = 1

    if i == len(dev_words) - 1:
        temp_ner.append(ner)
        dev_true_ners.append(temp_ner)
        sentence.append(word)
        dev_word_indexes.append(sentence)
        break
        
    sentence.append(word)
    temp_ner.append(ner)
    if dev_indexes[i+1] == "1":
        dev_word_indexes.append(sentence)
        dev_true_ners.append(temp_ner)
        sentence = []
        temp_ner = []


# In[241]:


# Map word to index (test).

test_word_indexes = []
sentence = []
for i in range(0, len(test_words)):
    try:
        word = final_vocabulary_dictionary[test_words[i]][0]
        
    except:
        word = 1

    if i == len(test_words) - 1:
        sentence.append(word)
        test_word_indexes.append(sentence)
        break
        
    sentence.append(word)
    if test_indexes[i+1] == "1":
        test_word_indexes.append(sentence)
        sentence = []


# In[242]:


# Function to prepare datasets.

class PrepareDataset(Dataset):
    def __init__(self, word_index, label):
        self.features = word_index
        self.labels = label
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return torch.LongTensor(feature).to(device), torch.LongTensor(label).to(device)


# In[243]:


# Function to prepare datasets.

class PrepareTestDataset(Dataset):
    def __init__(self, word_index):
        self.features = word_index
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        feature = self.features[index]
        return torch.LongTensor(feature).to(device)


# In[244]:


# Function to pad the sentence with 0s.

def collate_fn(data):
    sequences, labels = zip(*data)
    length = [len(seq) for seq in sequences]
    padded_seq = torch.zeros(len(sequences), max(length)).long()
    label_seq = torch.zeros(len(sequences), max(length)).long()
    for i, seq in enumerate(zip(sequences, labels)):
        end = length[i]
        padded_seq[i,:end] = seq[0]
        label_seq[i,:end] = seq[1]
        
    return padded_seq.to(device), label_seq.to(device), torch.tensor([length]).to(device)


# In[245]:


# Function to pad the sentence with 0s for test data.

def collate_fn_test(data):
    sequences = data
    length = [len(seq) for seq in sequences]
    padded_seq = torch.zeros(len(sequences), max(length)).long()
    for i, seq in enumerate(sequences):
        end = length[i]
        padded_seq[i,:end] = seq
        
    return padded_seq.to(device), torch.tensor([length]).to(device)


# In[246]:


# Split train - val.

train_word_indexes = word_indexes[:11200]
train_word_ners = true_ners[:11200]
val_word_indexes = word_indexes[11200:]
val_word_ners = true_ners[11200:]


# In[247]:


# Prepare datasets.

train_data_lstm = PrepareDataset(train_word_indexes, train_word_ners)
val_data_lstm = PrepareDataset(val_word_indexes, val_word_ners)
dev_data_lstm = PrepareDataset(dev_word_indexes, dev_true_ners)
test_data_lstm = PrepareTestDataset(test_word_indexes)


# In[274]:


# Create data loaders.

batch_size = 32

train_loader_lstm = torch.utils.data.DataLoader(train_data_lstm, batch_size = batch_size, collate_fn = collate_fn)
val_loader_lstm = torch.utils.data.DataLoader(val_data_lstm, batch_size = batch_size, collate_fn = collate_fn)
dev_loader_lstm = torch.utils.data.DataLoader(dev_data_lstm, batch_size = batch_size, collate_fn = collate_fn)
test_loader_lstm = torch.utils.data.DataLoader(test_data_lstm, batch_size = batch_size, collate_fn = collate_fn_test)


# In[275]:


# BLSTM architecture.

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout_p):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 100, padding_idx = 0)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.linear = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(p = dropout_p)
        self.elu = nn.ELU()
        self.lstm = nn.LSTM(input_size, hidden_dim // 2, n_layers, batch_first=True, bidirectional=True)   
        self.fc = nn.Linear(output_size, 10)
    
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        cell = self.init_cell(batch_size)
        x = self.embedding(x)
        x = self.dropout(x)
        out, hidden = self.lstm(x, (hidden, cell))
        out = self.dropout(out)
        out = self.elu(self.linear(out))
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # 2 x 20 x 128
        hidden = torch.zeros(2, batch_size, self.hidden_dim // 2)
        
        return hidden.to(device)
    
    def init_cell(self, batch_size):
        cell = torch.zeros(2, batch_size, self.hidden_dim // 2)
        
        return cell.to(device)


# In[276]:

def train():
    # Create the model and initialize the weights for the tags.

    model_1 = LSTM(100, 128, 256, 1, 0.33).to(device)
    weights = [0, 1, 0.7, 1, 1, 1, 1, 1, 1, 1]
    weight_tensor = torch.FloatTensor(weights).to(device)

    # In[277]:

    # Initialize loss function and optimizer.

    criterion = nn.CrossEntropyLoss(ignore_index=0, weight=weight_tensor, )

    optimizer = torch.optim.SGD(model_1.parameters(), lr=0.5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    # In[278]:

    # Model training.

    n_epochs = 105
    valid_loss_min = np.Inf

    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0

        model_1.train()  # prep model for training
        for data, target, _ in train_loader_lstm:
            optimizer.zero_grad()
            output, hidden = model_1(data)
            output = output.contiguous().view(-1, 10)
            target = target.contiguous().view(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        model_1.eval()  # prep model for evaluation
        for data, target, _ in val_loader_lstm:
            output, hidden = model_1(data)
            output = output.contiguous().view(-1, 10)
            target = target.contiguous().view(-1)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader_lstm.dataset)
        valid_loss = valid_loss / len(val_loader_lstm.dataset)
        scheduler.step(valid_loss)

        print('Epoch: {}/{} \tTraining Loss: {:.4f} \tValidation Loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss,
                                                                                      valid_loss))

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model_1.state_dict(), 'blstm1.pt')
            valid_loss_min = valid_loss
            
    return

def test():
    # In[279]:

    # Load the trained model for task 1.

    blstm1 = LSTM(100, 128, 256, 1, 0.33).to(device)
    blstm1.load_state_dict(torch.load("blstm1.pt", map_location=torch.device(dev)))
    blstm1.eval()

    # In[280]:

    # Get predictions on dev set.

    prediction_list = []
    for i, batch in enumerate(dev_loader_lstm):
        prediction = []
        lengths = []
        output, hidden = blstm1(batch[0])
        for b in batch[0]:
            lengths.append((torch.count_nonzero(b).item()))
        for i, res in enumerate(output):
            for j in range(lengths[i]):
                prediction.append(torch.argmax(res[j]).item())
        prediction_list.append(prediction)

    # In[281]:

    # Map predictions to tags.

    y_pred = []
    for item in prediction_list:
        y_pred = y_pred + item

    predictions = []
    for pred in y_pred:
        predictions.append(tags_dict_inv[pred])

    # In[282]:

    res = open("dev1.out", "w")
    for i in range(0, len(dev_words)):
        output = " ".join([str(dev_indexes[i]), dev_words[i], dev_tags[i], predictions[i]])
        res.write(output + "\n")
        try:
            if dev_indexes[i + 1] == "1":
                res.writelines("\n")
        except:
            continue

    res.close()

    # In[283]:

    # Get predictions on test set.

    prediction_list = []
    for i, batch in enumerate(test_loader_lstm):
        prediction = []
        lengths = []
        output, hidden = blstm1(batch[0])
        for b in batch[0]:
            lengths.append((torch.count_nonzero(b).item()))
        for i, res in enumerate(output):
            for j in range(lengths[i]):
                prediction.append(torch.argmax(res[j]).item())
        prediction_list.append(prediction)

    # In[284]:

    # Map predictions to tags.

    y_pred = []
    for item in prediction_list:
        y_pred = y_pred + item

    predictions = []
    for pred in y_pred:
        predictions.append(tags_dict_inv[pred])

    # In[285]:

    res = open("test1.out", "w")
    for i in range(0, len(test_words)):
        output = " ".join([str(test_indexes[i]), test_words[i], predictions[i]])
        res.write(output + "\n")
        try:
            if test_indexes[i + 1] == "1":
                res.writelines("\n")
        except:
            continue

    res.close()
    
    return

if __name__ == '__main__':
    # train()
    test()






