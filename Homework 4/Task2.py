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

train_data = pd.DataFrame(zip(train_indexes, train_words, train_tags), columns=["index", "word", "tag"])
dev_data = pd.DataFrame(zip(dev_indexes, dev_words, dev_tags), columns=["index", "word", "tag"])
test_data = pd.DataFrame(zip(test_indexes, test_words), columns=["index", "word"])

# ## Task 2: Using GloVe word embeddings

# In[286]:

# Read Glove file.

glove_words = []
glove_vectors = []
glove_dictionary = dict()
with open("glove.6B.100d.txt", encoding = "utf8") as f:
    for line in f:
        vector = []
        line = line.split(" ")
        
        for element in line[1:]:
            vector.append(float(element))
        
        glove_dictionary[line[0]] = np.array(vector)
        glove_words.append(line[0])
        glove_vectors.append(np.array(vector))


# In[287]:


# Sort the words by occurrences and create the vocabulary.

final_glove_words = ["< pad >", "< unk >"]
final_glove_words = final_glove_words + glove_words
final_glove_vectors = [np.zeros(100), np.random.rand(100)]
final_glove_vectors = final_glove_vectors + glove_vectors
index = range(0,len(final_glove_words))


# In[288]:


final_glove_vocabulary = pd.DataFrame(zip(final_glove_words, index, final_glove_vectors), columns = ["word", "index", "vector"])
final_glove_vocabulary


# In[289]:


final_glove_dictionary = final_glove_vocabulary[["word","index"]].set_index("word").T.to_dict("list")


# In[291]:


glove_embeddings = []
for embedding in final_glove_vocabulary.vector.values:
    glove_embeddings.append(embedding)
    
glove_embeddings = np.array(glove_embeddings)

# Create tags dictionary.

tags_dict = dict()
tags = list(train_data["tag"].unique())
for i, tag in enumerate(tags):
    tags_dict[tag] = i+1
    
# Create inverse tags dictionary.

tags_dict_inv = {v: k for k, v in tags_dict.items()}

# Convert tags to index.

new_tags = []
for tag in train_tags:
    new_tags.append(tags_dict[tag])

dev_new_tags = []
for tag in dev_tags:
    dev_new_tags.append(tags_dict[tag])

# In[292]:


word_indexes = []
true_ners = []
sentence = []
temp_ner = []
for i in range(0, len(train_words)):
    train_word = train_words[i].lower()
    ner = new_tags[i]
    try:
        word = final_glove_dictionary[train_word][0]
        
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


# In[293]:


dev_word_indexes = []
dev_true_ners = []
sentence = []
temp_ner = []
for i in range(0, len(dev_words)):
    dev_word = dev_words[i].lower()
    ner = dev_new_tags[i]
    try:
        word = final_glove_dictionary[dev_word][0]
        
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


# In[309]:


# Map word to index (test).

test_word_indexes = []
sentence = []
for i in range(0, len(test_words)):
    test_word = test_words[i].lower()
    try:
        word = final_glove_dictionary[test_word][0]
        
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
        padded_seq[i, :end] = seq[0]
        label_seq[i, :end] = seq[1]

    return padded_seq.to(device), label_seq.to(device), torch.tensor([length]).to(device)


# In[245]:


# Function to pad the sentence with 0s for test data.

def collate_fn_test(data):
    sequences = data
    length = [len(seq) for seq in sequences]
    padded_seq = torch.zeros(len(sequences), max(length)).long()
    for i, seq in enumerate(sequences):
        end = length[i]
        padded_seq[i, :end] = seq

    return padded_seq.to(device), torch.tensor([length]).to(device)


# In[295]:


train_word_indexes = word_indexes[:11200]
train_word_ners = true_ners[:11200]
val_word_indexes = word_indexes[11200:]
val_word_ners = true_ners[11200:]


# In[310]:


train_data_lstm = PrepareDataset(train_word_indexes, train_word_ners)
val_data_lstm = PrepareDataset(val_word_indexes, val_word_ners)
dev_data_lstm = PrepareDataset(dev_word_indexes, dev_true_ners)
test_data_lstm = PrepareTestDataset(test_word_indexes)


# In[311]:


# Create data loaders.

batch_size = 32

train_loader_lstm = torch.utils.data.DataLoader(train_data_lstm, batch_size = batch_size, collate_fn = collate_fn)
val_loader_lstm = torch.utils.data.DataLoader(val_data_lstm, batch_size = batch_size, collate_fn = collate_fn)
dev_loader_lstm = torch.utils.data.DataLoader(dev_data_lstm, batch_size = batch_size, collate_fn = collate_fn)
test_loader_lstm = torch.utils.data.DataLoader(test_data_lstm, batch_size = batch_size, collate_fn = collate_fn_test)


# In[298]:


# BLSTM architecture. GLOVE

class LSTM2(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout_p):
        super(LSTM2, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(glove_embeddings))
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


def train():
    # In[299]:

    model_2 = LSTM2(100, 128, 256, 1, 0.33).to(device)
    weights = [0, 1, 0.7, 1, 1, 1, 1, 1, 1, 1]
    weight_tensor = torch.FloatTensor(weights).to(device)

    # In[300]:

    # Initialize loss function and optimizer.

    criterion = nn.CrossEntropyLoss(ignore_index=0, weight=weight_tensor, )

    optimizer = torch.optim.SGD(model_2.parameters(), lr=0.5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    # In[301]:

    # Model training.

    n_epochs = 150
    valid_loss_min = np.Inf

    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0

        model_2.train()  # prep model for training
        for data, target, _ in train_loader_lstm:
            optimizer.zero_grad()
            output, hidden = model_2(data)
            output = output.contiguous().view(-1, 10)
            target = target.contiguous().view(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        model_2.eval()  # prep model for evaluation
        for data, target, _ in val_loader_lstm:
            output, hidden = model_2(data)
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
            torch.save(model_2.state_dict(), 'blstm2.pt')
            valid_loss_min = valid_loss
    return


def test():
    # In[302]:

    # Load the trained model for task 2.

    blstm2 = LSTM2(100, 128, 256, 1, 0.33).to(device)
    blstm2.load_state_dict(torch.load("blstm2.pt", map_location=torch.device(dev)))
    blstm2.eval()

    # In[303]:

    # Get predictions on dev set.

    prediction_list = []
    for i, batch in enumerate(dev_loader_lstm):
        prediction = []
        lengths = []
        output, hidden = blstm2(batch[0])
        for b in batch[0]:
            lengths.append((torch.count_nonzero(b).item()))
        for i, res in enumerate(output):
            for j in range(lengths[i]):
                prediction.append(torch.argmax(res[j]).item())
        prediction_list.append(prediction)

    # In[304]:

    # Map predictions to tags.

    y_pred = []
    for item in prediction_list:
        y_pred = y_pred + item

    predictions = []
    for pred in y_pred:
        predictions.append(tags_dict_inv[pred])

    # In[305]:

    res = open("dev2.out", "w")
    for i in range(0, len(dev_words)):
        output = " ".join([str(dev_indexes[i]), dev_words[i], dev_tags[i], predictions[i]])
        res.write(output + "\n")
        try:
            if dev_indexes[i + 1] == "1":
                res.writelines("\n")
        except:
            continue

    res.close()

    # In[312]:

    # Get predictions on test set.

    prediction_list = []
    for i, batch in enumerate(test_loader_lstm):
        prediction = []
        lengths = []
        output, hidden = blstm2(batch[0])
        for b in batch[0]:
            lengths.append((torch.count_nonzero(b).item()))
        for i, res in enumerate(output):
            for j in range(lengths[i]):
                prediction.append(torch.argmax(res[j]).item())
        prediction_list.append(prediction)

    # In[313]:

    # Map predictions to tags.

    y_pred = []
    for item in prediction_list:
        y_pred = y_pred + item

    predictions = []
    for pred in y_pred:
        predictions.append(tags_dict_inv[pred])

    # In[314]:

    res = open("test2.out", "w")
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

