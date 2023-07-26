#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json


# In[2]:


train_data = pd.read_csv("data/train", delimiter = "\t",  names= ['index', 'word', 'pos'], header= None)
dev_data = pd.read_csv("data/dev", delimiter = "\t",  names= ['index', 'word', 'pos'], header= None)
test_data = pd.read_csv("data/test", delimiter = "\t", names = ['index', 'word'], header = None)


# In[3]:


train_data


# ## Task 1: Vocabulary Creation

# In[4]:


train_words = train_data["word"].values
train_indexes = train_data["index"].values
train_pos_tags = train_data["pos"].values
train_words_dict = dict()

# Count words in the training set.

for word in train_words:
    train_words_dict[word] = train_words_dict.get(word,0) + 1


# In[17]:


unknown_count = 0
common_vocab = dict()
unknown_words = []

# Find words with less than 3 occurences and store the counts.

for word, count in train_words_dict.items():
    if count < 3:
        unknown_words.append(word)
        unknown_count = unknown_count + count
    else:
        common_vocab[word] = count


# In[29]:


print("Threshold for unknown words is: 3")


# In[21]:


# Sort the words by occurrences and create the vocabulary.

final_words = ["< unk >"]
final_counts = [unknown_count]
for word, count in sorted(common_vocab.items(), key = lambda item: item[1], reverse = True):
    final_words.append(word)
    final_counts.append(count)

index = range(1,len(final_words) + 1)


# In[23]:


final_vocabulary = pd.DataFrame(zip(final_words, index, final_counts), columns = ["word", "index", "occurences"])


# In[25]:


final_vocabulary


# In[26]:


# Export vocabulary as txt.

final_vocabulary.to_csv("vocab.txt", sep = "\t", index = False, header = False)


# In[27]:


print("Length of the vocabulary is:", len(final_words))


# ## Task 2: Model Learning

# In[10]:


# Create transition dictionary.

transition_dict = dict()
transition_json_dict = dict()
initial_dict = dict()
sentences_count = 0
pos_dict = dict()
transition_count = dict()

# Count (s'|s), with s = "< start >" for the beginning tags.

for i in range(0, len(train_pos_tags) - 1):
    index = train_indexes[i]
    
    if index == 1:
        sentences_count += 1
        transition = ("< start >", train_pos_tags[i])
    else:
        transition = (train_pos_tags[i-1], train_pos_tags[i])
    
    pos_dict[train_pos_tags[i]] = pos_dict.get(train_pos_tags[i], 0) + 1

    transition_count[transition] = transition_count.get(transition, 0) + 1
    
# Populate the transition dictionary with probabilities.

for key, value in transition_count.items():
    json_key = "(" + key[0] + "," + key[1] + ")"
    if key[0] == "< start >":
        transition_json_dict[json_key] = value / sentences_count
        transition_dict[key] = value / sentences_count
    else:
        transition_json_dict[json_key] = value / pos_dict[key[0]]
        transition_dict[key] = value / pos_dict[key[0]]


# In[28]:


print("Number of transition parameters is:", len(transition_dict))


# In[11]:


# Create emission dictionary.

emission_dict = dict()
emission_count = dict()
emission_dict_json = dict()

# Count (word | s) with word = "< unk >" for words whose occurrences are < 3 in the training set.

for i in range(0, len(train_pos_tags) - 1):
    if train_words[i] in unknown_words:
        emission = (train_pos_tags[i], '< unk >')
    else:
        emission = (train_pos_tags[i], train_words[i])
        
    emission_count[emission] = emission_count.get(emission, 0) + 1

# Populate the emission dictionary with probabilities.

for key, value in emission_count.items():
    json_key = "(" + key[0] + "," + key[1] + ")"
    emission_dict[key] = value / pos_dict[key[0]]
    emission_dict_json[json_key] = value / pos_dict[key[0]]


# In[ ]:


print("Number of emission parameters is:", len(emission_dict))


# In[12]:


# Export model as a json.

hmm_dict = dict()
hmm_dict["emission"] = emission_dict_json
hmm_dict["transition"] = transition_json_dict

with open('hmm.json', 'w') as f:
    json.dump(hmm_dict, f)


# ## Task 3: Greedy Decoding with HMM

# In[13]:


dev_words = dev_data["word"].values
dev_indexes = dev_data["index"].values
dev_pos_tags = dev_data["pos"].values


# In[14]:


test_words = test_data["word"].values
test_indexes = test_data["index"].values


# In[15]:


def greedy_decoding(words):
    predicted_tag = []
    for i in range(len(words)):
        if words[i] not in final_words:
            word = "< unk >"
        else:
            word = words[i]
        largest_prob = 0
        likely_tag = ""
        
        # If its the first word, look for (s'|<start>).
        
        if dev_indexes[i] == 1:
            for key in transition_dict.keys():
                if key[0] == "< start >":
                    
                    # Calculate t(s'|start) * e(word|s') and keep the maximum value of this term.
                    
                    try:
                        prob = transition_dict[key] * emission_dict[(key[1], word)]
                    except:
                        prob = 0
                    if prob > largest_prob:
                        largest_prob = prob
                        likely_tag = key[1]
                        
        # Else, consider the latest tag as s.
        
        else:
            for key in transition_dict.keys():
                if key[0] == latest_tag:
                    
                    # Calculate t(s'|s) * e(word|s') and keep the maximum value of this term.
                    
                    try:
                        prob = transition_dict[key] * emission_dict[(key[1], word)]
                    except:
                        prob = 0
                    if prob > largest_prob:
                        largest_prob = prob
                        likely_tag = key[1]
                        
        # Update the tags.
        
        if largest_prob != 0:
            latest_tag = likely_tag
        predicted_tag.append(likely_tag)
        
    return predicted_tag


# In[16]:


# Function to calculate the accuracy of the predicted tags.

def accuracy(predictions, true):
    n = len(true)
    right = 0
    for i in range(len(true)):
        if true[i] == predictions[i]:
            right += 1
    return right / n


# In[17]:


greedy_dev_predicted_tags = greedy_decoding(dev_words)


# In[18]:


print("Accuracy of the Greedy algorithm on the dev set is:", accuracy(greedy_dev_predicted_tags, dev_pos_tags))


# In[19]:


greedy_test_predicted_tags = greedy_decoding(test_words)


# In[20]:


# Write results to file.

greedy_out = open("greedy.out", "w")
for i in range(0, len(test_words)):
    output = "\t".join([str(test_indexes[i]), test_words[i], greedy_test_predicted_tags[i]])
    greedy_out.write(output + "\n")
    try:
        if test_indexes[i+1] == 1:
            greedy_out.writelines("\n")
    except: 
        continue

greedy_out.close()


# ## Task 4: Viterbi Decoding with HMM

# In[21]:


def viterbi_decoding(words, indexes):
    predicted_tags = []
    
    s = list(train_data["pos"].value_counts().index)
    
    pi_matrix = dict()
    count = 0
    for i in range(len(words)):
        count +=1
        
        if words[i] not in final_words:
            word = "< unk >"
        else:
            word = words[i]
        
        # If its the first word of the sentence, calculate the first row of the matrix.
        
        if indexes[i] == 1:
            for j in range(len(s)):
                try:
                    pi_matrix[(0,j)] = transition_dict[("< start >", s[j])] * emission_dict[(s[j], word)]
                except:
                    pi_matrix[(0,j)] = 0
        
        # Else, use the previous row's values to calculate the values for this current row.
        
        else:
            for k in range(len(s)):
                max_prob = []
                for l in range(len(s)):
                    try:
                        max_prob.append(pi_matrix[(indexes[i] - 2, l)] * transition_dict[(s[l], s[k])] * emission_dict[(s[k], word)])
                    except:
                        max_prob.append(0)
                        
                pi_matrix[(indexes[i] - 1, k)] = max(max_prob)
            
        probs = []
        
        # Find the tag that gave the maximum probability.
        
        for m in range(len(s)):
            probs.append(pi_matrix[(indexes[i] - 1, m)])
        
        predicted_tags.append(s[np.argmax(probs)])

    return predicted_tags


# In[22]:


# def viterbi_decoding(sentences):
#     predicted_tags = []
#     train_pos = list(train_data["pos"].values)
#     s = list(set(train_pos))
    
#     for sentence in sentences:
#         n = len(sentence)
#         pi = np.zeros(shape=(n, len(s)))
        
#         for i in range(len(s)):
#             if sentence[0] not in final_words:
#                 word = "< unk >"
#             else:
#                 word = sentence[0]
                
#             try:
#                 pi[0, i] = transition_dict[("< start >", s[i])] * emission_dict[(s[i], word)]
#             except:
#                 pi[0, i] = 0
#         for j in range(1, n):
#             if sentence[j] not in final_words:
#                 word = "< unk >"
#             else:
#                 word = sentence[j]
            
#             for k in range(len(s)):
#                 max_prob = []
#                 for ss in range(len(s)):
#                     try:
#                         max_prob.append(pi[j-1, ss] * transition_dict[(s[ss], s[k])] * emission_dict[(s[k], word)])
#                     except:
#                         max_prob.append(0)
#                 pi[j,k] = max(max_prob)
        
#         for i in range(len(pi)):
#             predicted_tags.append(s[np.argmax(pi[i])])
            
#     return predicted_tags


# In[23]:


# dev_sentences = []
# new_sentence = [dev_words[0]]

# for i in range(1, len(dev_words)):
#     if dev_indexes[i] == 1:
#         dev_sentences.append(new_sentence)
#         new_sentence = [dev_words[i]]
#     else:
#         new_sentence.append(dev_words[i])
#     if i == len(dev_words) - 1:
#         dev_sentences.append(new_sentence)

# test_sentences = []
# new_sentence = [test_words[0]]

# for i in range(1, len(test_words)):
#     if test_indexes[i] == 1:
#         test_sentences.append(new_sentence)
#         new_sentence = [test_words[i]]
#     else:
#         new_sentence.append(test_words[i])
#     if i == len(test_words) - 1:
#         test_sentences.append(new_sentence)


# In[24]:


# dev_predicted_tags = viterbi_decoding(dev_sentences)


# In[25]:


viterbi_dev_predicted_tags = viterbi_decoding(dev_words, dev_indexes)


# In[26]:


print("Accuracy of the Viterbi algorithm on the dev set is:", accuracy(viterbi_dev_predicted_tags, dev_pos_tags))


# In[27]:


viterbi_test_predicted_tags = viterbi_decoding(test_words, test_indexes)


# In[28]:


# Write results to file.

viterbi_out = open("viterbi.out", "w")
for i in range(0, len(test_words)):
    output = "\t".join([str(test_indexes[i]), test_words[i], viterbi_test_predicted_tags[i]])
    viterbi_out.write(output + "\n")
    try:
        if test_indexes[i+1] == 1:
            viterbi_out.writelines("\n")
    except: 
        continue

viterbi_out.close()

