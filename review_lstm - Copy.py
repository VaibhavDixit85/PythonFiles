#!/usr/bin/env python
# coding: utf-8

# In[34]:


import torch
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import re
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from gensim.models import KeyedVectors

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns


# In[35]:


import json
import pandas as pd

data_df = pd.read_json('D:/ISB_AMPBA/Term4/Deep_Learning/Tutorial/reviews (1)/reviews.json', lines=True)


# In[36]:


df = pd.DataFrame(data_df)


# In[37]:


df.head()


# In[38]:


df.rename(columns = {'overall' : 'Rating', 'reviewText' : 'Review'}, inplace = True)


# In[39]:


df=df[['Rating','Review']]


# In[40]:


"""
Let's take a quick look at the distribution of different ratings
"""
rating_frequency_count = df.Rating.value_counts()
sns.barplot(x=rating_frequency_count.index, y=rating_frequency_count.values)


# In[41]:


"""
Now let's find out a value for maxlen. For that, we first look at the distribution of review lengths in terms of 
number of words. A box plot is used to visualize this distribution.
"""

re_wordMatcher = re.compile(r'[a-z0-9]+') #Declare regex to extract words
numWords = df["Review"].map(lambda x: len(re_wordMatcher.findall(x.lower())))
g = sns.boxplot(numWords)


# In[42]:


"""
The above plot shows that there are few very long reviews (black dots on the right) but most of the reviews are
comparatively shorter than around 250 words. Specifically, let's find the 90th quantile of the review length. 
"""

reviewLen90 = np.quantile(numWords, 0.90)
print("90th quantile of review length:", reviewLen90)


# In[43]:


"""
Thus, 90% of reviews are of 191 words or shorter. We'll set maxlen close to this.
"""
maxlen = 213


# In[44]:


"""
Lets's create training and test datasets by keeping ratio between the positive and negative labels same.
We use sklearn.model_selection.StratifiedKFold setting number of folds (n_splits) = 5, which splits the data into
80% train and 20% test and for 5 folds. But we keep only the first fold for this demo.
"""
labels = np.array(df["Rating"])
skf = StratifiedKFold(n_splits=5)
for trn_idx, tst_idx in skf.split(labels.reshape((-1, 1)), labels):
    break

train_df, test_df = df.iloc[trn_idx], df.iloc[tst_idx]

print("Shape of train and test dataframes:", train_df.shape, test_df.shape)


# #### Data Processing: Review & Sentiment to Numeric Features & Labels for Model Training & Evaluation

# In[45]:


#Read FastText En model. If the file wiki.multi.en.vec' does not exist, download it from 
# https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec
get_ipython().system('pip install wget')
import wget
word2VecFile = os.path.join(os.curdir, 'wiki.multi.en.vec')

if os.path.exists(word2VecFile):
    print('Word2Vec file has been found and is being loaded...')
else:    
    print('Word2Vec file does not exist and needs to be downloaded')
    url = 'https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec'
    wget.download(url)
    print('Downloading from', url)
en_model = KeyedVectors.load_word2vec_format('wiki.multi.en.vec')


# In[46]:


embeddingDim = 300


# In[47]:


"""
Now let us create a numpy array containing the word vectors. Later this numpy array will be used for initilaizing 
the embedding layer in the model.
"""

vocab = list(en_model.vocab.keys())
print("Vocab size in pretrained model:", len(vocab))

# check if the word 'and' is present in the pretrained model
assert "and" in en_model

# check the dimension of the word vectors
assert embeddingDim == len(en_model["and"])

# initialize a numpy matrix which will store the word vectors
# first row is for the padding token
pretrained_weights = np.zeros((1+len(vocab), embeddingDim))

# tqdm just adds a progress bar
for i, token in enumerate(vocab):
    pretrained_weights[i, :] = en_model[token]

# map tokens in the vocab to ids
vocab = dict(zip(vocab, range(1, len(vocab)+1)))


# In[48]:


def reviewText2Features(reviewText):
    """
    Function which takes review text (basically a string!) as input and returns a features matrix X of shape
    (maxlen, embeddingDim). This is done by splitting the review into words and then representing each word by it's
    word vector obtained from the Word2Vec model. Sentences having more than maxlen words are truncated while shorter
    ones are zero-padded by pre-adding all zero vectors.
    """
    X = []
    
    reviewWords = re_wordMatcher.findall(reviewText.lower())
    
    """
    Tokenize the review using the word-matching regex and get its word vector from the pretrained Word2Vec model.
    Words not found in the Word2Vec model are ignored
    """
    for i, word in enumerate(reviewWords):
        if word not in en_model:
            continue
        if i >= maxlen:
            break
        # X.append(en_model[word])
        X.append(vocab[word])
    
    """
    Add zero padding in the begining of the sequence if the number of words is less than maxlen.
    """
    if len(X) < maxlen:
        # zero_padding = [[0.]*embeddingDim]*(maxlen - len(X))
        zero_padding = [0.]*(maxlen - len(X))
        X = zero_padding + X
    
    return X # np.array(X)
        
def row2Features(row):
    """
    Function which takes a datafram row as input and produces features and labels.
    
    Input: row | Type: pandas.core.series.Series
    
    Output: X, y | Type: X - np.ndarray of shape (maxlen, embeddingDim) & y - int where Positive = 0 & Negative = 1
    """    
    
    X = reviewText2Features(row["Review"])
    y = row["Rating"]
        
    return X, y


# In[49]:


"""
Now apply the above function on a sample row
"""
sampleRow = df.iloc[0]
reviewWords = re_wordMatcher.findall(sampleRow["Review"].lower())
print("Review:", sampleRow["Review"])
print("Rating:", sampleRow["Rating"])
print("Review words:", reviewWords)


# In[50]:


sampleRow


# In[51]:


"""
Give the sample row to the function row2Features
"""
X, y = row2Features(sampleRow)
print("Dimension of X:", len(X))
print("Label y:", y)


# In[52]:


def shuffleArray(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx, :]
    y = y[idx]
    return X, y

def generateModelReadyData(data, batchSize = 128, shuffle=False):
    """
    Generator function which generates features and labels in batches
    
    Input:
    data - DataFrame where each row has review and sentiment
    batchSize - No. of rows for which features will be created and returned in a batch.
    Note: This is useful for running mini-batch Gradient Descent optimization when the dataset is large.
    
    Output:
    X - 3D np.ndarray of shape (batchSize, maxlen, embeddingDim)
    y - 1D np. array of shape (batchSize,)        
    """
    
    while(True):
        X = []
        y = []
        for _, row in data.iterrows():
            """Generate features and label for this row"""
            X_, y_ = row2Features(row)

            """Keep accumulating the row-wise features"""
            X.append(X_)
            y.append(y_)   

            """If number of rows processed is greater than batchSize yield the batch and trim down X & y
            Note: This way we avoid running into memory issues by not bloating X and y bigger and bigger
            """
            if len(X) > batchSize:
                temp_X, temp_y = np.array(X[:batchSize]), np.array(y[:batchSize])
                if shuffle:
                    temp_X, temp_y = shuffleArray(temp_X, temp_y)
                
                X, y = X[batchSize:], y[batchSize:]                    
                yield temp_X, temp_y

        """Yield the remaining few rows when number of rows in data isn't a mutiple of batchSize"""
        if len(X) > 0:
            temp_X, temp_y = np.array(X), np.array(y)
            if shuffle:
                temp_X, temp_y = shuffleArray(temp_X, temp_y)
            
            yield temp_X, temp_y


# In[53]:


"""Let's test the generator function for few batches"""
numBatches = 0
for i, (X, y) in enumerate(generateModelReadyData(df, batchSize=128, shuffle=True)):
    if numBatches >= 3:
        break
    
    else:
        print("Batch:", i)
        assert X.shape == (128, maxlen)
        assert y.shape == (128,)
        print("Shape of X & y matches expected values")
    numBatches += 1


# #### Model Training in Pytorch

# In[54]:


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    print("cuda available")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# In[55]:


"""
Set random number seed using torch.manual_seed to make sure the same seed is used
by the Pytorch backend and hence ensure repeatable results
"""
torch.manual_seed(0)


# In[56]:


class SentimentNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, pretrained_weights):
        super(SentimentNet, self).__init__()
        
        self.embedding=nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))
        
        """
        Adding a dropout layer to force some of the feature values to zero.
        Note: Dropout is a regularization technique which sets the activation of few randomly chosen neurons of
        a hidden layer to zero. It can also be applied to the input layer where some of the input features are set to zero.
        For more details refer http://jmlr.org/papers/v15/srivastava14a.html
        """
        self.sentInputDropout = nn.Dropout(0.1)
        
        """
        Now let's stack a couple of bidirectional RNNs to process the input sequence and extract features
        """
        self.biLSTM1 = nn.LSTM(embedding_dim, hidden_dim[0], bidirectional=True, batch_first=True)
        self.biLSTMDropOut = nn.Dropout(0.1)
        
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(2*hidden_dim[0], 50)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        
        self.outputLayer = nn.Linear(50, 5)
        self.sigmoid = nn.Sigmoid()
        
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        
        batch_len = x.shape[0]
        out = self.embedding(x)
        out = self.sentInputDropout(out)
        out, hidden = self.biLSTM1(out)
        out = self.biLSTMDropOut(out)
        
        out = self.dense1(out)
        out = self.relu1(out)
        out = self.dropout2(out)
        
        out = self.outputLayer(out)
        out = self.sigmoid(out)
        out = out.view(batch_len, -1)
        out = out[:,-1]
        return out    


# In[57]:


model = SentimentNet(embeddingDim, [100], 1+len(vocab), pretrained_weights)
model.to(device)


# In[58]:


lr=0.005
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# In[59]:


epochs = 3
counter = 0
print_every = 1000
clip = 5
valid_loss_min = np.Inf

model = model.float()
model.train()
for i in range(epochs):
    print("Epoch:", i+1)
    #h = model.init_hidden(128)
    print("Running a pass over the training data...")
    for j, (inputs, labels) in enumerate(generateModelReadyData(train_df, batchSize=128, shuffle=True)):
        if j >= np.ceil(train_df.shape[0]/128):
            break
        
    #for inputs, labels in train_loader:
        counter += 1
        #h = tuple([e.data for e in h])
        inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        #output, h = model(inputs, h)
        output = model(inputs.long())
        #print(output.shape)
        #print(output)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()        
        if (j+1) % 100 == 0:
            print("Batches completed:", j+1)
    
    print("Batches completed:", j+1)

    #val_h = model.init_hidden(batch_size)
    val_losses = []
    model.eval()
    print("Running a pass over the test data...")
    for k, (inp, lab) in enumerate(generateModelReadyData(test_df, batchSize=128, shuffle=False)):
        if k >= np.ceil(test_df.shape[0]/128):
            break
    #for inp, lab in val_loader:
        #val_h = tuple([each.data for each in val_h])
        inp, lab = torch.from_numpy(inp), torch.from_numpy(lab)
        inp, lab = inp.to(device), lab.to(device)
        out = model(inp.long())
        val_loss = criterion(out.squeeze(), lab.float())
        val_losses.append(val_loss.item())
        if (k+1) % 100 == 0:
            print("Batches completed:", k+1)
    
    print("Batches completed:", k+1)

    model.train()
    print("Epoch: {}/{}...".format(i+1, epochs),
          "Step: {}...".format(counter),
          "Loss: {:.6f}...".format(loss.item()),
          "Val Loss: {:.6f}".format(np.mean(val_losses)))
    if np.mean(val_losses) <= valid_loss_min:
        torch.save(model.state_dict(), './state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
        valid_loss_min = np.mean(val_losses)


# In[60]:


"""
At this point we can load a pretrained model which was trained for 5 epochs and make predictions using it.
Uncomment and run the below line to load the pretrained model
"""
model.load_state_dict(torch.load('./state_dict.pt'))
model.to(device)


# In[62]:


test_losses = []
num_correct = 0
pred_proba = []
actual = []

model.eval()
for j, (X_test, y_test) in enumerate(generateModelReadyData(test_df, batchSize=128)):
    if j >= np.ceil(test_df.shape[0]/128):
        break
    
    inputs_test, labels_test = torch.from_numpy(X_test), torch.from_numpy(y_test)
    inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
    output_test = model(inputs_test.long())
    test_loss = criterion(output_test.squeeze(), labels_test.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output_test.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(labels_test.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)
    pred_proba.extend(output_test.cpu().squeeze().detach().numpy())
    actual.extend(y_test)
    
    if (j+1) % 100 == 0:
        print("Batches completed:", j+1)

print("Batches completed:", j+1)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_df)
print("Test accuracy: {:.3f}%".format(test_acc*100))


# In[63]:


ls = [6.827,6.827,6.827]


# In[65]:


## The mean accuracy with seed 0,1,2 is 6.827%
np.mean(ls)


# In[66]:


## The mean accuracy with seed 0,1,2 is 6.827%
np.std(ls)


# In[ ]:




