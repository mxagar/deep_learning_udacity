# Recurrent Neural Networks (RNN)

These are my personal notes taken while following the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).

The nanodegree is composed of six modules:

1. Introduction to Deep Learning
2. Neural Networks and Pytorch Guide
3. Convolutonal Neural Networks (CNN)
4. Recurrent Neural Networks (RNN)
5. Generative Adversarial Networks (GAN)
6. Deploying a Model

Each module has a folder with its respective notes. This folder is the one of the **fourth module**: Recurrent Neural Networks.

Additionally, note that:

- I made many hand-written nortes, which I will scan and push to this repostory.
- I forked the Udacity repository for the exercisesl [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch); all the material and  notebooks are there.
- This module in particular is mostly covered in the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891). I have a repository with notes on it: [computer_vision_udacity](https://github.com/mxagar/computer_vision_udacity), module `03_Advanced_CV_and_DL`.

## Overview of Contents

1. Recurrent Neural Networks (RNNs): Covered in [computer_vision_udacity](https://github.com/mxagar/computer_vision_udacity), module `03_Advanced_CV_and_DL`.
2. Long Short-Term Memory Networks (LSTM): Covered in [computer_vision_udacity](https://github.com/mxagar/computer_vision_udacity), module `03_Advanced_CV_and_DL`.
3. Implementation of RNNs and LSTMs: Covered in [computer_vision_udacity](https://github.com/mxagar/computer_vision_udacity), module `03_Advanced_CV_and_DL`.
4. Hyperparameters: Covered in [computer_vision_udacity](https://github.com/mxagar/computer_vision_udacity), module `03_Advanced_CV_and_DL`.
5. Attention Mechanisms: Covered in [computer_vision_udacity](https://github.com/mxagar/computer_vision_udacity), module `03_Advanced_CV_and_DL`.
6. [Embeddings and Word2Vec](#6.-Embeddings-and-Word2Vec)
7. [Sentiment Prediction RNN](#7.-Sentiment-Prediction-RNN)
8. [Project: Generating TV Scripts](#8.-Project:-Generating-TV-Scripts)

## 6. Embeddings and Word2Vec

Word embeddings map a word or a sentence to a vector. We can use neural networks to learn to do word embedding.

Usually, embeddings reduce the dimensionality of the data: a vocabulary of N = 100k words is a dictionary of N elements; we represent all the words with N sparse vectors, each with N elements (all 0 except one 1). If we apply an embedding, we can compress each word to a vector of 200 float elements.

Semantic word embeddings are able to represent words with vectors that capture word meaning relationships as algebraic relationships; therefore, we can perform algebraic operations with words (i.e., sum, multiplication, etc.) to extract vectors of those relationships: verb tense, gender, etc.

![Semantic word embeddings](./pics/semantic_word_embedding.png)

However, note that an embedding doesn't need to be semantic! We can have a simple embedding which compresses sparse vectors or integer-encoded words.

### 6.1 Dimensionality Reduction

Using a word vector passed through an embedding layer is an advantage, because we reduce its dimensionality, avoiding issues related to high dimensions.

A vocabulary of N = 100k words is a dictionary of N elements; we represent all the words with N sparse vectors, each with N elements (all 0 except one 1). Operating with such sparse and large vectors is a waste of resources: we need large but unused memory chunks and we create sparse matrices when applying multiplications to the sparse vectors, which worsens the situation.

If we add an embedding layer, however, the sparse one-hot encoded word vectors are transformed into more compact word vector representations. The embedding layer is a fully connected layer which maps an `N`-dimensional one-hot encoded vector to an `m`-dimensional vector of floats, where `m << N`. The weights of the embedding layer are learnt so that the final application works better.

```
w_embedded = E * w_onehot
w_onehot: 1 x N, [0, 0, 1, 0, 0, 0, 0] (N = 7)
w_embedded: 1 x m, [0.1, 3.5] (m = 2)
E: N x m, embedding layer weight matrix
m: embedding dimension
N: vocabulary size
```

![Embedding look up](./pics/embedding_lookup.png)

In a way, such an embedding layer is like a look-up table: the row of the embedding matrix with the index that equals to the position of the value 1 in the one-hot encoded word is taken. **Therefore: we don't need to one-hot encode the words really, it's enough if we create a dictionary which maps each word with an integer, which will be the look-up index in the embedding matrix. That saves a lot of memory**.


![Integer to Embedding](./pics/tokenize_lookup.png)


### 6.2 Word2Vec: Semantic Embeddings

In addition to dimensionality reduction, we can extend embeddings to capture semantic relationships between words, as presented by Mikolov et al. in their work **Word2Vec**. The concept originates from the notion that word vectors that belong to similar contexts should have similar representations.

When we achieve that, vector arithmetic operations are possible between word vectors to extract or apply semantic information represented as vectors. For instance, a distance vector can be applied to verbs which maps them from a tense to another.

![Vector distance](./pics/vector_distance.png)

In order to obtain such embedding representations, two ways were tried by Mikolov et al.:

- CBOW: Continuous Bags of Words. Given a word `w`, we input its previous and successor words to the model and try to predict `w`. 
- Skip-gram: It is the inverse of CBOW. We input the word `w` and try to predict the context words.

![CBOW and Skip-Gram](./pics/word2vec_cbow_skipgram.png)

Skip-gram works usually better. In practice, we have:

- An input vector of `N` values (all 0 except one 1).
- An output vector of also `N` values (with values between 1 and 0)
- One hidden layer of `m` neurons; `m` is the embedding dimension. There is no activation after the hidden layer, but we apply softmax for the output vector.

That means we have two layer weight matrices:

- First, input to hidden: `N x m`. This is like a lookup table in which each row is the embedding representation of the word we're looking for.
- Second, hidden to output: `m x N`. We are computing the probabilities of a context word given an input word. If we take a row from the first matrix (this happens actually, because the one-hot vector acts like a lookup key) and multiply the second matrix, we obtain the probabilities of every word in the vocabulary to come up in the context.

**The intuition behind it is that similar words or words that belong to the same context are forced to have similar embedding vectors.**

Some interesting additional material:

- [Word2Vec Tutorial](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- `literature/Mikolov_Word2vec_2013.pdf`
- `literature/Mikolov_Word2vec_2_2013.pdf`

### 6.3 Word2Vec Notebook 1: Skip-Gram Model Training

The notebook can be found in

[deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch) `/ word2vec-embeddings / Skip_Grams_Exercise.ipynb`

This notebook implements the first paper by Mikolov et al.: `literature/Mikolov_Word2vec_2013.pdf`.

A Skip-gram model is defined and trained to obtain a semantic embedding.

In order to train, a Wikipedia text from Matt Mahoney is used.

The  notebook has the following sections:

1. Text pre-processing: a vocabulary is built with all the unique words in the text and some symbols are replaced by symbol names (`. ->  <PERIOD>`). Additionally, subsampling of the words is done based on their occurrence: a probability of removing a word is defined based on its frequency in the text.
2. Batch generation: we write a generator of batches which receives the text with integer-encoded words and produces sequences of input-target word pairs (encoded as integers).
3. Similarity function
4. SkipGram Model Definition and Training. Note that the training takes very long. This is optimized in the next notebook.
5. Embedding vector visualization with t-SNE
6. Save the embedding matrix as a dataframe

Note that the training takes quite a long time. INstead off using this approach, we can try the next notebook, which is based in the second paper by Mikolov et al.; that paper improves the training speed.

In the next section, some word arithmetics are shown; these were not in the original Udacity notebook, I tested them.

```python
### -- 0. utils.py

import re
from collections import Counter

def preprocess(text):

    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()
    
    # Remove all words with  5 or fewer occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]

    return trimmed_words


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: Two dictionaries, vocab_to_int, int_to_vocab
    """
    word_counts = Counter(words)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

### -- 1. Text pre-processing

# read in the extracted text file      
with open('data/text8') as f:
    text = f.read()

# print out the first 100 characters
print(text[:100])

import utils

# get list of words
words = utils.preprocess(text)
print(words[:30])

# print some stats about this word data
print("Total words in text: {}".format(len(words))) 
print("Unique words: {}".format(len(set(words)))) # `set` removes any duplicate words
# Total words in text: 16680599
# Unique words: 63641

vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]

print(int_words[:30])
# [5233, 3080, 11, 5, 194, 1, 3133, 45, 58, 155, 127, 741, 476, 10571, 133, 0, 27349, 1, 0, 102, 854, 2, 0, 15067, 58112, 1, 0, 150, 854, 3580]

from collections import Counter
import random
import numpy as np

threshold = 1e-5
word_counts = Counter(int_words)
print(list(word_counts.items())[0])  # dictionary of int_words, how many times they appear

# discard some frequent words, according to the subsampling equation
# create a new list of words for training
text_size = len(int_words)
def subsampling_probability(i):
    f = word_counts[i] / text_size
    return 1 - np.sqrt(threshold / f)

# Note: we do not remove a word above a threshold
# but we compute its removal probability with subsampling_probability
# and then remove it with that probability.
# That is achieved by generating a random value for each word
# and checking whether it is below the subsampling_probability;
# if so, we remove the integer, else we take it.
# Words that appear few times have a lower removal probablity.
# The effect is that we remove the bias of words that appear frequently.
train_words = [i for i in int_words if random.random() < 1 - subsampling_probability(i)]
# Equivalent:
# train_words = [i for i in int_words if subsampling_probability(i) < random.random()]

print("Number of words in text: ",text_size)
print("Reduced number of words in text",len(train_words))
print(int_words[:30]) # Many low integer values
print(train_words[:30]) # We see that most low integer values are gone
# Number of words in text:  16680599
# Reduced number of words in text 4628122
# [5233, 3080, 11, 5, 194, 1, 3133, 45, 58, 155, 127, 741, 476, 10571, 133, 0, 27349, 1, 0, 102, 854, 2, 0, 15067, 58112, 1, 0, 150, 854, 3580]
# [3080, 3133, 741, 10571, 27349, 15067, 58112, 854, 10712, 1324, 19, 362, 3672, 36, 1423, 7088, 247, 44611, 2877, 5233, 10, 8983, 279, 4147, 6437, 4186, 447, 4860, 6753, 7573]

### -- 2. Batch generation

# Given a window size C, a random range R = random([1,C]) is taken.
# Then, given an index idx, R words before and after it are taken.
# Example:
# [5233, 58, 741, 10571, 27349, 0, 15067, ... ] -> R=2, idx=2 -> [5233, 58, 10571, 27349]
def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''
    R = random.randrange(1, window_size+1)
    left = []
    if idx < R:
        left = words[:idx]
    else:
        left = words[(idx-R):idx]
    right = []
    if idx+R > len(words)-1:
        right = words[:-1]
    else:
        right = words[(idx+1):(idx+1+R)]

    return left+right

# run this cell multiple times to check for random window selection
int_text = [i for i in range(10)]
print('Input: ', int_text)
idx=5 # word index of interest

target = get_target(int_text, idx=idx, window_size=5)
print('Target: ', target)  # you should get some indices around the idx
# Input:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Target:  [1, 2, 3, 4, 6, 7, 8, 9]

# Batch generator
# We pass a batch size: this will be the number of words for which we take R context words
# before and after them. Therefore, for each word in the batch we have a maximum of 2R targets.
# Thus, for each batch we get a maximum of batch_size*2*R input-target pairs.
def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''
    
    n_batches = len(words)//batch_size
    
    # only full batches
    words = words[:n_batches*batch_size]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x]*len(batch_y))
        yield x, y

int_text = [i for i in range(20)]
x,y = next(get_batches(int_text, batch_size=4, window_size=5))

print('x\n', x) # [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]
print('y\n', y) # [1, 0, 0, 1, 2, 0, 1, 0, 1, 2, 2, 0, 1, 2]

### -- 3. Similarity function

# The cosine similarity will be used to find the most similar words in the embedding matrix.
# This is a human-understandable indicator of how good the clustering of similar words is working.
def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    """ Returns the cosine similarity of validation words with words in the embedding matrix.
        Here, embedding should be a PyTorch embedding module.
    """
    
    # Here we're calculating the cosine similarity between some random words and 
    # our embedding vectors. With the similarities, we can look at what words are
    # close to our random words.
    
    # sim = (a . b) / |a||b|
    
    embed_vectors = embedding.weight
    
    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    
    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent 
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000,1000+valid_window), valid_size//2))
    valid_examples = torch.LongTensor(valid_examples).to(device)
    
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes
        
    return valid_examples, similarities

### -- 4. SkipGram Model Definition and Training

import torch
from torch import nn
import torch.optim as optim

class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        
        # Embedding matrix: one row for each word in vocab x embedding dimension
        self.embed = nn.Embedding(n_vocab, n_embed)
        # Map back from embedding dimension to number of words in vocab
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.embed(x)
        scores = self.output(x)
        log_ps = self.log_softmax(scores)
        
        return log_ps

embedding_dim = 300
model = SkipGram(len(vocab_to_int), embedding_dim)

inputs, targets = next(get_batches(train_words, 2))
inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
log_ps = model(inputs)
print("inputs: ", inputs)
print("targets: ", targets)
print("log_ps: ", log_ps)
print(log_ps.shape) # torch.Size([3, 63641])
print(targets.shape) # torch.Size([3])

filepath = 'checkpoint_last.pth'
def save_model(filepath, model):
    torch.save(model.state_dict(), filepath)
    
def load_checkpoint(filepath, n_vocab, n_embed):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    model = SkipGram(n_vocab, n_embed).to(device)
    model.load_state_dict(checkpoint)

    return model

#model = load_checkpoint(filepath, n_vocab, n_embed)

# check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

embedding_dim=300 # you can change, if you want

model = SkipGram(len(vocab_to_int), embedding_dim).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

print_every = 500 #5
steps = 0
epochs = 5

# Model saving parameters
min_loss = np.Inf
filepath = "checkpoint_last.pth"
# Test saving & loading
save_model(filepath, model)
model = load_checkpoint(filepath, len(vocab_to_int), embedding_dim)
print(model)
# SkipGram(
#  (embed): Embedding(63641, 300)
#  (output): Linear(in_features=300, out_features=63641, bias=True)
#  (log_softmax): LogSoftmax(dim=1)
#)

# train for some number of epochs
for e in range(epochs):
    
    # get input and target batches
    for inputs, targets in get_batches(train_words, 512):
    #for inputs, targets in get_batches(train_words, 2):
        steps += 1
        inputs, targets = torch.LongTensor(inputs), torch.LongTensor(targets)
        inputs, targets = inputs.to(device), targets.to(device)
        
        log_ps = model(inputs)
        loss = criterion(log_ps, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if steps % print_every == 0:                  
            # getting examples and similarities      
            valid_examples, valid_similarities = cosine_similarity(model.embed, device=device)
            _, closest_idxs = valid_similarities.topk(6) # topk highest similarities
            
            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            for ii, valid_idx in enumerate(valid_examples):
                closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
            print("...")
            
            # save checkpoint if loss smaller than ever
            if loss.item() < min_loss:
                min_loss = loss.item()
                save_model(filepath, model)

### -- 5. Embedding vector visualization with t-SNE

filepath = "checkpoint_last.pth"
model = load_checkpoint(filepath, len(vocab_to_int), embedding_dim)
print(model)

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# getting embeddings from the embedding layer of our model, by name
embeddings = model.embed.weight.to('cpu').data.numpy()

# We visualize the first viz_words from the embedding matrix in 2D
# i.e., vectors are projected on to a 2D plane: 300 -> 2
viz_words = 600
tsne = TSNE() # n_components = 2, by default
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

# Similar words should be clustered together
fig, ax = plt.subplots(figsize=(16, 16))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

### -- 6. Save the embedding matrix as a dataframe

import pandas as pd

embeddings = model.in_embed.weight.to('cpu').data.numpy()

df = pd.DataFrame(embeddings)
words = [int_to_vocab[i] for i in range(len(vocab_to_int))]
df["words"] = words
df.to_csv('data/embedding.csv',sep=',', header=True, index=False) # The dataframe is around 208 MB
df = pd.read_csv('data/embedding.csv')

```
#### Question in the Forum

Hi,

In the Embeddings exercise with the Skip-Gram the model is trained with batches of input & target words encoded as integers. However, the model outputs one-hot encoded vectors. As explained in the videos and the notebook, the size of the sparse one-hot vector is the maximum number of the word integer (+1); that way, the word integer is the index in the one-hot encoded vector.

That is perfectly understandable for the programmer, and I can understand that the embedding layer uses the integer-encoded word as lookup index automatically. However, the loss function, which should be unaware of any encoding strategies, compares one-hot vectors with integer-encoded words:

	loss = criterion(log_ps, targets)

If one checks the sizes both tensors, they are different, as expected:

	log_ps: [n, m] (log probabilities)
	targets: [n] (integers)
	n: input word sequence number in batch
	m: vocabulary size

That is irritating, because the loss function seems to be inferring that in order to compare both vectors one is used as the index to look in the other. Where is described that behaviour? I would say that the loss function shouldn't make any assumptions, it should just compute the difference/similarity value between tensors of the same shape.

Or am I missing something?

Thank you,

Mikel

### 6.4 Word2Vec Notebook 2: Skip-Gram Model Training with Negative Sampling

The notebook can be found in

[deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch) `/ word2vec-embeddings / Negative_Sampling_Exercise.ipynb`

This notebook implements the second paper by Mikolov et al.: `literature/Mikolov_Word2vec_2_2013.pdf`

The motivation behind is that the training of the Skip-gram model as defined above takes a very long time. Using sparse vectors from the output is indeed very inefficient. The solution proposed by Mikolov et al. consists in using **negative sampling**. To that end, the loss of the word we are trying to compress and the loss of some other hundred words are taken into account in each pass, not the loss of all the words in the vocabulary. This significantly speeds up the computation.

The concept is simple, but it requires to implement an apparently special model with several forward functions and a custom loss function class.

I haven't gone through the code thoroughly, but it seems a bit more complicated.

The final output is the same: the embedding matrix; this time the training is much faster and the t-SNE visualization looks nice.

The take-away is that I can use this notebook to generate my own semantic embeddings, which can be used later on! In particular, I have saved the embedding as a CSV after its visualization. I also tried some word arithmetics with the embedding matrix; these were not in the original Udacity notebook.

```python
### -- Saving the embedding matrix

import pandas as pd

embeddings = model.in_embed.weight.to('cpu').data.numpy()

df = pd.DataFrame(embeddings)
words = [int_to_vocab[i] for i in range(len(vocab_to_int))]
df["words"] = words
df.to_csv('data/embedding.csv',sep=',', header=True, index=False) # The dataframe is around 208 MB
df = pd.read_csv('data/embedding.csv')

### -- Tests with word arithmetics

# In this section I try to do some word arithmetics to check how well the semantics was captured by the embedding.
# Conclusion: it's not that good, but words are not completely unrelated. Maybe more training would help.

df = df.set_index("words")

def cosine(v1, v2):
    """Returns cosine of two vectors = a.b / |a|*|b|."""
    v1_length = np.sqrt(np.sum(v1**2,axis=0))
    v2_length = np.sqrt(np.sum(v2**2,axis=0))
    c = np.dot(v1,v2) / (v1_length*v2_length)
    
    return c

def compute_cosines(df, v):
    """Computes cosines of all vocabulary vectors against one input vector v."""
    E = np.array(df.values)
    v = np.array(v)
    E_lengths = np.sqrt(np.sum(E**2,axis=1))
    v_length = np.sqrt(np.sum(v**2,axis=0))
    cosines = np.dot(E,v) / (E_lengths*v_length)
    return cosines

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

v_queen = df.loc["king",:] - df.loc["man",:] + df.loc["woman",:]
cosine(np.array(df.loc["queen",:]),np.array(v_queen)) # 0.3289111913984208

cos = compute_cosines(df,v_queen)
[int_to_vocab[i] for i in list(largest_indices(cos, 5)[0])] 
# ['woman', 'king', 'her', 'elizabeth', 'born']

v_man = df.loc["boy",:] + df.loc["adult",:]
cos = compute_cosines(df,v_man)
[int_to_vocab[i] for i in list(largest_indices(cos, 5)[0])]
# ['adult', 'boy', 'children', 'girls', 'super']

v_girl = df.loc["boy",:] - df.loc["man",:]
cos = compute_cosines(df,v_girl)
[int_to_vocab[i] for i in list(largest_indices(cos, 5)[0])]
# ['boy', 'nintendo', 'gba', 'nes', 'spock']

```

## 7. Sentiment Prediction RNN

In this section, the sentiment analysis network presented by [Andrew Trask](http://iamtrask.github.io) in the 1st module is improved.

The section is implemented in a notebook: [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch) `/ sentiment-rnn / Sentiment_RNN_Exercise.ipynb`

Since we pass sequences of words to the RNN model based on LSTM cells, the performance is expected to be better than the fully connected network by Andrew Trask.


## 8. Project: Generating TV Scripts



