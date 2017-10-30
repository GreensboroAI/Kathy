# These things we need for NLP
import nltk
#nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

#Things we need for tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

#These things we need for JSON
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?']

# Loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #Tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        #Add to our words list
        words.extend(w)
        #Add to documents in our corpus
        documents.append((w, intent['tag']))
        #Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


#stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

#remove duplicates
classes = sorted(list(set(classes)))

print(len(documents), 'documents')
print(len(classes), 'classes', classes)
print(len(words), 'unique stemmed words', words)


#Turn our data structure of words into tensors of numbers for Tensorflow
#Create our training data
training = []
output = []
#Create an empty array for our output
output_empty = [0] * len(classes)

#Training set bag of words for each sentence
for doc in documents:
    #initialize our bag of words
    bag = []
    #list of tokenized words for pattern
    pattern_words = doc[0]
    #stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    #create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    #output is 0 for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

#Shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

#Create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

print(train_x[1])
print(train_y[1])

#Reset underlying graph data just in case
tf.reset_default_graph()
#Build the neural network we are going to use for our chatbot
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

#Define the model and setup the tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
#Start training with gradient descent
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

import pickle
pickle.dump({'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open("training_data", "wb"))
