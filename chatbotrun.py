#Lets restore all our data structures
import pickle
data = pickle.load(open('training_data', 'rb'))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

#import our chatbot intents file again
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

#load our saved model
# reset underlying graph data
import tensorflow as tf
import tflearn
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

model.load('./model.tflearn')

import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
stemmer = LancasterStemmer()
#Create a bag of words from user input
def clean_up_sentence(sentence):
    #tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    #stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

#Return bag of words array: 0 or 1 for each word in the bag of words that exists in the sentence
def bow(sentence, words, show_details=False):
    #Tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    #bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print('found in bag: %s' % w)

    return(np.array(bag))

ERROR_THRESHOLD = 0.25
def classify(sentence):
    #Generate probability from the model
    results = model.predict([bow(sentence, words)])[0]
    #filter our predictions below our threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    #Sort by the strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    #Return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    #If we have a classification then find the matching intent tag
    if results:
        #loop as long as their are matches to process
        while results:
            for i in intents['intents']:
                #find a tag matching the first result
                if i['tag'] == results[0][0]:
                    #a random response from the intent
                    return print(random.choice(i['responses']))

            results.pop(0)


print(response('is your shop open today?'))
print(response('What do you sell?'))
print(response('Do you accept visa?'))
print(response('Hi?', show_details=True))