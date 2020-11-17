import json
import numpy as np
import pandas as pd
import pickle
import random

import nltk
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

'''
#Commented out to speed up the UI

words = []
classes = []
documents = []
ignore_words = ['?']
intents=pd.read_json('intents.json')
# loop through each sentence in the intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to words list
        words.extend(w)
        # add to documents in corpus
        documents.append((w, intent['tag']))
        # add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
# lemmatize and lowercase each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))


# creating the training data
training = []
# create an empty array for the output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize the bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create the bag of words array with 1 if word match found in current pattern, else 0
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
# shuffle features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])

# 3 Layers NN Model. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Stochastic gradient descent with Nesterov accelerated gradient
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=0)


#save model
model.save("model.h5")
#save words and classes
PIK = 'model_files.dat'
with open(PIK, 'wb') as f:
	pickle.dump(words, f)
	pickle.dump(classes, f)
	pickle.dump(intents, f)
'''
#load
from tensorflow.keras.models import load_model
model = load_model('model.h5', compile=False)

with open('model_files.dat', 'rb') as f:
	words = pickle.load(f)
	classes = pickle.load(f) 
	intents = pickle.load(f) 

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # lemmatize each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    return(np.array(bag))

def chat(sentence):
    ERROR_THRESHOLD = 0.25
    
    # load input data and generate probabilities
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    #context index of highest probability
    results_index = np.argmax(results)
    tag = classes[results_index]
    #loop through the sentences
    for tg in intents['intents']:
    	#get the tag with highest probability
    	if tg['tag'] == tag:
    		#get it's responses
    		responses = tg['responses']
    #return one of them at random
    return random.choice(responses)