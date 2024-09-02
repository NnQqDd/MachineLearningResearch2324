#import os
import sys
import numpy as np
import sklearn
import pickle
import keras
import tensorflow
import tensorflow.keras.backend as K
# import keras.backend as K
from keras.layers import LSTM, Dense, Bidirectional, Dropout, Lambda, Input, Flatten, Activation, RepeatVector, Permute, Concatenate
from keras.models import Sequential, Model
from model_wrapper import PerceptronModel, ClassicalModel, LSTMModel, BagOfModels
from keras.models import load_model

def load_perceptron_model(h5):  # Deals with compatibility issues 
  X, Y = [], [0]
  for i in range(768):
    X.append(1)
  X = np.array([X])
  Y = np.eye(11)[Y]
  model = Sequential()
  model.add(Dense(units=128, activation='relu'))
  model.add(Dropout(0.6))
  model.add(Dense(units=11, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.fit(X, Y, batch_size=1, epochs=1, verbose=0)
  #model.summary()
  model.load_weights(h5)
  return model

def load_lstm_model(h5): # Deals with compatibility issues 
  input_seq = Input(shape=(8, 96))
  activations = Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.2))(input_seq)
  attention = Dense(1, activation='tanh')(activations)
  attention = Flatten()(attention)
  attention = Activation('softmax')(attention)
  attention = RepeatVector(64)(attention)
  attention = Permute([2, 1])(attention)
  sent_representation = Concatenate()([activations, attention])
  sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(None, 192))(sent_representation)
  sent_representation = Dropout(0.4)(sent_representation)
  probabilities = Dense(11, activation='softmax')(sent_representation)
  model = Model(inputs=input_seq, outputs=probabilities)
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.load_weights(h5)
  return model

def predict(x, selection): # x is a numerical list of length 768
  if(selection == 'sfm'):
    return w_lgreg.inference(x)
  if(selection == 'svm'):
    return w_svm.inference(x)
  if(selection == 'mlp'):
    return w_perceptron.inference(x)
  if(selection == 'lstm'):
    return w_lstm.inference(x)
  return bag.inference(x)

LGREG_MODEL_PATH = 'models/utc_lgreg.pkl'
SVM_MODEL_PATH = 'models/utc_svm.pkl'
PERCEPTRON_WEIGHT_PATH = 'models/utc_perceptron_4.h5' 
LSTM_WEIGHT_PATH = 'models/utc_lstm_3.h5' 

w_lgreg = None
w_svc = None

with open(LGREG_MODEL_PATH, 'rb') as file:
  model = pickle.load(file)
  w_lgreg = ClassicalModel(model)

with open(SVM_MODEL_PATH, 'rb') as file:
  model = pickle.load(file)
  w_svc = ClassicalModel(model)

w_perceptron = PerceptronModel(load_perceptron_model(PERCEPTRON_WEIGHT_PATH)) 
w_lstm = LSTMModel(load_lstm_model(LSTM_WEIGHT_PATH), 8, 96)
bag = BagOfModels([w_lstm, w_svc, w_perceptron])
bag.n_label = 11

print('Done loading models.')
# .inference