import pickle
import keras
import numpy as np

class ClassicalModel:
  def __init__(self, model):
    self.model = model

  def inference(self, x):
    return (self.model).predict([x])[0]

  def interview(self, X):
    return (self.model).predict(X)

  def predict_proba(self, X):
    return self.model.predict_proba(X)


class LSTMModel:
  def __init__(self, model, vector_size, token_size):
    self.model, self.vector_size, self.token_size = model, vector_size, token_size

  def inference(self, x):
    x = np.reshape([x], (1, self.vector_size, self.token_size))
    return np.argmax(self.model.predict(x, verbose=0), axis=-1)[0]

  def interview(self, X):
    if not isinstance(X, np.ndarray):
      X = np.array(X)
    X = np.reshape(X, (X.shape[0], self.vector_size, self.token_size))
    return np.argmax(self.model.predict(X, verbose=0), axis=-1)

  def predict_proba(self, X):
    if not isinstance(X, np.ndarray):
      X = np.array(X)
    X = np.reshape(X, (X.shape[0], self.vector_size, self.token_size))
    return self.model.predict(X, verbose=0)


class PerceptronModel:
  def __init__(self, model):
    self.model = model

  def inference(self, x):
    return np.argmax(self.model.predict(np.array([x]), verbose=0), axis=-1)[0]

  def interview(self, X):
    if not isinstance(X, np.ndarray):
      X = np.array(X)
    return np.argmax(self.model.predict(X, verbose=0), axis=-1)

  def predict_proba(self, X):
    if not isinstance(X, np.ndarray):
      X = np.array(X)
    return self.model.predict(X, verbose=0)


# Each model "inference" method must return a non negative integer.
# The first model is the base model.
class BagOfModels:
  def __init__(self, models):
    self.models = models
    self.n_label = 100

  def inference(self, x):
    return (self.interview(np.array([x])))[0]

  def interview(self, X):
    predictions = []
    for i in range(len(self.models)):
      predictions.append(self.models[i].interview(X))
    labels = []
    for i in range(len(X)):
      cnt = [0]*self.n_label
      for pred in predictions:
        cnt[pred[i]] += 1
      best_label = np.argmax(cnt)
      if cnt[predictions[0][i]] == cnt[best_label]:
        labels.append(predictions[0][i])
      else:
        labels.append(best_label)
    return labels