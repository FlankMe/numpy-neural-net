"""
Master file for the neural network

At initialisation, it requires [batchShape, costFunction, graph], where:

* batchShape is the shape of the minibatch fed during training (note that the 
size of the batch will be ignored);
* costFunction can be chosen among the options in file _costFunctions.py;
* graph is a list of dictionaries and represents the arbitrary archiecture 
of the network. It needs to be consistent with the below example:
graph = [
  {'layerType':'conv',        'numNeurons':8,       'config':[4, 4, 0, 2]},
  {'layerType':'activation',  'activationFunction':'lrelu'},
  {'layerType':'dropout',     'threshold':0.8},

  {'layerType':'fc',          'numNeurons':64},
  {'layerType':'activation',  'activationFunction':'lrelu'},
  {'layerType':'dropout',     'threshold':0.5},

  {'layerType':'fc',          'numNeurons':10},
  {'layerType':'activation',  'activationFunction':'softmax'},
  ]

https://github.com/flankme
@author: Riccardo Rossi
"""

# Import libraries and files
import numpy as np
import time; np.random.seed(int(time.time()))

import _supportingFunctions as sfs  # such as im2col, one-hot-key, etc
import _activationFunctions as afs  # such as relu, tanh, lrely, etc
import _costFunctions as cfs        # such as mean squared errors, cross-entropy
import _optimiser as optim          # implementation of optimisation algos
import _implementedLayers as il     # such as fully-connected, convolutional, etc


class NumpyNet:

  def __init__(self, batchShape, costFunction, graph):

    # The number of examples of the minibatch is irrelevant for the architecture
    batchShape = batchShape[1:]
   
    # Build the network's graph   
    self.graph = []
    for layer in graph:
      layerType = layer['layerType']
      assignedLayerType = il.assignLayerType(layerType)

      layer['batchShape'] = batchShape
      newLayer = assignedLayerType(layer)
      self.graph.append(newLayer)
      batchShape = newLayer.nextBatchShape
      
    self.cf = cfs.assignCostFunction(costFunction)
    
    # Initialise to False a few flags
    self.AdamParameters = False
    
  
  def predict(self, X, TRAINING=False):
    
    # If activation is a vector, it will now have shape (1 x m) 
    activation = np.atleast_2d(X)  

    # Main feedforward loop
    for layer in self.graph:
      activation = layer.fwd(activation, TRAINING=TRAINING)

    if TRAINING:
      self.prediction = [activation]

    return (activation)
  
     
  def fit(self, X, Y, 
          method='SGD', epochs=10, minibatchSize=64,
          LEARNINGRATE=3e-4, L2PENALTY=1e-4, DROPOUT=False,
          VERBOSE=True):

    assert len(X) == len(Y), "Lengths of X and y are different"
    
    # If Adam optimiser is used, then initialise all layers
    if method == 'Adam':
      if not self.AdamParameters:
        optim.AdamInitialise(self)
    
    # Main optimisation loop
    for epoch in range(epochs):
      idx = np.random.permutation(len(Y))
      
      # Loop per minibatch
      for i in range(0, int(len(Y) / minibatchSize)):
        miniX = X[idx[i*minibatchSize:(i+1)*minibatchSize]]
        miniY = Y[idx[i*minibatchSize:(i+1)*minibatchSize]]
        
        # Forward pass and measure the loss
        predictedY = self.predict(miniX, TRAINING=True)
        loss, perturbation = self.cf(miniY, predictedY)
        if L2PENALTY:
          loss += cfs.addL2penalty(self.graph, L2PENALTY)
        
        # Backward pass
        optim.optimisationStep(self, perturbation, method, 
                               LEARNINGRATE, L2PENALTY, DROPOUT)
        
      if VERBOSE:
        print ("Epoch:", epoch + 1, "; Loss for minibatch:", loss)


  def score(self, X, y):
    
    assert self.cf.__name__ == 'xentropy', (
      "Score method is only implemented for the cross-entropy cost function")
        
    yHat = self.predict(X)
    correctAnswers = (yHat.argmax(axis=1) == y.argmax(axis=1)).sum() 
    score = float(correctAnswers) / len(y)
    
    return score
