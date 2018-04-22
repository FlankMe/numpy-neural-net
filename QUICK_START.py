"""
Testing script for NumpyNet
"""

import numpy as np
import time
np.random.seed(int(time.time()))

import _supportingFunctions as sfs
import _activationFunctions as afs
import _costFunctions as cfs
import _optimiser as optim
import _implementedLayers as il

import NumpyNet
import pandas as pd


# Load data. You can change the condition to False when running it the 2nd time
if True:
  data = pd.DataFrame.from_csv('MNISTtrain.csv')
  portion = 0.80
  trainingSet = int(portion * len(data))
  idx = np.random.permutation(len(data))
  idxTraining = idx[:trainingSet]
  idxTest = idx[trainingSet:2*trainingSet]
  
  X = data.values[idxTraining] 
  X = X.reshape(-1, 1, 28, 28)
  X = 1.0 / 255 * X
  y = data.index[idxTraining]
  y = sfs.oneHotKey(y, 10)  
  
  Xtest = data.values[idxTest] 
  Xtest = Xtest.reshape(-1, 1, 28, 28)
  Xtest = 1.0 / 255 * Xtest 
  ytest = data.index[idxTest]
  ytest = sfs.oneHotKey(ytest, 10)  
  
  del data  
  

# Start measuring the time 
start = time.time()
  
# Define the graph
outputsize = 10
graph = [
  {'layerType':'conv',        'numNeurons':8,       'config':[4, 4, 0, 2]},
  {'layerType':'activation',  'activationFunction':'lrelu'},
  {'layerType':'dropout',     'threshold':0.8},

  {'layerType':'conv',        'numNeurons':4,       'config':[1, 1, 0, 1]},
  {'layerType':'activation',  'activationFunction':'lrelu'},
  {'layerType':'dropout',     'threshold':0.8},

  {'layerType':'fc',          'numNeurons':64},
  {'layerType':'activation',  'activationFunction':'lrelu'},
  {'layerType':'dropout',     'threshold':0.5},

  {'layerType':'fc',          'numNeurons':outputsize},
  {'layerType':'activation',  'activationFunction':'softmax'},
  ]

# Create the neural net and launch the training
costFunction = 'xentropy'
nn = NumpyNet.NumpyNet(X.shape, costFunction, graph)
nn.fit(X, y, method='Adam', epochs=20, LEARNINGRATE=3e-4, L2PENALTY=1e-4, VERBOSE=True)

# Measure the score
print "\nScore on training set", nn.score(X, y)
print "Score on test set", nn.score(Xtest, ytest)

# End of the programme
print "It took", int(time.time() - start), "seconds"
