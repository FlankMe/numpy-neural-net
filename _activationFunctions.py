"""
Activation functions
"""
import numpy as np


# List of activation functions implemented
def assignActivationFunction(func):
  AF =   {'relu' : relu,
          'lrelu' : leakyRelu,
          'sigmoid' : sigmoid,
          'tanh' : tanh,
          'identity' : identity,
          'softmax' : softmax,
    }
  return (AF[func])


# Implementations
def relu(z, BACK=False):
  if not BACK:
    return (np.maximum(0, z))
  else:
    return ((z>0))
  
def leakyRelu(z, BACK=False, LEAKYCOEFF=0.05):
  if not BACK:
    return (np.maximum(LEAKYCOEFF * z, z))
  else:
    return (LEAKYCOEFF + (z>0) * (1 - LEAKYCOEFF))

def sigmoid(z, BACK=False):
  if not BACK:  
    return (1.0 / (1 + np.exp(-z)))
  else:
    return (z * (1 - z))
  
def tanh(z, BACK=False):
  if not BACK:  
    return (2 * sigmoid(2 * z) - 1)
  else:
    return (1 - z ** 2)
    
def identity(z, BACK=False):
  if not BACK:  
    return (z)
  else:
    return (1)
    
def softmax(z, BACK=False):
  if not BACK:  
    z -= z.max(axis=1, keepdims=True)
    totals = np.exp(z).sum(axis=1, keepdims=True)
    return (np.exp(z) / totals)
  else:
    return (z * (1 - z))