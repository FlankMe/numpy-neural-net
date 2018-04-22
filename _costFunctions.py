"""
Cost functions
"""
import numpy as np
import _supportingFunctions as sfs


# List of cost functions implemented
def assignCostFunction(func):
  CF = {'mse' :       mse,
        'xentropy' :  xentropy,
    }
  return CF[func]


# Mean Squared Erros
def mse(Y, predictedY):

  # perturbation is the derivative of the cost function wrt the prediction
  Y = np.atleast_2d(Y).T if len(Y.shape) == 1 else np.atleast_2d(Y)
  perturbation = predictedY - Y   # shape is (batch x outputs)

  # Calculate the total loss
  loss = 0.5 * (perturbation ** 2).sum() / len(Y)

  return loss, perturbation  
  

# Cross-entropy
def xentropy(Y, predictedY):
  
  # Prevent division by 0 or calculation of log(0)
  EPSILON = 1e-10                     
  
  # Y is assumed to be in one-hot-key form already 
  assert len(Y.shape) != 1 or len(np.unique(Y)) == 1, (
    "Y should be transformed to one-hot-key form with sfs.oneHotKey() func")
    
  # perturbation is the derivative of the cost function wrt the prediction
  perturbation = (predictedY - Y) / (
                                      predictedY * (1-predictedY) + EPSILON
                                    )   

  # Calculate the total loss
  loss = - (   Y      * np.log(predictedY +     EPSILON) +
              (1 - Y) * np.log(1 - predictedY + EPSILON) 
            ).sum() / len(Y)

  return loss, perturbation      


def addL2penalty(graph, L2PENALTY=1e-6):

  # Add the L2 penalty to the cost function  
  assert graph is not None, "The graph was not passed to the addL2penalty"

  loss = 0
  for layer in graph:
    try:
      loss += 0.5 * L2PENALTY * (layer.Weights ** 2).sum()
      loss += 0.5 * L2PENALTY * (layer.bias ** 2).sum()
    except:
      continue
  return loss