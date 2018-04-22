import numpy as np
import _supportingFunctions as sfs

"""
Main optimisation step
"""
def optimisationStep(net, perturbation, method='SGD', 
                     LEARNINGRATE=3e-4, L2PENALTY=1e-4, DROPOUT=False):   

  activation = net.prediction.pop()
  
  for layer in reversed(net.graph):

    perturbation, activation, weightsGradient, biasGradient = layer.back(perturbation, activation)

    if weightsGradient is None or biasGradient is None:
      continue
    
    effectiveLearningRate = LEARNINGRATE    
    if method == 'Adam':
      # https://arxiv.org/pdf/1412.6980.pdf
      weightsGradient, biasGradient, LearningRateAdj = (
        AdamUpdateGradient(net, layer, weightsGradient, biasGradient)      )
      effectiveLearningRate *= LearningRateAdj
    
    weightsGradient = sfs.clipValues(weightsGradient) 
    biasGradient = sfs.clipValues(biasGradient)    

    layer.Weights -= effectiveLearningRate * weightsGradient
    layer.bias -= effectiveLearningRate * biasGradient

    layer.Weights *= 1 - L2PENALTY
    layer.bias *= 1 - L2PENALTY



""" 
Adam optimiser
Algorithm from paper: https://arxiv.org/pdf/1412.6980.pdf
"""
def AdamUpdateGradient(net, layer, weightsGradient, biasGradient):

  BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-8
  currentBeta1, currentBeta2 = net.AdamParameters
  
  weightsMomentum, biasMomentum = layer.AdamMomentum
  weightsVariance, biasVariance = layer.AdamVariance

  weightsMomentum = BETA1 * weightsMomentum + (1 - BETA1) * weightsGradient
  biasMomentum = BETA1 * biasMomentum + (1 - BETA1) * biasGradient

  weightsVariance = BETA2 * weightsVariance + (1 - BETA2) * (weightsGradient ** 2)
  biasVariance = BETA2 * biasVariance + (1 - BETA2) * (biasGradient ** 2)

  weightsGradient = weightsMomentum / (np.sqrt(weightsVariance) + EPSILON)
  biasGradient = biasMomentum / (np.sqrt(biasVariance) + EPSILON)

  layer.AdamMomentum = weightsMomentum, biasMomentum 
  layer.AdamVariance = weightsVariance, biasVariance 
  
  LearningRateAdj = np.sqrt(1 - currentBeta2) / (1 - currentBeta1)

  if layer == net.graph[0]:
    net.AdamParameters = [currentBeta1 * BETA1, currentBeta2 * BETA2]
  
  return weightsGradient, biasGradient, LearningRateAdj


def AdamInitialise(net):
  
  # Initialise variables possibly used by Adam optimiser  
  currentBETA1, currentBETA2 = 0.9, 0.999
  for layer in net.graph:
    try:
      layer.AdamMomentum = [np.zeros(layer.Weights.shape), np.zeros(layer.bias.shape)]
      layer.AdamVariance = [np.zeros(layer.Weights.shape), np.zeros(layer.bias.shape)]
    except:
      continue
  net.AdamParameters = [currentBETA1, currentBETA2]
  return (net)
    
