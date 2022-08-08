"""
Layers of the neural network
"""
import numpy as np
import _supportingFunctions as sfs
import _activationFunctions as afs


# Types of layers implemented
def assignLayerType(layerType):
  LAYERS = {
            'fc' :    FullyConnected, 
            'conv' :  Convolutional, 
            'activation' : Activation,
            'dropout' : Dropout,
            }
  return (LAYERS[layerType])
  
  
  
""" FULLY CONNECTED LAYER """
class FullyConnected():

  def __init__(self, parameters):
    numNeurons = parameters['numNeurons']
    batchShape = parameters['batchShape']
    
    if not np.isscalar(batchShape):
      batchShape = int(np.prod(batchShape))
    self.batchShape = batchShape
    self.nextBatchShape = numNeurons
    
    self.Weights = np.random.randn(batchShape, numNeurons)
    self.Weights *= np.sqrt(2.0 / batchShape)
    self.bias = np.zeros(numNeurons)
    
  def fwd(self, inputData, TRAINING=False):
    if not len(inputData.shape) == 2:
      inputData = inputData.reshape(inputData.shape[0], -1)
    if TRAINING:
      self.cachedInput = [inputData]
      
    dotProduct = np.dot(inputData, self.Weights) + self.bias
    return (dotProduct)
    
  def back(self, perturbation, activation):
    inputData = self.cachedInput.pop()
    biasGradient = np.sum(perturbation, axis=0)
    weightsGradient = np.dot(inputData.T, 
                             perturbation)
                             
    perturbation = np.dot(perturbation,
                          self.Weights.T)
    return (perturbation, inputData, weightsGradient, biasGradient)


""" CONVOLUTIONAL LAYER """
class Convolutional():

  def __init__(self, parameters):
    assert parameters['config'], "Config required when initialising a conv layer"
    assert not np.isscalar(parameters['batchShape']), "Input needs to be multi-dimensional"
    
    Hfield, Wfield, padding, stride = parameters['config']
    if padding < 0:
      padding = (stride - 1.0) / 2
      assert padding == int(padding), "Padding size is inconsistent"
    self.config = [padding, stride]

    self.batchShape = parameters['batchShape']
    numNeurons = parameters['numNeurons']
    CHANNELS, HEIGHT, WIDTH = self.batchShape
    newH = (HEIGHT + 2 * padding - Hfield) / stride + 1
    newW = (WIDTH + 2 * padding - Wfield) / stride + 1
    self.nextBatchShape = (numNeurons, newH, newW)
    
    self.Weights = np.random.randn(numNeurons, CHANNELS, Hfield, Wfield)
    self.Weights *= np.sqrt(2.0 / CHANNELS * Hfield * Wfield)
    self.bias = np.zeros(shape=(numNeurons, 1))

    
  def fwd(self, inputData, TRAINING=False):
  # Re-adapted: https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/  
    padding, stride  = self.config
    nFilters, dFilter, hFilter, wFilter = self.Weights.shape
    minibatchSize = inputData.shape[0]
    Cnext, Hnext, Wnext = self.nextBatchShape

    InputData_in_column = sfs.im2col(inputData, hFilter, wFilter, 
                                     padding=padding, stride=stride)
    Weights_in_column = self.Weights.reshape(nFilters, -1)
    
    if TRAINING:
      self.cachedInput = [[inputData, InputData_in_column]]

    dotProduct = np.dot(Weights_in_column, InputData_in_column) + self.bias
    dotProduct = dotProduct.reshape(int(nFilters), 
                                    int(Hnext), 
                                    int(Wnext), 
                                    int(minibatchSize))
    dotProduct = dotProduct.transpose(3, 0, 1, 2)

    return (dotProduct)
  
  def back(self, perturbation, activation):
  # Re-adapted: https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/  
    Cnext, Hnext, Wnext = self.nextBatchShape
    perturbation = perturbation.reshape((-1, 
                                           int(Cnext), 
                                           int(Hnext), 
                                           int(Wnext)))      
    activation = activation.reshape(perturbation.shape)   
    
    padding, stride = self.config
    inputData, inputData_in_col = self.cachedInput.pop()
    nFilter, dFilter, hFilter, wFilter = self.Weights.shape

    biasGradient = np.sum(perturbation, axis=(0, 2, 3))
    biasGradient = biasGradient.reshape(nFilter, -1)

    perturbation = perturbation.transpose(1, 2, 3, 0)
    perturbation_in_col = perturbation.reshape(nFilter, -1)
    weightsGradient = np.dot(perturbation_in_col, inputData_in_col.T)
    weightsGradient = weightsGradient.reshape(self.Weights.shape)

    W_in_col = self.Weights.reshape(nFilter, -1)
    perturbation_in_col = np.dot(W_in_col.T, 
                                 perturbation_in_col)
    
    minibatchSize = inputData.shape[0]
    C, H, W = self.batchShape
    perturbation = sfs.col2im(perturbation_in_col, 
                              (minibatchSize, C, H, W), hFilter, wFilter, 
                              padding=padding, stride=stride)
    
    return (perturbation, inputData, weightsGradient, biasGradient)

""" ACTIVATION LAYER """
class Activation():
  
  def __init__(self, parameters):
    self.batchShape = parameters['batchShape']
    self.nextBatchShape = self.batchShape
    self.parameters = parameters
    self.af = afs.assignActivationFunction(parameters['activationFunction'])
    
  def fwd(self, inputData, TRAINING=False):
    return self.af(inputData)

  def back(self, perturbation, activation):
    derivActivation = self.af(activation, BACK=True)
    perturbation = np.multiply(perturbation,
                               derivActivation)
    return (perturbation, activation, None, None)


""" DROPOUT """
# Algorithm from paper: https://arxiv.org/abs/1207.0580
class Dropout():
  
  def __init__(self, parameters):
    self.batchShape = parameters['batchShape']
    self.nextBatchShape = self.batchShape
    self.threshold = parameters['threshold']
    self.parameters = parameters
    
  def fwd(self, inputData, TRAINING=False):
    if TRAINING:
      self.mask = (np.random.random(size=inputData.shape) < self.threshold)
      inputData = np.multiply(inputData,
                              self.mask / self.threshold)
    return inputData
    
  def back(self, perturbation, activation):
    # Reshape the perturbation if the preceeding layer is a conv layer
    if not np.isscalar(self.nextBatchShape):
      Cnext, Hnext, Wnext = self.nextBatchShape
      perturbation = perturbation.reshape((-1, 
                                           int(Cnext), 
                                           int(Hnext), 
                                           int(Wnext)))   
      activation = activation.reshape(perturbation.shape)   

    perturbation = np.multiply(perturbation, 
                               self.mask)
    return (perturbation, activation, None, None)
