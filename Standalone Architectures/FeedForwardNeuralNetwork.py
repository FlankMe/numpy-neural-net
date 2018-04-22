"""
Implementation of a feed-forward neural network in Numpy with L2 regularization

The weights can be calibrated via
- Stochastic Gradient Descent (function self.backPropSGD)
- Adam Optimiser              (function self.backPropAdam)

Here is an example of how to create an instance:
  layers = [10, 50, 30, 10, 5] 
  # 10 is the number of inputs 
  # 50, 30, 10 are the sizes of the hidden layers
  # 5 is the size of the output 
  net = FeedForwardNeuralNetwork(layers)

(https://github.com/flankme)
"""

import numpy as np
import time; np.random.seed(int(time.time()))

class FeedForwardNeuralNetwork:
    
    def __init__(self, layers):
        '''
        Accept a list of integers as architecture for the network, of the 
        form [input, hidden_1, hidden_2, ..., hidden_n, output]
        '''
        self._generateNetwork(np.array(layers))
        self._defineActivationFunction()        

    ''' Leaky ReLU unit '''
    def _defineActivationFunction(self):

        leakyCoeff = 0.05
        self._actFunction = lambda x : np.maximum(leakyCoeff * x, x)
        self._derFunction = lambda x : leakyCoeff * (x<=0) + (x>0)

    ''' Mean squared errors as cost function '''
    def _costFunction(self, X, y, L2PENALTY=False):
    
        # perturbation is the gradient of the cost function w.r.t. the output
        y = np.atleast_2d(y).T if len(y.shape) == 1 else np.atleast_2d(y)
        perturbation = self.predict(X) - y   # (batch x outputs)

        # Calculate the total error
        error = 0.5 * (perturbation * perturbation).sum() / len(X)
        if L2PENALTY:
            for weights in self._weights:
                error += 0.5 * L2PENALTY * (weights * weights).sum()
        
        return (error, perturbation)
   
    def predict(self, X):

        activation = np.atleast_2d(X)  # if X is a vector, it has shape (1 x m) 
        self._actValues = [activation] # create a list of cashed activations
        
        # Feed the input through the layers and store the activation values
        for weights in self._weights[:-1]:
            activation = self._addOnes(activation)            
            hidden_dotProduct = np.dot(activation, 
                                       weights)
            activation = self._actFunction(hidden_dotProduct)
            self._actValues.append(activation)
            
        # The Last layer does not require to store the activation function
        activation = self._addOnes(activation)
        activation = np.dot(activation, self._weights[-1])
        return(activation)          

    ''' Stochastic Gradient Descent '''
    def backPropSGD(self, X, y, LEARNINGRATE=3e-4, L2PENALTY=1e-6):    

        # LEARNINGRATE is the learning rate
        # L2PENALTY is the coefficient of the L2 regularisation
    
        error, perturbation = self._costFunction(X, y, L2PENALTY=L2PENALTY)

        # BACK-PROPAGATE THROUGH THE NETWORK
        # This is a function of the weights, activation values, and the
        # derivatives of the activation function
        updatedWeights = []
        while self._weights:
            
            weights = self._weights.pop()
            activation = self._actValues.pop()
            activation = self._addOnes(activation)
            
            weightsGradient = np.dot(activation.T, 
                                     perturbation)
            weightsGradient = self._clipGradient(weightsGradient)                                    
            updatedWeights.append(weights * (1 - L2PENALTY) - 
                                  LEARNINGRATE * weightsGradient)

            perturbation = np.multiply(np.dot(perturbation,
                                              weights.T),
                                       self._derFunction(activation))
            perturbation = perturbation[:, :-1]
        
        updatedWeights.reverse()
        self._weights = updatedWeights
            
        return (error)
    
    ''' Adam optimiser '''
    # https://arxiv.org/pdf/1412.6980.pdf
    def backPropAdam(self, X, y, LEARNINGRATE=3e-4, L2PENALTY=1e-6, 
                     BETA1=0.9, BETA2=0.999, EPSILON=1e-8):   
        # LEARNINGRATE is the learning rate
        # L2PENALTY is the coefficient of the L2 regularisation

        error, perturbation = self._costFunction(X, y, L2PENALTY=L2PENALTY)

        # BACK-PROPAGATE THROUGH THE NETWORK
        updatedWeights = []
        updatedMomentum = [] 
        updatedVariance = []
        
        while self._weights:
            
            weights = self._weights.pop()
            momentum = self._AdamMomentum.pop()
            variance = self._AdamVariance.pop()
            
            activation = self._actValues.pop()
            activation = self._addOnes(activation)
            
            gradient = np.dot(activation.T, perturbation)
            momentum = BETA1 * momentum + (1 - BETA1) * gradient
            variance = BETA2 * variance + (1 - BETA2) * (gradient * gradient)
            
            weightsGradient = momentum / (np.sqrt(variance) + EPSILON)
            weightsGradient = self._clipGradient(weightsGradient)                                    

            adjAlpha = (LEARNINGRATE * 
                        np.sqrt(1 - self._AdamBETA2) / 
                        (1 - self._AdamBETA1))

            updatedMomentum.append(momentum)
            updatedVariance.append(variance)
            updatedWeights.append(weights * (1 - L2PENALTY) - 
                                  adjAlpha * weightsGradient)

            perturbation = np.multiply(np.dot(perturbation,
                                              weights.T),
                                       self._derFunction(activation))
            perturbation = perturbation[:, :-1]

        updatedWeights.reverse()
        self._weights = updatedWeights
        updatedMomentum.reverse()
        self._AdamMomentum = updatedMomentum
        updatedVariance.reverse()
        self._AdamVariance = updatedVariance
        
        self._AdamBETA1 *= BETA1
        self._AdamBETA2 *= BETA2            
        
        return (error)
        
    def _generateNetwork(self, layers):
        BIAS = 0.0
        self._weights = []

        for i in range(layers.size - 1):
            weights = np.random.randn(layers[i], layers[i+1]) 
            weights *= np.sqrt(2.0 / layers[i])         # rescale the weights
            bias = BIAS * np.ones((1, layers[i+1]))
            self._weights.append(np.concatenate((weights, 
                                                 bias), 
                                                axis=0))
            
        # Function that adds a column of 1s to allow for a bias coefficient
        self._addOnes = lambda x : np.concatenate((x, 
                                                   np.ones((x.shape[0], 1))),
                                                  axis=1)
        
        # Initialise the time step t for Adam optimiser in case it is used
        self._AdamBETA1 = 0.9
        self._AdamBETA2 = 0.999
        self._AdamMomentum = []
        self._AdamVariance = []
        for weights in self._weights:
            self._AdamMomentum.append(np.zeros(weights.shape))
            self._AdamVariance.append(np.zeros(weights.shape))
        
        # Define the function that clips the gradient above a certain threshold
        maxGradient = 50.0
        self._clipGradient = lambda x : np.minimum(maxGradient, 
                                                   np.maximum(-maxGradient, x))