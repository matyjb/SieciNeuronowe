import numpy as np
from Perceptron import Perceptron
# robione przy pomocy http://edu.pjwstk.edu.pl/wyklady/nai/scb/wyklad3/w3.htm

class NeuralNetwork:
  # ns - lista z ilościami neuronów w warstwach - np [3,2,1] - 3 inputy w warstwie pierwszej, 2 neurony w warstwie drugiej, 1 neuron na wyjsciu
  def __init__(self, ns, r=0, alpha=1):
    f = lambda x: (1-np.exp(-alpha*x))/(1+np.exp(-alpha*x))
    self.network = [[Perceptron(f,np.random.rand(ns[i]) / 3 ,r) for j in range(percInLayer)] for (percInLayer, i) in zip(ns[1:],range(len(ns[1:])))]
    # for i in ns[1:]:
    #   for j in range(i):
    #     w = np.random.rand(ns[i-1]) / 3 # wektor wag wartości nie przekraczjących 0.(3)
  
  def classify(self,x):
    xx = np.copy(x)
    for layer in self.network:
      xx = np.array([p.classify(xx) for p in layer])
    return xx
        

nn = NeuralNetwork([3,2,1])
print(nn.network)
print(nn.classify(np.array([2,2,2])))



