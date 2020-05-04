import numpy as np
from Perceptron import Perceptron
# robione przy pomocy http://edu.pjwstk.edu.pl/wyklady/nai/scb/wyklad3/w3.htm
# http://www.neurosoft.edu.pl/media/pdf/jbartman/sztuczna_inteligencja/NTI%20cwiczenie_5.pdf

class NeuralNetwork:
  # ns - lista z ilościami neuronów w warstwach - np [3,2,1] - 3 inputy w warstwie pierwszej, 2 neurony w warstwie drugiej, 1 neuron na wyjsciu
  def __init__(self, ns, f, fprim, bias=[], r=0):
    self.bias = bias
    self.fprim = fprim
    self.network = [[Perceptron(f,np.random.rand(ns[i]+1) / 3 ,r) for j in range(percInLayer)] for (percInLayer, i) in zip(ns[1:],range(len(ns[1:])))]
  
  def classify(self, x, allNeuronsOutputs=False):
    xx = np.copy(x)
    allOutputs = []
    for (layer,layerBias) in zip(self.network, self.bias):
      xx = np.insert(xx,0,layerBias)
      xx = np.array([p.classify(xx) for p in layer])
      allOutputs.append(xx)
    return xx if not allNeuronsOutputs else allOutputs

  def learn(self,xk,dk,eta=0.1):
    allOutputs = self.classify(xk,True)
    # print(allOutputs)
    
    #stack of errors is each layers starting with output layer
    errors = [allOutputs[-1] - dk]

    for (layer, layerIndex) in zip(reversed(self.network),reversed(range(len(self.network)))):
      if layerIndex == len(self.network)-1: # dla warsty output
        factor = errors[-1] * self.fprim(allOutputs[layerIndex])
      else:
        factor = errors[-1][1:] * self.fprim(allOutputs[layerIndex])

      # print(factor)
      # errors.append(np.array([sum(factor * np.array([perc.w[percIndex] for perc in layer])) for percIndex in range(len(self.network[layerIndex-1]))]))
      wLen = len(layer[0].w)
      # print(wLen)
      # for wIndex in range(wLen):
      errorsInThisLayer = np.array([sum(factor * np.array([perc.w[wIndex] for perc in layer])) for wIndex in range(wLen)])
      errors.append(errorsInThisLayer)

    # wyliczanie wag
    errors = list(reversed(errors)) # by indexy xgadzały się w indexami warstw
    for (layer, err, layerIndex) in zip(self.network,errors[:-1],range(len(self.network))):
      for (perc, percIndex) in zip(layer, range(len(layer))):
        # print(allOutputs[layerIndex])
        # print("before ",perc.w)
        perc.w = perc.w + err * eta * allOutputs[layerIndex][percIndex]
        # print("after  ",perc.w)

        
alpha=1
f = lambda x: 1/(1+np.exp(-alpha*x))
fprim = lambda x: alpha * np.exp(-alpha*x) / (np.power(1+np.exp(-alpha*x),2))
eta = 0.5
nn = NeuralNetwork([2,2,1],f,fprim,[1,1])
# wagi początkowe (nadpisanie randomowych)
nn.network[0][0].w = np.array([0.86,-0.16,0.28])
nn.network[0][1].w = np.array([0.83,-0.51,-0.86])
nn.network[1][0].w = np.array([0.04,-0.43,0.48])

print(nn.network)
# print(nn.classify(np.array([2,2,2]), True))
# print(nn.classify(np.array([2,2,2])))
xk = np.array([[0,0],[0,1],[1,0],[1,1]])
dk = np.array([[0],[1],[1],[0]])
for i in range(1):
  nn.learn(xk[i % 4],dk[i % 4],eta)

#testing
for i in range(4):
  print(xk[i]," => ",nn.classify(xk[i]))

