import numpy as np
from Perceptron import Perceptron
# robione przy pomocy http://edu.pjwstk.edu.pl/wyklady/nai/scb/wyklad3/w3.htm
# http://www.neurosoft.edu.pl/media/pdf/jbartman/sztuczna_inteligencja/NTI%20cwiczenie_5.pdf

class NeuralNetwork:
  # ns - lista z ilościami neuronów w warstwach - np [3,2,1] - 3 inputy w warstwie pierwszej, 2 neurony w warstwie drugiej, 1 neuron na wyjsciu
  def __init__(self, ns, f, fprimfx, bias, r=0):
    self.bias = bias
    self.fprimfx = fprimfx
    self.network = [[Perceptron(f,np.random.rand(ns[i]+1) / 3 ,r) for j in range(percInLayer)] for (percInLayer, i) in zip(ns[1:],range(len(ns[1:])))]
  
  def classify(self, x, allNeuronsOutputs=False):
    # xx = np.copy(x)
    allOutputs = [np.insert(x,0,self.bias[0])]
    for (layer,layerBiasIndex) in zip(self.network, range(len(self.bias))):
      # xx = np.insert(xx,0,layerBias)
      xx = np.array([p.classify(allOutputs[-1]) for p in layer])
      if layerBiasIndex != len(self.bias) - 1:
        allOutputs.append(np.insert(xx,0,self.bias[layerBiasIndex + 1]))
      else:
        allOutputs.append(xx)
    return allOutputs[-1] if not allNeuronsOutputs else allOutputs

  def learn(self,xk,dk,eta=0.1):
    allOutputs = self.classify(xk,True)
    # print(allOutputs)
    neuronsOutPrims = [self.fprimfx(allOutputs[i]) for i in range(len(allOutputs))]
    # print(neuronsOutPrims)
    
    #stack of errors in outputs of each layer starting with output layer
    errors = [dk - allOutputs[-1]]
    # stack of deltas in each layer neurons starting with ouput layer
    deltas = [errors[0] * self.fprimfx(allOutputs[-1])]
    for (layer,layerIndex) in reversed(list(zip(self.network[:-1],range(len(self.network[:-1]))))):
      # print("layer: ", layerIndex)
      z = []
      for neuronIndex in range(len(layer)+1): #+1 bo jest jeszcze dummy neuron
        # print(">neuron: ", neuronIndex)
        # zebrac wagi z warstwy nastepnej dla tego neurona
        wagi = np.array([nextNeuron.w[neuronIndex] for nextNeuron in self.network[layerIndex+1]])
        # print(">>wagi: ", wagi)
        # print(">>primki: ", neuronsOutPrims[layerIndex+2])
        # print(">>bledy: ", errors[-1])
        err = np.sum(wagi*neuronsOutPrims[layerIndex+2]*errors[-1])
        # print(">>* sum: ", err)
        z.append(err)
      deltas.append(self.fprimfx(allOutputs[layerIndex+1]) * z)
      errors.append(np.array(z))

    errors = list(reversed(errors))
    deltas = list(reversed(deltas))
    ########
    # print("zs: ",errors)
    # print("delty: ",deltas)
    #dostosywanie wag
    # w = w + eta*deltas[i] * allOutputs[i]
    for (layer,layerIndex) in zip(self.network,range(len(self.network))):
      # print(">layer: ",layerIndex)
      # print("outputy: ",allOutputs[layerIndex])
      # print("delty:   ",deltas[layerIndex])
      for (neuron,neuronIndex) in zip(layer,range(len(layer))):
        if layerIndex == len(self.network) - 1:
          neuron.w += eta*deltas[layerIndex][neuronIndex]*allOutputs[layerIndex]
        else:
          neuron.w += eta*deltas[layerIndex][neuronIndex+1]*allOutputs[layerIndex]
        # print("nowe w: ",neuron.w)

  def __repr__(self):
    return str(self.network) + "\nwarstw = " + str(len(self.network))  + " + 1 warstwa z inputami\nwyjscia dummy neuronów = " + str(self.bias) + " (w kolejnych warstwach)"
  def __str__(self):
    return self.__repr__()

# alpha=1
# f = lambda x: 1/(1+np.exp(-alpha*x))
# fprimfx = lambda x: x*(1-x)
# eta = 0.1
# nn = NeuralNetwork([2,2,1],f,fprimfx,[1,1])
# # wagi początkowe (nadpisanie randomowych)
# nn.network[0][0].w = np.array([0.2, 0.1,  0.2])
# nn.network[0][1].w = np.array([0.1, 0.3, 0.1])
# nn.network[1][0].w = np.array([-0.2, -0.1,  0.2])

# # print(nn)
# xk = np.array([[1,0]])
# dk = np.array([ [1] ])
# nn.learn(xk[0],dk[0],eta)


print("##########################################")
alpha=1
f = lambda x: 1/(1+np.exp(-alpha*x))
fprim = lambda x: alpha * np.exp(-alpha*x) / (np.power(1+np.exp(-alpha*x),2))
eta = 0.5
nn = NeuralNetwork([2,2,1],f,fprim,[1,1])
# wagi początkowe (nadpisanie randomowych)
nn.network[0][0].w = np.array([0.86, -0.16,  0.28])
nn.network[0][1].w = np.array([0.83, -0.51, -0.86])
nn.network[1][0].w = np.array([0.04, -0.43,  0.48])

print(nn)
# print(nn.classify(np.array([2,2,2]), True))
# print(nn.classify(np.array([2,2,2])))
xk = np.array([[0,0],[0,1],[1,0],[1,1]])
dk = np.array([ [0] , [1] , [1] , [0] ])
for i in range(40000):
# for i in range(1):
  nn.learn(xk[i % 4],dk[i % 4],eta)

#testing
for i in range(4):
  print(xk[i]," => ",nn.classify(xk[i]))

