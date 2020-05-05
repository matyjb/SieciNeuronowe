import numpy as np
from Perceptron import Perceptron
from enum import Enum

# robione przy pomocy:
# http://edu.pjwstk.edu.pl/wyklady/nai/scb/wyklad3/w3.htm
# http://www.neurosoft.edu.pl/media/pdf/jbartman/sztuczna_inteligencja/NTI%20cwiczenie_5.pdf
# http://www-users.mat.umk.pl/~philip/propagacja2.pdf

class FunctionsForNetwork(Enum):
    SINUS = 1
    TANGENS = 2

class NeuralNetwork:
  # ns - lista z ilościami neuronów w warstwach - np [3,2,1] - 3 inputy w warstwie pierwszej, 2 neurony w warstwie drugiej, 1 neuron na wyjsciu
  def __init__(self, ns, functionType, alpha, bias, r=0):
    self.bias = bias
    if functionType == FunctionsForNetwork.TANGENS:
      f = lambda x: (1-np.exp(-alpha*x))/(1+np.exp(-alpha*x))
      fprimfx = lambda x: alpha*(1-np.power(x,2))/2
    else:
      f = lambda x: alpha/(1+np.exp(-alpha*x))
      fprimfx = lambda x: x*(1-x)
    
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
    # trzyma wektory (ktore odpowiadają warstwom) które mają wyniki dla poszczególnych neuronów w danej warstwie
    # + tez wektor inputowy xk na początku (wraz z biasem)
    allOutputs = self.classify(xk,True)
    # print(allOutputs)
    # trzyma wyliczone pochodne dla allOutputs (trzeba pamietac ze fprimfx jest funkcją o argumencie f(x) )
    neuronsOutPrims = [self.fprimfx(allOutputs[i]) for i in range(len(allOutputs))]
    # print(neuronsOutPrims)
    
    # przechowuje ostatnie błedy dla neuronów w danej warstwie (czyli tutaj dla ostatniej warstwy (i to bedzie w przypadku XORa wektor o dlugosci 1))
    lastZ = dk - allOutputs[-1]
    # pomocnicza lista przechowujaca sumy błędów na wyjściach pomnozone przez wage i wartość f'(x) poszczególnych neuronów dla kolejnych warstw zaczynając od warstwy ostatniej
    deltas = [lastZ * self.fprimfx(allOutputs[-1])]
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
        err = np.sum(wagi*neuronsOutPrims[layerIndex+2]*lastZ)
        # print(">>* sum: ", err)
        z.append(err)
      deltas.append(self.fprimfx(allOutputs[layerIndex+1]) * z)
      lastZ = np.array(z)

    deltas = list(reversed(deltas))
    # print("delty: ",deltas)
    #dostosywanie wag
    # w = w + eta*deltas[i] * allOutputs[i]
    for (layer,layerIndex) in zip(self.network,range(len(self.network))):
      # print(">layer: ",layerIndex)
      # print("outputy: ",allOutputs[layerIndex])
      # print("delty:   ",deltas[layerIndex])
      for (neuron,neuronIndex) in zip(layer,range(len(layer))):
        if layerIndex == len(self.network) - 1:
          # na wyjsciu mamy tylko neurony (brak dummy neurona) dlatego bez +1 przy neuronIndex
          neuron.w += eta*deltas[layerIndex][neuronIndex]*allOutputs[layerIndex]
        else:
          neuron.w += eta*deltas[layerIndex][neuronIndex+1]*allOutputs[layerIndex]
        # print("nowe w: ",neuron.w)

  def __repr__(self):
    return str(self.network) + "\nwarstw = " + str(len(self.network))  + " + 1 warstwa z inputami\nwyjscia dummy neuronów = " + str(self.bias) + " (w kolejnych warstwach)"
  def __str__(self):
    return self.__repr__()
