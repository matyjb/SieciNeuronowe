import numpy as np
from perceptron import Perceptron
from functions import Functions

# robione przy pomocy:
# http://edu.pjwstk.edu.pl/wyklady/nai/scb/wyklad3/w3.htm
# http://www.neurosoft.edu.pl/media/pdf/jbartman/sztuczna_inteligencja/NTI%20cwiczenie_5.pdf
# http://www-users.mat.umk.pl/~philip/propagacja2.pdf

class NeuralNetwork:
  # w - lista z wektorami wag dla poszczególnych perceptronów w poszczególnych warstwach zaczynająć od inputa
  def __init__(self, w, layersx0s=None, f=Functions.SINUS, alpha=1):
    if layersx0s is None:
      layersx0s = np.ones(np.size(w,axis=0))
    self.layersx0s = layersx0s
    self.network = list(map(lambda layer: [Perceptron(ww, f, alpha) for ww in layer],w))
  
  # x - wektor w postaci [x1,x2,x3...] | bez x0 !!! x0 doklejane juz w funkcji
  def classify(self, x, debug=False, withAllOutputs=False):
    allOutputs = [x]
    for (layer,biasInput) in zip(self.network,self.layersx0s):
      allOutputs[-1] = np.insert(allOutputs[-1],0,biasInput) # dodaj x0
      allOutputs.append(np.array([neuron.calcOutput(allOutputs[-1]) for neuron in layer]))

    if debug:
      print(allOutputs)
    return allOutputs[1:] if withAllOutputs else allOutputs[-1]

  # x - wektor w postaci [x1,x2,x3...] | bez x0 !!! x0 doklejane juz w funkcji
  def classifyPrim(self, x):
    prims = []
    currentLayerInput = np.copy(x) # zmienna pomocnicza
    for (layer,biasInput) in zip(self.network,self.layersx0s):
      currentLayerInput = np.insert(currentLayerInput,0,biasInput) # dodaj x0
      prims.append(np.array([neuron.calcOutputPrim(currentLayerInput) for neuron in layer]))
      currentLayerInput = np.array([neuron.calcOutput(currentLayerInput) for neuron in layer]) #output tej warstwy staje sie inputem następnej

    return prims

  def learnStep(self, xIn, dOut, eta=0.1):
    outputs = self.classify(xIn, withAllOutputs=True)
    outputsPrims = self.classifyPrim(xIn)
    deltas = [None] * len(self.network) # lista ze zmiennymi pomocniczymi: błąd * fprim, dla kazdego z neurona w poszczególnych warstwach
    
    # obliczenia dla wartwy output
    deltas[-1] = (dOut - outputs[-1]) * outputsPrims[-1]

    # propagacja błędów
    for layerIndex in reversed(range(len(self.network)-1)):
      errors = [None] * len(self.network[layerIndex]) # błedy dla kazdego neurona w layerIndex warstwie
      for neuronIndex in range(len(self.network[layerIndex])):
        wagi = np.array([neuron.w[neuronIndex+1] for neuron in self.network[layerIndex+1]])
        errors[neuronIndex] = np.sum(deltas[layerIndex+1]*wagi)

      deltas[layerIndex] = np.array(errors) * outputsPrims[layerIndex]
    
    xIn = np.insert(xIn,0,self.layersx0s[0])
    layersInputs = np.insert(outputs,0,None)[:-1]
    layersInputs[0] = xIn
    # poprawianie wag w[warstwa,neuron] += eta*deltas[warstwa,neuron]*layersInputs[warstwa]
    for layerIndex in range(len(self.network)):
      for neuronIndex in range(len(self.network[layerIndex])):
        wDelta = eta * layersInputs[layerIndex] * deltas[layerIndex][neuronIndex]
        self.network[layerIndex][neuronIndex].w += wDelta
  
  def learnStep2(self, xIn, dOut, eta=0.1, beta=1):
    outputs = self.classify(xIn, withAllOutputs=True)
    outputsPrims = self.classifyPrim(xIn)
    deltas = [None] * len(self.network) # lista ze zmiennymi pomocniczymi: błąd * fprim, dla kazdego z neurona w poszczególnych warstwach
    
    # obliczenia dla wartwy output
    deltas[-1] = (dOut - outputs[-1])*beta

    # propagacja błędów
    for layerIndex in reversed(range(len(self.network)-1)):
      errors = [None] * len(self.network[layerIndex]) # błedy dla kazdego neurona w layerIndex warstwie
      for neuronIndex in range(len(self.network[layerIndex])):
        wagi = np.array([neuron.w[neuronIndex+1] for neuron in self.network[layerIndex+1]])
        errors[neuronIndex] = np.sum(deltas[layerIndex+1]*wagi)

      deltas[layerIndex] = np.array(errors)*beta*beta
    
    xIn = np.insert(xIn,0,self.layersx0s[0])
    layersInputs = np.insert(outputs,0,None)[:-1]
    layersInputs[0] = xIn
    # poprawianie wag w[warstwa,neuron] += eta*deltas[warstwa,neuron]*layersInputs[warstwa]
    for layerIndex in range(len(self.network)):
      for neuronIndex in range(len(self.network[layerIndex])):
        if layerIndex == len(self.network)-1:
          wDelta = eta * layersInputs[layerIndex] * deltas[layerIndex][neuronIndex]
        else:
          wDelta = eta * (1-np.power(layersInputs[layerIndex],2)) * deltas[layerIndex][neuronIndex]
        self.network[layerIndex][neuronIndex].w += wDelta

  # xk - array wektorów uczących
  # dk - array wartości oczekiwanych
  # eta - stała uczenia
  def learn(self, xk, dk, eta=0.1, iterations=15000):
    for i in range(iterations):
      for (x,d) in zip(xk,dk):
        self.learnStep(x, d, eta)
  
  def learn2(self, xk, dk, eta=0.1, iterations=15000):
    for i in range(iterations):
      for (x,d) in zip(xk,dk):
        self.learnStep2(x, d, eta, 1)

  def __repr__(self):
    s = ""
    for (layer,i) in zip(self.network,range(len(self.network))):
      s += ">warstwa " + str(i) + "\n" + str(np.array(layer)) + "\n"

    return s + "stale wejscia w warstwach: " +str(self.layersx0s)
  def __str__(self):
    return self.__repr__()
