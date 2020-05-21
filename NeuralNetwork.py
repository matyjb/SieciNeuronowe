import numpy as np
from perceptron import Perceptron
from functions import Functions
import matplotlib.pyplot as plt

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

  def _backPropGetWDeltas(self, xIn, dOut, eta=0.1):
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
    # wyliczanie zmian wag wDelta[warstwa,neuron] += eta*deltas[warstwa,neuron]*layersInputs[warstwa]
    wDeltas = []
    for layerIndex in range(len(self.network)):
      tmp = []
      for neuronIndex in range(len(self.network[layerIndex])):
        tmp.append(eta * layersInputs[layerIndex] * deltas[layerIndex][neuronIndex])
      wDeltas.append(np.array(tmp))
    return np.array(wDeltas)
    

  # wDelta - zmiany wag w poszczególnych warstwach w poszczególnych neuronach
  def _addWDelta(self, wDeltas):
    for layerIndex in range(len(self.network)):
      for neuronIndex in range(len(self.network[layerIndex])):
        self.network[layerIndex][neuronIndex].w += wDeltas[layerIndex][neuronIndex]

  # xk - array wektorów uczących
  # dk - array wartości oczekiwanych
  # eta - stała uczenia
  # mode - energia całkowita/cząstkowa [False/True]
  def learn(self, xk, dk, eta=0.1, iterations=15000, mode=True):
    coIle = 500
    energia = []
    for i in range(iterations):
      if mode:
        # cząstkowa
        for (x,d) in zip(xk,dk):
          if i % coIle == 0:
            energia.append(1/2*np.sum(np.power((d - self.classify(x)),2)))
          wDeltas = self._backPropGetWDeltas(x,d,eta)
          self._addWDelta(wDeltas)
      else:
        # całkowita
        wDeltasSum = self._backPropGetWDeltas(xk[0],dk[0],eta)
        energiaSum = 0
        for (x,d) in zip(xk[1:],dk[1:]):
          if i % coIle == 0:
            energiaSum += 1/2*np.sum(np.power((d - self.classify(x)),2))

          wDeltasSum += self._backPropGetWDeltas(x,d,eta)

        if i % coIle == 0:
          energia.append(energiaSum)
          
        self._addWDelta(wDeltasSum)
    
    plt.plot(np.array(range(len(energia)))*coIle,energia)
    plt.title("energia cząstkowa" if mode else "energia całkowita")
    plt.xlabel("iteracja")
    plt.ylabel("energia")
    plt.show()



  def __repr__(self):
    s = ""
    for (layer,i) in zip(self.network,range(len(self.network))):
      s += ">warstwa " + str(i) + "\n" + str(np.array(layer)) + "\n"

    return s + "stale wejscia w warstwach: " +str(self.layersx0s)
  def __str__(self):
    return self.__repr__()
