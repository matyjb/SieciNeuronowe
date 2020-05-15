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
  
  # x - wektor w postaci [x1,x2,x3...] | bez x0 !!! x0=1 doklejane juz w funkcji
  def classify(self, x, debug=False):
    currentLayerOutput = np.copy(x) # zmienna pomocnicza
    for (layer,biasInput) in zip(self.network,self.layersx0s):
      currentLayerOutput = np.insert(currentLayerOutput,0,biasInput) # dodaj x0
      currentLayerOutput = np.array([neuron.calcOutput(currentLayerOutput) for neuron in layer])

    return currentLayerOutput

  # TODO - rewrite
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
    return str(self.network) + "\nwarstw = " + str(len(self.network))  + " + 1 warstwa z inputami"
  def __str__(self):
    return self.__repr__()
