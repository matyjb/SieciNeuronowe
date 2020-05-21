import numpy as np
from functions import Functions

class Perceptron:
  ###
  # w - lista wag na wejsciach neurona
  # f - typ funkcji wyjsciowej (jedna z Functions)
  # alpha - parametr alpha do funkcji wysciowej w przypadku sinusoidalnej/tangensoidalnej
  ###
  def __init__(self, w, f=Functions.STEP, alpha=1):
    self.w = w
    self.fname = f[0]
    self.f = lambda x: f[1](x,alpha)
    
    # uzywane w NeuralNetwork
    self.fprim = lambda x: f[2](x,alpha)

  def getZ(self, xIn):
    return np.dot(xIn,self.w)

  def calcOutput(self,xIn):
    return self.f(self.getZ(xIn))

  # uzywane w NeuralNetwork
  def calcOutputPrim(self,xIn):
    return self.fprim(self.getZ(xIn))

  # uczy perceptron przy pomocy listy wektorów uczacych xk i wektora watości oczekiwanych dk
  # eta - stała uczenia
  # grouped - czy użyć algorytmu zgrupowanego
  # debug - czy printowac nowe wagi w kazdych krokach
  # zwraca ilość cykli, która zajęła by ustabilizować wagi i koncowe wagi
  def learn(self, xk, dk, eta = 1, grouped=False, cycle=0, debug=False, maxCycles=1000):
    if debug and cycle == 0:
      if grouped:
        print("%4s | %20s" % ("cykl","nowa waga"))
      else:
        print("%4s | %5s | %20s | %20s | %7s | %7s | %7s | %20s" % ("cykl","czas","wektor uczący","waga","z","y","d","nowa waga"))

    n = len(xk)
    
    initW = np.copy(self.w) # wartość wag z jaką cykl się rozpoczą 
    # (używana potem do sprawdzenia czy po wykonaniu cyklu wagi sie zmieniły i  czy uruchomić kolejny cykl)
    
    if(grouped):
      # wyznaczanie wartości y (w postaci wektora)
      yk = np.array([self.calcOutput(xk[i]) for i in range(n)])
      # obliczanie sumy źle rozpoznanych wektorów przez wagę w
      scalars = dk - yk # zawiera wartości: -1 0 lub 1
      ## kombinacja liniowa wektorów xk
      s = np.dot(scalars, xk)
      # obliczanie nowej wagi
      self.w = self.w + eta * s
      if debug:
        print("%4s | %20s" % (cycle, self.w))
    else:
      for k in range(n):
        x = xk[k]
        d = dk[k]
        y = self.calcOutput(x)

        oldw = np.copy(self.w) # wartosc uzywana tylko do printa
        z = self.getZ(x) # wartosc uzywana tylko do printa
        # obliczanie nowej wagi
        self.w = self.w + eta * (d - y) * x
        if(debug):
          print("%4s | %5s | %20s | %20s | %7s | %7s | %7s | %20s" % (cycle, cycle*n+k, x,oldw,z,y,d,self.w))
          # print(str(cycle) + "\t| " + str(cycle*n+k) + "\t| " + str(x) + "\t| " + str(oldw) + "\t| " + str(self.getZ(x)) + "\t| " + str(y) + "\t| " + str(d) + "\t| " + str(self.w))
    
    if (self.w == initW).all() or maxCycles==1:
      # wagi nie zmieniły się od poprzedniego cyklu czyli koniec uczenia
      # lub został osiągnięty limit cykli
      return (self.w, cycle)
    else:
      return self.learn(xk,dk,eta,grouped,cycle+1,debug, maxCycles-1)

  def __repr__(self):
    return "Perceptron | wagi = " + str(self.w) + " | funkcja: " + self.fname
  def __str__(self):
    return self.__repr__()