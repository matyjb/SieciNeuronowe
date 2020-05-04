import numpy as np

class Perceptron:
  def __init__(self, f, w=[], r=1):
    self.w = w
    self.r = r
    self.f = f

  def classify(self,x):
    xw = np.dot(x,self.w)
    return self.f(xw)

  # uczy perceptron przy pomocy listy wektorów xk i wektora dk
  # grouped - czy użyć algorytmu zgrupowanego
  # debug - czy printowac nowe wagi w kazdych krokach
  # zwraca ilość cykli, która zajęła by ustabilizować wagi
  def learn(self, xk, dk, grouped=False, t=0, debug=False, maxCycles=np.Inf):
    N = len(xk)
    oldw = np.copy(self.w)
    if(grouped):
      # wyznaczanie wartości y (w postaci wektora)
      yk = np.array([self.classify(xk[i]) for i in range(N)])
      # obliczanie sumy źle rozpoznanych wektorów przez wagę w
      scalars = dk - yk # zawiera wartości: -1 0 lub 1
      ## kombinacja liniowa wektorów xk
      s = np.dot(scalars, xk)
      # obliczanie nowej wagi
      self.w = self.w + self.r * s
    else:
      for k in range(N):
        x = xk[k]
        d = dk[k]
        y = self.f(np.dot(x, self.w))
        # obliczanie nowej wagi
        self.w = self.w + self.r * (d - y) * x
        if(debug):
          print(">k="+str(k)+" wektor uczący: "+str(x)+" nowa waga: "+ str(self.w))
    
    print(">>t="+str(t)+" nowa waga: "+ str(self.w) + "\n")
    if (self.w == oldw).all() or maxCycles==1:
      # wagi nie zmieniły się od poprzedniego cyklu czyli koniec uczenia
      # lub został osiągnięty limit cykli
      return t
    else:
      return self.learn(xk,dk,grouped,t+1,debug, maxCycles-1)

  def __repr__(self):
    return "Perceptron wejścia: " + str(np.size(self.w)) + " (bias = " + str(self.r) + ")"
  def __str__(self):
    return self.__repr__()
   