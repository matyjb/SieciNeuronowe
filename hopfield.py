import numpy as np
from functions import Functions

class Modes:
  SYNC = "synchroniczny"
  ASYNC = "asynchroniczny"

class Hopfield:

  # ###
  # w - kwadratowa macierz wag (np.matrix)
  # f - typ funkcji wyjsciowej dla nauronów (jedna z Functions)
  # alpha - parametr alpha do funkcji wysciowej w przypadku sinusoidalnej/tangensoidalnej
  # b - wektor z bias'ami
  # mode - tryb sync/async
  ###
  def __init__(self, w, f=Functions.STEP, alpha=1, b=None, mode=Modes.ASYNC):
    self.w = np.matrix(w)
    if b is None:
      b = np.zeros(np.size(w,axis=0))
    self.b = b
    
    self.fname = f[0]
    self.f = lambda x: f[1](x,alpha)
    self.fprim = lambda x: f[2](x,alpha)

    self.mode = mode

  # wylicza ?klasyfikacje? zadanego wektora
  def classify(self, x, t=1, debug=False, xtminus2=None):
    
    if debug and t==1:
      print("%4s | %20s | %40s | %20s" % ("czas","v","u","nowe v"))


    v = np.copy(x)
    n = len(v)
    # u = np.zeros(n)

    if self.mode == Modes.ASYNC:
      for i in range(n):
        vOld = np.copy(v)
        u = (v*self.w).A1 + self.b
        v[i] = self.f(u[i])
        if debug:
          print("%4s | %20s | %40s | %20s" % (t,vOld,u,v))
    elif self.mode == Modes.SYNC:
      # jesli w jest symetryczna to nowe v moze skakac miedzy dwoma stanami lub zbiegac do jednego
      vOld = np.copy(v)
      u = (v*self.w).A1 + self.b
      v = np.array([self.f(ui) for ui in u])
      if debug:
        print("%4s | %20s | %40s | %20s" % (t,vOld,u,v))
    
    if xtminus2 is not None and (xtminus2 == v).all():
      # metastabilny
      return ((x,xtminus2),t)
    elif (v == x).all():
      # nic sie nie zmieniło po iteracji wiec koniec klasyfikacji
      return (v,t)
    else:
      return self.classify(v, t+1, debug, x)
  
  def __repr__(self):
    return "Hopfield | " + str(np.size(self.w, axis=0)) + " wejść \t| tryb: "+str(self.mode)+" \t| funkcja aktywacji: "+self.fname+"\n wagi = \n" + str(self.w)
  def __str__(self):
    return self.__repr__()