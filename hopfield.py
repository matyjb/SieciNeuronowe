import numpy as np
from functions import Functions

class Modes:
  SYNC = 0
  ASYNC = 1

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
  def classify(self, x, t=0, debug=False):
    if t==40:
      return (x,t)
    
    if debug and t==0:
      print("%4s | %20s | %20s | %20s" % ("czas","v","u","nowe v"))


    v = np.copy(x)
    n = len(v)
    # u = np.zeros(n)

    if self.mode == Modes.ASYNC:
      for i in range(n):
        vOld = np.copy(v)
        u = (v*self.w).A1 + self.b
        v[i] = self.f(u[i])
        if debug:
          print("%4s | %20s | %20s | %20s" % (t,vOld,u,v))
    elif self.mode == Modes.SYNC:
      vOld = np.copy(v)
      u = (v*self.w).A1 + self.b
      v = np.array([self.f(ui) for ui in u])
      if debug:
        print("%4s | %20s | %20s | %20s" % (t,vOld,u,v))
    
    if not (v == x).all():
      return self.classify(v, t+1, debug)
    else:
      return (v,t)
  


  # dobiera wagi sieci regułą Hebba
  # xk - lista wektorów uczących o tej samej N długości
  # resetuje wektor b na zera
  def hebb(self,xk):
    N = len(xk[0])
    K = len(xk)
    neww = np.zeros((N,N))
    for i in range(N):
      for j in range(N):
        if(i!=j):
          neww[i,j] = sum([xk[k,i] * xk[k,j] for k in range(K)])
    self.w = np.matrix(neww / N)
    self.b = np.zeros(N)
