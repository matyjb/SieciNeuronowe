import numpy as np

class HopfieldAsync:

  def __init__(self, f=None, w=[], b=[]):
    self.w = w
    self.f = f
    self.b = b

  # wylicza ?klasyfikacje? zadanego wektora i numer (czas) iteracji
  def classify(self, x, t=0):
    if t==40:
      return x
    print(x)
    v = np.copy(x)
    n = len(v)
    u = np.zeros(n)

    for i in range(n):
      u[i] = sum([v[j]*self.w[i,j]-self.b[j] for j in range(n)])
      v[i] = self.f(u[i])
    
    if not (v == x).all():
      return self.classify(v, t+1)
    else:
      return (v,t)

  # uczy sieć regułą Hebba
  # xk - lista wektorów uczących o tej samej N długości
  # resetuje wektor b na zera
  def learnHebb(self,xk):
    N = len(xk[0])
    K = len(xk)
    neww = np.zeros((N,N))
    for i in range(N):
      for j in range(N):
        if(i!=j):
          neww[i,j] = sum([xk[k,i] * xk[k,j] for k in range(K)])
    self.w = neww / N
    self.b = np.zeros(N)
