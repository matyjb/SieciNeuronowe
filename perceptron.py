import numpy as np

class Perceptron:
  def __init__(self, xk, dk, w0, r, f):
    self.xk = xk
    self.dk = dk
    self.w = w0
    self.r = r
    self.f = f
    self.k = -1

  def learningStep(self):
    # zinkrementuj indeks k
    self.k = (self.k + 1) % len(self.xk)

    x = self.xk[self.k]
    d = self.dk[self.k]

    y = self.f(np.dot(x, self.w))
    
    # obliczanie nowej wagi
    self.w = self.w + self.r * (d - y) * x
    

class PerceptronAlgZgrupowany:
  def __init__(self, xk, dk, w0, c, f):
    self.xk = xk
    self.dk = dk
    self.w = w0
    self.c = c
    self.f = f
    self.cycle = 0

  def learningStep(self):
    self.cycle += 1
    # wyznaczanie wartości y (w postaci wektora)
    yk = np.array([self.f(np.dot(self.xk[i], self.w)) for i in range(len(self.xk))])

    # obliczanie sumy źle rozpoznanych wektorów przez wagę w
    scalars = self.dk - yk # zawiera wartości: -1 0 lub 1
    ## kombinacja liniowa wektorów xk
    # s = np.sum([scalars[i] * self.xk[i] for i in range(len(scalars))], axis=0)
    s = np.dot(scalars, self.xk)

    # obliczanie nowej wagi
    self.w = self.w + self.c * s


w0 = np.array([1,1,1])
r = 1
f = lambda x: 1 if x > 0 else 0
dk = [0,0,1,0]
xk = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])

# testowe z ćwiczeń
# w0 = np.array([0.5,0,1])
# r = 1
# f = lambda x: 1 if x > 0 else 0
# dk = [0,0,1,0]
# xk = np.array([[1,0,0],[1,1,0],[1,1,1],[1,0,1]])

p0 = Perceptron(xk,dk,w0,r,f)
for i in range(10):
  p0.learningStep()
  print("k="+str(p0.k)+" nowa waga: "+ str(p0.w))

print("----algorytm zgrupowany----")

p1 = PerceptronAlgZgrupowany(xk,dk,w0,r,f)
for i in range(6):
  p1.learningStep()
  print("cykl="+str(p1.cycle)+" nowa waga: "+ str(p1.w))
