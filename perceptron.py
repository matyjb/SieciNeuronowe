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
  def learn(self, xk, dk, grouped=False, t=0, debug=False):
    N = len(xk)
    oldw = np.copy(self.w)
    if(grouped):
      # wyznaczanie wartości y (w postaci wektora)
      yk = np.array([self.f(np.dot(xk[i], self.w)) for i in range(N)])
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
          print("k="+str(k)+" nowa waga: "+ str(self.w))
    
    print("t="+str(t)+" nowa waga: "+ str(self.w))
    if (self.w == oldw).all():
      # wagi nie zmieniły się od poprzedniego cyklu czyli koniec uczenia
      return t
    else:
      return self.learn(xk,dk,grouped,t+1,debug)
    

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

p0 = Perceptron(f,w0,r)
p0.learn(xk,dk)

print("----algorytm zgrupowany----")

p1 = Perceptron(f,w0,r)
p1.learn(xk,dk,grouped=True)
