import numpy as np
import itertools

class HopfieldAsync:

  def __init__(self, f=None, w=[], b=[]):
    self.w = w
    self.f = f
    self.b = b

  # wylicza ?klasyfikacje? zadanego wektora i numer (czas) iteracji
  def classify(self, x, t=0):
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

# funkcja pomocnicza do wyswietlenia wynikow zadania
def exec(x,f,w,b):
  h = HopfieldAsync(f, w, b)
  for i in range(len(x)):
    xIn = x[i]
    (xOut,t) = h.classify(xIn)
    if t == 0:
      print(str(xIn) + " => " + str(xOut) + " t=" + str(t) + " punkt stały!")
    else:
      print(str(xIn) + " => " + str(xOut) + " t=" + str(t))


# generuje wszystkie kombinacje wektora o n długości
# i dwóch mozliwych wartościach
def genxs(n, v1, v2):
  return np.array([list(i) for i in itertools.product([v1, v2], repeat=n)])

# print("---przykład z notatek---")
w = np.array([[0,1],[1,0]])
x = genxs(2,0,1)
b = np.array([0,0])
f = lambda x: 1 if x > 0 else 0
exec(x,f,w,b)

# to psuje
# print("---przykład z ZadRozwiazaniaHopfield.pdf str 9---")
# w = np.array([[-1,3/4],[3/4,0]])
# x = genxs(2,0,1)
# b = np.array([0,0])
# f = lambda x: 1 if x >= 0 else 0
# exec(x,f,w,b)

print("---przykład z ZadRozwiazaniaHopfield.pdf str 13---")
w = np.array([[0,1,-1],[1,0,1],[-1,1,0]])
x = genxs(3,0,1)
b = np.array([0,0,0])
f = lambda x: 1 if x > 0 else 0
exec(x,f,w,b)

print("---projekt---")
c = 2/3
w = np.array([[0,-c,c],[-c,0,-c],[c,-c,0]])
x = genxs(3,-1,1)
b = np.array([0,0,0])
f = lambda x: 1 if x > 0 else -1
exec(x,f,w,b)

print("---przykładzik z internetu z tic tac toe---")
h = HopfieldAsync(f=f)
# trzy obrazy 3x3 (kółko, krzyzyk, puste):
# C C C | C B C | B B B
# C B C | B C B | B B B
# C C C | C B C | B B B
# C = 1 B = -1
xk = np.array([[1,1,1,1,-1,1,1,1,1],[1,-1,1,-1,1,-1,1,-1,1],[-1,-1,-1,-1,-1,-1,-1,-1,-1]])
h.learnHebb(xk)
# print("nowe wagi:")
# print(h.w)
# print("---klasyfikacja kolka i krzyzyk---")
x = genxs(9,-1,1)
stale = 0
kolka = 0
krzyzyki = 0
puste = 0
for i in range(len(x)):
    xIn = x[i]
    (xOut,t) = h.classify(xIn)
    # if t == 0:
    #   print(str(xIn) + ";" + str(xOut) + ";" + str(t) + ";1")
    # else:
    #   print(str(xIn) + ";" + str(xOut) + ";" + str(t) + ";0")
    if t == 0:
      stale += 1
      print(xOut)
    if (xOut == xk[0]).all():
      kolka += 1
    if (xOut == xk[1]).all():
      krzyzyki += 1
    if (xOut == xk[2]).all():
      puste += 1

print("stale    = " + str(stale))
print("kolka    = " + str(kolka))
print("krzyzyki = " + str(krzyzyki))
print("puste    = " + str(puste))
