import numpy as np
import itertools


# klasa dynamiczna bo pewnie w projekt cz3 bedzie nauczanie tej sieci
# bo to w cz2 to mozna za pomocą funkcji zrobić
# (przekazać do classify() te parametry: w, g, b)
class HopfieldAsync:

  def __init__(self, w, g, b):
    self.w = w
    self.g = g
    self.b = b

  # wylicza ?klasyfikacje? zadanego wektora i numer (czas) iteracji
  def classify(self, x, t=0):
    v = np.copy(x)
    n = len(v)
    u = np.zeros(n)

    for i in range(n):
      u[i] = sum([v[j]*self.w[i,j]-b[j] for j in range(n)])
      v[i] = self.g(u[i])
    
    if not (v == x).all():
      return self.classify(v, t+1)
    else:
      return (v,t)

# funkcja pomocnicza do wyswietlenia wynikow zadania
def exec(x,w,b,g):
  h = HopfieldAsync(w, g, b)
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

print("---przykład z notatek---")
w = np.array([[0,1],[1,0]])
x = genxs(2,0,1)
b = np.array([0,0])
g = lambda x: 1 if x > 0 else 0
exec(x,w,b,g)

# to psuje
# print("---przykład z ZadRozwiazaniaHopfield.pdf str 9---")
# w = np.array([[-1,3/4],[3/4,0]])
# x = genxs(2,0,1)
# b = np.array([0,0])
# g = lambda x: 1 if x >= 0 else 0
# exec(x,w,b,g)

print("---przykład z ZadRozwiazaniaHopfield.pdf str 13---")
w = np.array([[0,1,-1],[1,0,1],[-1,1,0]])
x = genxs(3,0,1)
b = np.array([0,0,0])
g = lambda x: 1 if x > 0 else 0
exec(x,w,b,g)

print("---projekt---")
c = 2/3
w = np.array([[0,-c,c],[-c,0,-c],[c,-c,0]])
x = genxs(3,-1,1)
b = np.array([0,0,0])
g = lambda x: 1 if x > 0 else -1
exec(x,w,b,g)

