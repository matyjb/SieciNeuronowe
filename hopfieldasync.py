import numpy as np

class HopfieldAsync:

  def __init__(self, w, g, b):
    self.w = w
    self.g = g
    self.b = b

  def calc(self, x):
    v = np.copy(x)
    n = len(v)
    u = np.zeros(n)

    for i in range(n):
      # tmp = 
      u[i] = sum([v[j]*self.w[i,j]-b[j] for j in range(n)])
      v[i] = self.g(u[i])
    
    if (v != x).all():
      return self.calc(v)
    else:
      return v
  
  def isStablePoint(self,x):
    xx = self.calc(x)
    return (xx == x).all()

def exec(x,w,b,g):
  h = HopfieldAsync(w, g, b)

  for i in range(len(x)):
    xx = x[i]
    if h.isStablePoint(xx):
      print(str(xx) + " => " + str(h.calc(xx)) + " punkt stały!")
    else:
      print(str(xx) + " => " + str(h.calc(xx)))


print("---przykład z notatek---")
w = np.array([[0,1],[1,0]])
x = np.array([[0,0],[0,1],[1,0],[1,1]])
b = np.array([0,0])
g = lambda x: 1 if x > 0 else 0
exec(x,w,b,g)

print("---projekt---")
c = 2/3
w = np.array([[0,-c,c],[-c,0,-c],[c,-c,0]])
x = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
b = np.array([0,0,0])
g = lambda x: 1 if x > 0 else -1
exec(x,w,b,g)