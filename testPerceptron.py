import numpy as np
from perceptron import Perceptron

## przyklady z zeszytu (z zajec)
bias = 0.5
w0 = np.array([bias,0,1])
dk = [0,0,1,0]
xk = np.array([[1,0,0],[1,1,0],[1,1,1],[1,0,1]])

p0 = Perceptron(w0)
p0.learn(xk, dk, debug=True)

print("----algorytm zgrupowany----")
bias = 1
w0 = np.array([bias,0,1])
dk = [0,1,0,0]
xk = np.array([[1,0,0],[1,1,0],[1,0,1],[1,1,1]])

p1 = Perceptron(w0)
p1.learn(xk, dk, grouped=True, debug=True)
## ####
