import numpy as np
from perceptron import Perceptron

## przyklady z zeszytu (z zajec)
w0 = np.array([0.5,0,1])
dk = [0,0,1,0]
bias = 1
xk = np.array([[bias,0,0],[bias,1,0],[bias,1,1],[bias,0,1]])

p0 = Perceptron(w0)
print(p0)
p0.learn(xk, dk, debug=True)

print("----algorytm zgrupowany----")
w0 = np.array([1,0,1])
dk = [0,1,0,0]
bias = 1
xk = np.array([[bias,0,0],[bias,1,0],[bias,0,1],[bias,1,1]])

p1 = Perceptron(w0)
print(p1)
p1.learn(xk, dk, grouped=True, debug=True)
## ####
