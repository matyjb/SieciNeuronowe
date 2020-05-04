import numpy as np
from Perceptron import Perceptron

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
p0.learn(xk,dk, debug=True)

print("----algorytm zgrupowany----")

p1 = Perceptron(f,w0,r)
p1.learn(xk,dk,grouped=True,debug=True)

##

w0 = np.array([-0.12,0.4,0.65])
r = 0.1
f = lambda x: 1 if x > 0 else 0
dk = [0,0,1,0]
xk = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])

p0 = Perceptron(f,w0,r)
p0.learn(xk,dk, debug=True)

print("----algorytm zgrupowany----")

p1 = Perceptron(f,w0,r)
p1.learn(xk,dk,grouped=True,debug=True)
