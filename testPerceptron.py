import numpy as np
from perceptron import Perceptron
from functions import Functions

print("###########################################")
print("############--TEST PERCEPTRON--############")
print("###########################################")

# print("\n# Z ZESZYTU ZAJĘĆ #\n")
# ## przyklady z zeszytu (z zajec)
# w0 = np.array([0.5,0,1])
# dk = [0,0,1,0]
# bias = 1
# xk = np.array([[bias,0,0],[bias,1,0],[bias,1,1],[bias,0,1]])

# p0 = Perceptron(w0)
# print(p0)
# p0.learn(xk, dk, debug=True)

# print("----algorytm zgrupowany----")
# w0 = np.array([1,0,1])
# dk = [0,1,0,0]
# bias = 1
# xk = np.array([[bias,0,0],[bias,1,0],[bias,0,1],[bias,1,1]])

# p1 = Perceptron(w0)
# print(p1)
# p1.learn(xk, dk, grouped=True, debug=True)
## ####
print("\n# PROJEKT #\n")
print("a)")
print("(i)\n")
eta = 1
w = np.array([1,1,1])
xk = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
dk = [0,0,1,0]
f = Functions.STEP

p = Perceptron(w, f)
p.learn(xk, dk, eta, debug=True)
print("\n(ii)\n")
eta = 0.1
w = np.array([-0.12,0.4,0.65])
xk = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
dk = [0,0,1,0]
f = Functions.STEP

p = Perceptron(w, f)
p.learn(xk, dk, eta, debug=True)
print("\nb)\n")
print("(i)\n")
eta = 1
w = np.array([1,1,1])
xk = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
dk = [0,0,1,0]
f = Functions.STEP

p = Perceptron(w, f)
p.learn(xk, dk, eta, grouped=True, debug=True)
print("\n(ii)\n")
eta = 0.1
w = np.array([-0.12,0.4,0.65])
xk = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
dk = [0,0,1,0]
f = Functions.STEP

p = Perceptron(w, f)
p.learn(xk, dk, eta, grouped=True, debug=True)


