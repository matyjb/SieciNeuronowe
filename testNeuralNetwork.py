import numpy as np
from neuralNetwork import NeuralNetwork
from functions import Functions

print("###########################################")
print("###############--TEST  XOR--###############")
print("###########################################")

# print("\n# Z INTERNETU #\n")
# iterations = 15000
# alpha=1
# eta = 0.1
# #warstwa 0
# w00 = [0.2, 0.1,  0.2]
# w01 = [0.1, 0.3, 0.1]
# # warstwa 1 (output)
# w10 = [-0.2, -0.1,  0.2]

# w = np.array([[w00,w01],[w10]])
# x0s = np.array([1,1])
# nn = NeuralNetwork(w, layersx0s=x0s, f=Functions.SINUS, alpha=alpha)

# print(nn)

# xk = np.array([[1,0],[0,1],[0,0],[1,1]])
# dk = np.array([ [1] , [1] , [0] , [0] ])
# nn.learn(xk,dk,eta,iterations)

# # #testing
# for i in range(4):
#   print(xk[i]," => ",nn.classify(xk[i]))

print("\n# PROJEKT #\n")
iterations = 5000
alpha=1
eta = 0.5
#warstwa 0
w00 = [0.86, -0.16,  0.28]
w01 = [0.83, -0.51, -0.86]
# warstwa 1 (output)
w10 = [0.04, -0.43,  0.48]

w = np.array([[w00,w01],[w10]])
x0s = np.array([1,1])
nn = NeuralNetwork(w, layersx0s=x0s, f=Functions.SINUS, alpha=alpha)

print("--przed uczeniem--")
print(nn)

xk = np.array([[1,0],[0,1],[0,0],[1,1]])
dk = np.array([ [1] , [1] , [0] , [0] ])
nn.learn(xk,dk,eta,iterations)

print("--po uczeniu--")
print(nn)
# #testing
for i in range(4):
  print(xk[i]," => ",nn.classify(xk[i]))

