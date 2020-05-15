import numpy as np
from neuralNetwork import NeuralNetwork
from functions import Functions

learningStepsAmount = 60000

alpha=1
eta = 0.1
w = np.array([[[0.2, 0.1,  0.2],[0.1, 0.3, 0.1]],[[-0.2, -0.1,  0.2]]])
nn = NeuralNetwork(w, f=Functions.SINUS, alpha=alpha)

print(nn)

# xk = np.array([[1,0],[0,1],[0,0],[1,1]])
# dk = np.array([ [1] , [1] , [0] , [0] ])
# for i in range(learningStepsAmount):
#   nn.learn(xk[i % 4],dk[i % 4],eta)

# # #testing
# for i in range(4):
#   print(xk[i]," => ",nn.classify(xk[i]))

# print("##########################################")
# alpha=1
# f = lambda x: 1/(1+np.exp(-alpha*x))
# fprimfx = lambda x: x*(1-x)
# eta = 0.5
# nn = NeuralNetwork([2,2,1],FunctionsForNetwork.SINUS,alpha,[1,1])
# # wagi początkowe (nadpisanie randomowych)
# nn.network[0][0].w = np.array([0.86, -0.16,  0.28])
# nn.network[0][1].w = np.array([0.83, -0.51, -0.86])
# nn.network[1][0].w = np.array([0.04, -0.43,  0.48])

# # print(nn)
# xk = np.array([[0,0],[0,1],[1,0],[1,1]])
# dk = np.array([ [0] , [1] , [1] , [0] ])
# for i in range(learningStepsAmount):
# # for i in range(1):
#   nn.learn(xk[i % 4],dk[i % 4],eta)

# #testing
# for i in range(4):
#   print(xk[i]," => ",nn.classify(xk[i]))

