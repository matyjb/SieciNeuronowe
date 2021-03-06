import numpy as np
import itertools
from hopfield import *
from functions import Functions


# funkcja pomocnicza do wyswietlenia wynikow zadania
def exec(x, w, b=None, mode=Modes.ASYNC, f=Functions.STEP):
  h = Hopfield(w, b=b, mode=mode, f=f)
  print(h)
  for i in range(len(x)):
    xIn = x[i]
    (xOut,t) = h.classify(xIn,debug=True)
    print(str(xIn) + " => " + str(xOut) + " t=" + str(t))


# generuje wszystkie kombinacje wektora o n długości
# i dwóch mozliwych wartościach
def genxs(n, v1, v2):
  return np.array([list(i) for i in itertools.product([v1, v2], repeat=n)])

print("###########################################")
print("#############--TEST HOPFIELD--#############")
print("###########################################")

# print("\n# Z NOTATEK #\n")
# w = np.array([[0,1],[1,0]])
# x = genxs(2,0,1)
# b = np.array([0,0])
# print("sync)\n")
# exec(x,w,b,Modes.SYNC)
# print("\nasync)")
# exec(x,w,b,Modes.ASYNC)

# print("---przykład z ZadRozwiazaniaHopfield.pdf str 9---") #inna step function !!!!
# w = np.array([[-1,3/4],[3/4,0]])
# x = genxs(2,0,1)
# b = np.array([0,0])
# exec(x,w,b,Modes.ASYNC)

# print("---przykład z ZadRozwiazaniaHopfield.pdf str 13---") #inna step function !!!!
# w = np.array([[0,1,-1],[1,0,1],[-1,1,0]])
# x = genxs(3,0,1)
# b = np.array([0,0,0])
# exec(x,w,b,Modes.SYNC)

print("\n# PROJEKT #\n")
c = 2/3
w = np.array([[0,-c,c],[-c,0,-c],[c,-c,0]])
x = genxs(3,-1,1)
b = np.array([0,0,0])
print("sync)\n")
exec(x,w,b,Modes.SYNC,Functions.STEP1)
print("\nasync)\n")
exec(x,w,b,Modes.ASYNC,Functions.STEP1)

# print("---ZADANIE 3---")
# w = np.array([[0,1],[-1,0]])
# x = np.array([[1,-1],[-1,1]])
# b = np.array([0,0])
# f = lambda x: 1 if x > 0 else -1
# exec(x,f,w,b)


# print("---przykładzik z internetu z tic tac toe---")
# h = HopfieldAsync(f=f)
# # trzy obrazy 3x3 (kółko, krzyzyk, puste):
# # C C C | C B C | B B B
# # C B C | B C B | B B B
# # C C C | C B C | B B B
# # C = 1 B = -1
# xk = np.array([[1,1,1,1,-1,1,1,1,1],[1,-1,1,-1,1,-1,1,-1,1],[-1,-1,-1,-1,-1,-1,-1,-1,-1]])
# h.learnHebb(xk)
# # print("nowe wagi:")
# # print(h.w)
# # print("---klasyfikacja kolka i krzyzyk---")
# x = genxs(9,-1,1)
# stale = 0
# kolka = 0
# krzyzyki = 0
# puste = 0
# for i in range(len(x)):
#     xIn = x[i]
#     (xOut,t) = h.classify(xIn)
#     # if t == 0:
#     #   print(str(xIn) + ";" + str(xOut) + ";" + str(t) + ";1")
#     # else:
#     #   print(str(xIn) + ";" + str(xOut) + ";" + str(t) + ";0")
#     if t == 0:
#       stale += 1
#       print(xOut)
#     if (xOut == xk[0]).all():
#       kolka += 1
#     if (xOut == xk[1]).all():
#       krzyzyki += 1
#     if (xOut == xk[2]).all():
#       puste += 1

# print("stale    = " + str(stale))
# print("kolka    = " + str(kolka))
# print("krzyzyki = " + str(krzyzyki))
# print("puste    = " + str(puste))
