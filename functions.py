import numpy as np

class Functions:
  STEP = ("Step",lambda x,alpha=None: 1 if x > 0 else 0,None)
  SINUS = ("Sinusoidalna",lambda x,alpha=1: 1/(1+np.exp(-alpha*x)),lambda x,alpha: Functions.SINUS[1](x,alpha)*(1-Functions.SINUS[1](x,alpha)))
  TANGENS = ("Tangensoidalna",lambda x,alpha=1: (1-np.exp(-alpha*x))/(1+np.exp(-alpha*x)),lambda x,alpha: alpha*(1-np.power(Functions.TANGENS[1](x),2))/2)