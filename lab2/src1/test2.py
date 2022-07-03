from cvxpy import *
import numpy as np
a = np.zeros((2,2))
b = np.zeros((2,2))
a[0][0]=1
a[0][1]=2
a[1][0]=3
a[1][1]=4
b[0][0]=5
b[0][1]=6
b[1][0]=7
b[1][1]=8
print(a*b*0.5)