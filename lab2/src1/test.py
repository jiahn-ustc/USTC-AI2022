
from cvxpy import *
import numpy as np

a=Variable()
b =Variable()
constraints = [a>=1,a<=2,b>=1,b<=2]


# constraints += [sum_squares(alpha)==0]
const = a*b
obj = Minimize(const)
prob = Problem(obj,constraints)
prob.solve(verbose=True)
print(prob.status)
print(prob.value)

