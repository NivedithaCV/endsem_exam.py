import exam_utilities as ul
import numpy as np
import matplotlib.pyplot as plt
# x = (-3, -2, -1, 0, 1, 2, 3)
# y = (7.5, 3, 0.5, 1, 3, 6, 14)
# sigma = (1,1,1,1,1,1,1)
# c=ul.polynomial_fitting(x,y,sigma)
# print(c)
h=0.1
n=20/0.1
b=0.0008*5000
k=0.0008
x=np.arange(0,2+0.1,h)
print(x)
t=np.arange(0,4+0.008,k)
boundaryConditions=[0,0]
initialCondition= 20*abs(np.sin(np.pi*x))

T=ul.pde_explicit(x,t,h,k,boundaryConditions,initialCondition)
#print(T.shape)
p=[0,10,20,50,100,200,500]
for i in p:
    plt.plot(x,T[:,i],label=i)
    plt.legend()
plt.xlabel("position")
plt.ylabel("Temperature")
plt.show()
