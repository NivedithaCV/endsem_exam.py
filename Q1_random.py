import exam_utilities as ul
import matplotlib.pyplot as plt
import numpy as np
import math as m

walk =ul.random_walk(1000, 572, 16381, 1000)


number=[100,200,300,400,500,700,900,1000]
number=np.asarray(number)
for k in number:
    print(k)
    r=0
    m=[0 for i in range(len(number))]
    n=[0 for j in range(len(number))]
    walk0 =ul.random_walk(1000, 572, 16381, k)

    RMS=np.sqrt(np.mean(np.square(walk0[0])+np.square(walk0[1])))
    RMS_n=(k**0.5)

    m[r]=RMS
    n[r]=RMS_n
    r=r+1
plt.plot(m,n)
plt.xlabel("RMS")
plt.ylabel("sqrt(N)")
plt.show()

#a part
plt.plot(walk[0],walk[1])
plt.show()

#bplotted RMS vs sqrt(N) o show the relation
