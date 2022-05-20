
import numpy as np
import math
import matplotlib.pyplot as plt



def cholesky_decomp(A):
    #decomposing the matrix
    n = len(A)
    A=np.array(A)
    shp= A.shape()
    M = np.zeros(shp)


    for i in range(n):
        for j in range(i+1):
            off_dia = np.dot(M[i,:j], M[j,:j])
            if (i == j):

                #diagonal
                d=max(A[i,i] - off_dia, 0)
                M[i, j] = np.sqrt(d)

            else:

                diff=A[i,j] - off_dia
                M[i,j] = (1.0 / M[j,j]) * diff
    return M

def cholesky(L):
    L = cholesky_decomp(np.array(L))
    Z = [[0 for x in range(len(L))] for y in range(len(L))]
    n = len(L)


    # forward subtitution
    for i in range(n):
        Z[i][i] = 1/L[i][i]
    for i in range(1, n):
        for j in range(i):
            sum = 0
            for k in range(j, i):
                sum += L[i][k]*Z[k][j]
            Z[i][j] = - sum / L[i][i]


    #Backward substitution
    x = [[0 for i in range(n)] for j in range(n)]
    L_t = np.transpose(L)
    for i in range(n,-1,-1):
        x[i][i] = (Z[i][i]) / (L[i][i])
    for i in range(n - 1,-1,-1):
        for j in reversed(range((i + 1), n)):
            sum = 0
            for k in range(i + 1, j + 1):
                sum += ((L_t[i][k]) * (x[k][j]))

            x[i][j] = (Z[j][i] - sum) / (L[i][i])
    return x

#______________________________________________________________________________________________
#store L and U in A
def L_Udec(A,c_A):
    for j in range(c_A):
        for i in range(len(A)):

            #diagonal
            if i==j:
                sum=0
                for u in range(i):
                    sum=sum+A[i][u]*A[u][i]
                A[i][i]=A[i][i]-sum

                #elements of upper triangle
            if i<j:
                sum=0
                for k in range(i):
                    sum=sum+A[i][k]*A[k][j]
                A[i][j]=A[i][j]-sum

                #elements of lower triangle
            if i>j:
                sum=0
                for z in range(j):
                    sum=sum+A[i][z]*A[z][j]
                A[i][j]=(A[i][j]-sum)/A[j][j]
    return(A)

def forw_backw(A,B,col):
    for k in range(col):
        r=len(A)

        #forward substitution
        Y=[[0] for y in range(r)]
        for i in range(r):
            sum=0
            for k in range(i):
                sum=sum+A[i][k]*Y[k][0]
            Y[i][0]=B[i][0]-sum
        print("matrix Y",Y)

        #backward substitution
        X=[[0] for w in range(r)]
        for l in range(r-1,-1,-1):
            sum=0
            for m in range(l+1,r):
                sum=sum+A[l][m]*X[m][0]
            X[l][0]=(Y[l][0]-sum)/A[l][l]
    print("solution matrix is",X)

    return(X)


#_______________________________________________________________________________________________
def rungeKutta(x0, y0, x, h):
    # Count number of iterations using step size or
    # step height h
    n = (int)((x - x0)/h)
    # Iterate for number of iterations
    y = y0
    for i in range(1, n + 1):
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = h * dydx(x0, y)
        k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1)
        k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2)
        k4 = h * dydx(x0 + h, y + k3)

        # Update next value of y
        y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)

        # Update next value of x
        x0 = x0 + h
    return y
##____________________________________________________________________________________________________________________________________________________
#inputs taken = the function containg equations, value of x, value of t and iteration number
def coupled_ODE_RK4(f,u0,t0,tf,n):
    t = np.linspace(t0, tf, n+1)
    u = np.array((n+1)*[u0])
    h = t[1]-t[0]

    for i in range(n):
        k1 = h * f(u[i], t[i])
        k2 = h * f(u[i] + 0.5 * k1, t[i] + 0.5*h)
        k3 = h * f(u[i] + 0.5 * k2, t[i] + 0.5*h)
        k4 = h * f(u[i] + k3, t[i] + h)
        u[i+1] = u[i] + (k1 + 2*(k2 + k3 ) + k4) / 6
    return u, t


#_______________________________________________________________________________________________________________________________________________

def polynomial_fitting(x,y,error):
    e_2= np.square(error)
    n_=np.ones(len(error))
    n=np.sum(np.divide(n_,e_2))
    sx=np.sum(np.divide(x,e_2))
    sy=np.sum(np.divide(y,e_2))
    x_2 =np.square(x)
    x_3 =np.power(x,3)
    x_4 =np.power(x,4)
    yx  =np.multiply(y,x)
    yx_2=np.multiply(y,x_2)
    sx_2=np.sum(np.divide(x_2,e_2))
    sx_3=np.sum(np.divide(x_3,e_2))
    sx_4=np.sum(np.divide(x_4,e_2))
    syx =np.sum(np.divide(yx,e_2))
    syx_2=np.sum(np.divide(yx_2,e_2))
    M= [[n,sx,sx_2],[sx,sx_2,sx_3],[sx_2,sx_3,sx_4]]
    B=[[sy],[syx],[syx_2]]
    c=len(B)
    d=len(M)
    L_Udec(M,d)
    fit=forw_backw(M,B,c)
    return(fit)


def polynomial_fitting_4(x, y, error):
    X = [];Y = [];B = []
    #para = order + 1  # no of parameters
    # ar_x = np.array(x)
    # ar_y = np.array(y)
    # ar_Dy = np.array(Dy)
    for i in range(5):
        X.append([])
        for j in range(5):
            X[i].append(np.sum(((x ** (i + j))) / (error ** 2)))
    for i in range(5):
        B.append([])
        for j in range(5):
            if (i == j):
                B[i].append(1)
            else:
                B[i].append(0)
    for i in range(5):
        Y.append([])
        Y[i].append(np.sum(np.multiply((x ** i), y) / error ** 2))

    c=len(Y)
    d=len(X)
    L_Udec(X,d)
    fit=forw_backw(X,Y,c)

    # L_Udec(M,d)
    # fit=forw_backw(M,B,c)
    #inv = linalg.inv(np.array(X))
    # parameter = np.dot(inv, Y)
    return fit

#__________________________________________________________________________________________________________

def chebyshevfns(x, order):
    if order == 0:
        return 1
    elif order == 1:
        return 2*x - 1
    elif order == 2:
        return 8*x**2 - 8*x + 1
    elif order == 3:
        return 32*x**3 - 48*x**2 + 18*x - 1

#Defining the function for chebyshev fit
def fitw_chebyshev(xvals, yvals, order):
    n = len(xvals)
    para = order + 1
    A = np.zeros((para, para))
    b = np.zeros(para)

    for i in range(para):
        for j in range(para):
            total = 0
            for k in range(n):
                total += chebyshevfns(xvals[k], j) * chebyshevfns(xvals[k], i)

            A[i, j] = total

    for i in range(para):
        total = 0
        for k in range(n):
            total += chebyshevfns(xvals[k], i) * yvals[k]

        b[i] = total
    c=len(b)
    d=len(A)
    L_Udec(A,d)
    fit=forw_backw(A,b,c)
    # LU = LUdecomp(A, b)
    # F = fwrdsub(LU,b)
    # para = bkwdsub(LU, F)
    return para,A

#______________________________________________________________________________
def RK4coupled(x, y, tf, n, f, g):
    X = []
    Y = []
    t = np.linspace(t0, tf, n+1)
    for i in range(n):
        k1 = h * f(x, y, t)
        l1 = h * g(x, y, t)
        k2 = h * f(x + h / 2, y, t + l1 / 2)
        l2 = h * g(x + h / 2, y + k1 / 2, t + l1 / 2)
        k3 = h * f(x + h / 2, y, t + l2 / 2)
        l3 = h * g(x + h / 2, y + k2 / 2, t + l2 / 2)
        k4 = h * f(x + h, y, t + l3)
        l4 = h * g(x + h, y + k3, t + l3)
        y = y + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + 1 / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
        x = x + h
        X.append(x)
        Y.append(y)
        T.append(t)
    return X, Y

#_________________________________________________________________________________
def pde_explicit(x,t,h,k,boundaryConditions,initialCondition):

    n=len(x)
    m=len(t)
    T=np.zeros((n,m))
    T[0,:] = boundaryConditions[0]
    T[2,:]= boundaryConditions[1]
    T[:,0] = initialCondition
    factor =k/h**2
    for j in range(1,m):
        for i in range(1,n-1):
            T[i,j] = factor*T[i-1,j-1] + (1-2*factor)*T[i,j-1] + factor*T[i+1,j-1]
    return(T)

#______________________________________________________________________________________________________
def rms(X, Y):
    n = len(X); m = len(Y)
    X = np.array(X)
    Y = np.array(Y)
    sumx_square = 0.0; sumy_square = 0.0; sumz_square = 0.0

    sumx_square = np.sum(np.square(X))
    sumy_square = np.sum(np.square(Y))

    rms_x = np.sqrt(np.add(sumx_square,sumy_square) /( n))
    # rms_y = math.sqrt(sumy_square / m)
    # rms_z = math.sqrt(sumz_square / l)

    return rms_x
#________________________________________________________________________________________________________
def rand_LCG(seed, n,a, m):
    rand = np.zeros(n)
    rand[0] = seed
    for i in range(1, n):
        rand[i] = (a*rand[i-1]) % m
    print(rand.round(3))

def randomwalk( steps,a,m):
    X=[],Y=[]
    x0 = 0; y0 = 0
    phi = rand_LCG(3,steps, a,m)
    theta = rand_LCG(7, steps, a,m)
    # phi = np.random.random(steps)
    # theta = np.random.random(steps)

    for i in range(steps):
        phi0 = 2*np.pi*phi[i]
        theta0 = np.pi * theta[i]
        k0 = np.cos(phi0) * np.sin(theta0)
        k1 = np.sin(phi0) * np.sin(theta0)
        r = 1
        x0 = x0 + r*k0; y0 = y0 + r*k1
        X.append(x0)
        Y.append(y0)
    return X, Y
#___________________________________________________________________________________________________________



def rand_MLC(seed, a, m, N):
    x = [seed]
    for i in range(N):
        x.append((a*x[i])%m)
    x = [i/m for i in x]
    return x



def random_walk(seed, a, m, N):
    origin=[0,0]
    x=0;y=0
    xstep = [origin[0]]
    ystep= [origin[1]]
    numbers = rand_MLC(seed, a, m, N-1)

    theta = [numbers[i] * 2*np.pi for i in range(len(numbers))]

    for i in range(N):
        x += np.cos(theta[i])
        y += np.sin(theta[i])

        xstep.append(x)
        ystep.append(y)

    return xstep, ystep



## generate a random walk of 200 steps

#__________________________________________________________________________________________________________
# def fitwth_orthonormal_Legendre(x,oder):
#     if oder==0:
#         return(1)
#     if oder==1:
#         return(x)
#     if oder==2:
#         return(0.5*(3*np.square(x)-1)
#     if oder==3:
#         return(0.5* (5*np.power(x,3)-(3*x)))
#     if oder==4:
#         return(1/8)



# def rand_walk():
# # Probability to move up or down
#     # prob = [0.05, 0.95]
#     #
#     # # statically defining the starting position
#     # start = 2
#     # positions = 2
#
#     # creating the random points
#     rr = rand_LCG(3,10,572,16381)
# #     downp = rr < prob[0]
# #     upp = rr > prob[1]
# #
# #
# #     for idownp, iupp in zip(downp, upp):
# #     	down = idownp and positions[-1] > 1
# #     	up = iupp and positions[-1] < 4
# #     	positions.append(positions[-1] - down + up)
# #
# # # plotting down the graph of the random walk in 1D
# #     plt.plot(positions)
# #     plt.show()
