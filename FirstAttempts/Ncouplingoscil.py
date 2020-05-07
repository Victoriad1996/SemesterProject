from numpy import loadtxt
from pylab import figure, plot, xlabel, grid, legend, title, savefig
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

def vectorfield(w,p) :
    """
    Defines the differential equations for the coupled spring-mass system.

    Arguments:
        w :  vector of the state variables:
                  w = [x1,y1,x2,y2]
        t :  time
        p :  vector of the parameters:
                  p = [m1,m2,k1,k2,L1,L2,b1,b2]
    """
    m=p[0]
    k=p[1]
    L=p[2]
    b=p[3]

    #Take the position and velocity values from w.
    x = []
    y = []
    for i in range(np.int(len(w) / 2)):
        x.append(w[2 * i])
        y.append(w[2 * i + 1])
    # Create f = (x1',y1',x2',y2'):
    n=len(x)-1
    f=[]
    f.append(y[0])
    f.append((-b[0]*y[0]+(k[1]+k[0])*(L[0]-x[0])-k[1]*(L[1]-x[1]))/m[0])
    for i in range(1, n):
        f.append(y[i])
        f.append((-b[i]*y[i]-k[i]*(L[i-1]-x[i-1]-L[i]+x[i])+k[i+1]*(L[i]-x[i]-L[i+1]+x[i+1]))/m[i])
    f.append(y[n])
    f.append((-b[n]*y[n]+(k[n+1]+k[n])*(L[n]-x[n])-k[n]*(L[n-1]-x[n-1]))/m[n])

    return f



# Parameter values

#Number of masses (useful for initialization)
n=5
# Masses:
m=3*np.ones(n)

# Spring constants
k=10*np.ones(n+1)

# Natural lengths
L=[i for i in range(n+1)]

# Friction coefficients
b=0.5*np.ones(n)

# Initial conditions
# x are the initial positions; y are the initial velocities

#x positions
x=[i for i in range(n)]

#y random velocity
y= np.random.normal(0, 1, n)
#y=[(-1)**i for i in range(n)]
# ODE solver parameters
stoptime = 20.0
numpoints = 50000

# Create the time samples for the output of the ODE solver.
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

#Initial conditions
w=[]
for i in range(len(x)):
    w.append(x[i])
    w.append(y[i])
#Additional parameters
p=m,k,L,b


#Approximates solution with Euler Explicit and Heun Method.
#Approximates the error by the difference between the two approximations
def ode_sol_err(f, IC, stoptime, numpoints,p):
    h=stoptime/(numpoints-1)
    y=np.zeros((numpoints,np.size(IC)))
    y_heun=np.zeros((numpoints,np.size(IC)))
    err=np.zeros((numpoints,1))
    y[0]=IC
    y_heun[0]=IC
    for i in range(1,numpoints):
        f_eval=f(y[i-1],p)
        delta=[h*z for z in f_eval]
        y[i]=[sum(z) for z in zip(delta,y[i-1])]
        f_eval2=f(y[i],p)
        delta=[sum(z) for z in zip(f_eval,f_eval2)]
        delta=[0.5*h*z for z in delta]
        y_heun[i]=[sum(z) for z in zip(y_heun[i-1],delta)]
        diff=[sum(z) for z in zip(-y[i],y_heun[i])]
        err[i]=np.linalg.norm(diff)
    return y_heun,y, err;

#Print and save the estimations
def print_and_save_sol(file_name,y,t,numpoints,plot_name):
    with open(file_name,'w') as f:
        n_masses=len(y[0])
        for j in range(numpoints):
            t1=t[j]
            print(t1,end=' ',file=f)
            for i in range(n_masses-1):
                print(y[j][i],end=' ',file=f)
            print(y[j][n_masses -1],end='\n',file=f)

    #t, x1, xy, x2, y2, x3, y3 = loadtxt('two_springs.dat', unpack=True)
    t= loadtxt(file_name, unpack=True)[0]
    z=[]
    for i in range (n):
        z.append(loadtxt(file_name, unpack=True)[2*i+1])
        #w.append(loadtxt('two_springs.dat', unpack=True)[2*(i+1)])
    #z.append(loadtxt('two_springs.dat', unpack=True)[2*i+1])
    figure(1, figsize=(6, 4.5))

    xlabel('t')
    grid(True)
    lw = 1

    for i in range(len(z)):
        plot(t, z[i], linewidth=lw)

    #legend((r'$x_1$', r'$x_2$',r'$x_3$'), prop=FontProperties(size=16))
    title(plot_name)
    savefig('two_springs3.png', dpi=100)

    plt.show()
    return z


y_heun,y,err=ode_sol_err(vectorfield,w,stoptime,numpoints,p)




#Save and print the solution of the Euler Estimate
file_name='Euler_estimate.dat'
plot_name='Mass positions for the Coupled \n Mass System with Euler Estimate'
z=print_and_save_sol(file_name,y,t,numpoints,plot_name)


#Save the error estimate
with open('error_est.dat','w') as f:
    for i in range(len(err)):
        print(err[i],end=' ',file=f)

#Plot the error estimate
lw = 1
plt.plot(t,err,linewidth=lw)
xlabel(t)
title('Estimation of the error')
savefig('Error_estimate.png', dpi=100)
plt.show()
