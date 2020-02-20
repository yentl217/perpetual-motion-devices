# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:41:12 2020
@author: rheaa
note need CVXOPT ver >=1.2.3 installed (\w 1.2.0 'hermitian' requirement seemed to fuck things up)
"""

#TODO: add dimension as input
#NOTE: at this time we are just looking at ladder systems, so Aberg's results apply.

import numpy as np
#This is the SDP solver
import picos as pic
import matplotlib.pyplot as plt

#setting up some useful qubit states
plus2 = pic.new_param('plus2', np.array([[.5,.5],[.5,.5]]))    # |+X+|
minus2 = pic.new_param('minus2', np.array([[.5,-.5],[-.5,.5]])) # |-X-|
zero2 = pic.new_param('zero2', np.array([[1.,0.],[0.,0.]]))    # |0X0|
one2  = pic.new_param('one2', np.array([[0.,0.],[0.,1.]]))     # |1X1|
I2 = pic.new_param('I2',np.eye(2))                             # Identity
X = pic.new_param('X', np.array([[0.,1.],[1.,0.]]))          # pauli X
Z =  pic.new_param('Z', np.array([[1.,0.],[0.,-1.]]))        # pauli Z
Y = pic.new_param('Y', np.array([[0.,-1j],[1j,0.]]))         # pauli Y

#Setting up qutrit basis states
zero3 = pic.diag([1,0,0])
one3 = pic.diag([0,1,0])
two3 = pic.diag([0,0,1])

#Qutrit-qubit energy eigenspace projectors
#TODO: find/write Python routine to automate this process
Pi0 = pic.new_param('Pi0', pic.kron(zero2,zero3)+pic.kron(one2,one3)) # |00X00|+|11X11|
Pi1 =  pic.new_param('Pi1', pic.kron(zero2,one3)+pic.kron(one2,two3)) # |01X01|+|12X12|
Pim1 = pic.new_param('Pim1', pic.kron(one2,zero3)) # |10X10|
Pi2 = pic.new_param('Pi2', pic.kron(zero2,two3)) # |02X02|

### def initial state rho ### 
rho = pic.new_param('rho', np.ones((3,3))*(1/3))
#rho = pic.diag([1/3,1/3,1/3])

### def Gibbs state's (system 1=qutrit, sys 2=qubit) ### 
b  = .5   # inverse temp
E0 = 0
E1 = 1.  # energy level spacing
E2 = 2.

def g(x):
    return np.exp(-b * x)
g1_0= g(E0)/(g(E0)+g(E1)+g(E2))
g1_1= g(E1)/(g(E0)+g(E1)+g(E2))
g1_2= g(E2)/(g(E0)+g(E1)+g(E2))
gamma1 = pic.diag([g1_0,g1_1,g1_2])

g2_0= g(E0)/(g(E0)+ g(E1))
g2_1= g(E1)/(g(E0)+ g(E1))
gamma2 = pic.new_param('gamma', np.array([[g2_0,0.],[0.,g2_1]])) 

### dephasing map in energy eigenbasis ###
def dephase2(M):  # 2 qubits
    return (Pi0*M*Pi0) + (Pi1*M*Pi1) + (Pi2*M*Pi2) + (Pim1*M*Pim1) 

def f(rho1,sig2,gam1,gam2,d2,cov=True):
    # rho -> sigma under GP (set cov=False)/ GPC map? Output \approx 1 --> yes, output<1 --> no.  
    #    dim(rho1) = dim(gam1)
    #    dim(sig2) = dim(gam2) = d2
           
    id_d2 =pic.new_param('id_d2',np.eye(d2)) 
    ## def problem & variables ##
    p = pic.Problem()   
    X1 = p.add_variable('X1', (3,3), 'hermitian') 
    X2 = p.add_variable('X2', (d2,d2), 'hermitian')
    X3 = p.add_variable('X3', (d2,d2), 'hermitian')

    p.add_constraint((sig2|X2)+(gam2|X3)==1) 
    if cov==True:
        p.add_constraint(pic.kron(id_d2, X1) - dephase2(pic.kron(X2, rho1)) - pic.kron(X3, gam1) >> 0. )         
    else:
        p.add_constraint(pic.kron(id_d2, X1) - pic.kron(X2, rho1) - pic.kron(X3, gam1) >> 0. )  

    p.add_constraint(X1 >> 0) 
    p.add_constraint(X2 >> 0) 
    p.add_constraint(X3 >> 0)  
    
    ## objective fn & solve ## 
    p.set_objective('min', pic.trace(X1))
    p.solve(verbose = 0,solver = 'cvxopt')   
    return p.obj_value().real

def sig(theta, p):
    # sig(t,p) = p |tXt| + (1-p) |t_barXt_bar|
    #    |t> = cos t/2 |0> + sin t/2 |1>, <t|t_bar>=0
    return 0.5 * (I2 + np.cos(theta)*(2*p-1)*Z + np.sin(theta)*(2*p-1)*X)


###Generate some data### 
threshold = 0.99999 # set threshold for f(rho,sig) <=1
h = 20 # (h+1)^2 sigma points will be checked

x1=[]
z1=[]
n1=[]


for i in range(0,h+1):
    for j in range(0,h+1):
        t = np.pi * i/h
        p = j / h
        if f(rho, sig(t,p),gamma1,gamma2,2) >= threshold:  
            x1.append(np.sin(t)*(2*p-1))
            z1.append(np.cos(t)*(2*p-1))
            n1.append(1.) #n1.append(f(rho, sig(t,p),gamma_1))


### plot results ###
xx = np.linspace(-1.0, 1.0, 100)
yy = np.linspace(-1.0, 1.0, 100)
XX, YY = np.meshgrid(xx,yy)
F = XX**2 + YY**2 - 1.0

fig = plt.figure()  
fig, ax1= plt.subplots(1, 1, figsize=(2,2))

ax1.contour(XX,YY,F,0, colors='k', linewidths=0.5)
sc=ax1.scatter(x1,z1,0.1,c=n1,alpha=1, cmap="coolwarm")
ax1.plot(0, 2*g2_0-1, marker='o',  markersize=5, color='red')
ax1.tick_params(labelsize=14)


# get rid of top/right border 
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)


# extra spacing in between plots 
#fig.subplots_adjust(wspace=0.7)

# axis labels/ coords
ax1.set_xlabel(r'$X$',fontsize = 16)
ax1.set_ylabel(r'$Z$',rotation=0,fontsize = 16)
ax1.xaxis.set_label_coords(1.08, 0.05)
ax1.yaxis.set_label_coords(0.01, 1.03)

ax1.yaxis.set_ticks(np.array([-1,-0.5, 0,0.5, 1]))
ax1.set_yticklabels(np.array(["-1"," ","0"," ","1"]),fontsize=14)
ax1.xaxis.set_ticks(np.array([-1,-0.5, 0,0.5, 1]))
ax1.set_xticklabels(np.array(["-1"," ","0"," ","1"]),fontsize=14)



axes = plt.gca()
axes.set_xlim([-1.25,1.25])
axes.set_ylim([-1.25,1.25])


# colourbar 
fig.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=0, hspace=0)
#cbar_ax = fig.add_axes([0.6, 0.2, 0.05, 0.67])
#cbar = fig.colorbar(sc, cax=cbar_ax)
#cbar.ax.set_title('$f( \\rho, \\sigma )$',fontsize=15)


fig.savefig("CGP3_qutrit_to_qubit_threshold" + str(threshold) + "beta_" +str(b)+".png", dpi=800)

np.savetxt('qutrit_to_qubit_threshold' + str(threshold) + 'beta_' +str(b)+'.out', (x1,z1,n1))


"""
xx = np.linspace(-1.0, 1.0, 100)
yy = np.linspace(-1.0, 1.0, 100)
XX, YY = np.meshgrid(xx,yy)
F = XX**2 + YY**2 - 1.0
### single plot ###
fig=plt.figure(figsize=(7,5)) 
ax = plt.subplot(1, 1, 1)
ax.contour(XX,YY,F,0, colors='k', linewidths=0.5)
sc=ax.scatter(x,z,1,c=n,alpha=1, cmap="coolwarm")
ax.plot(np.sin(np.pi*theta)*(2*prob-1), np.cos(np.pi*theta)*(2*prob-1), marker='o',  markersize=5, color='black')
ax.plot(0, 2*g0-1, marker='o',  markersize=5, color='gray')
fig.savefig("CGP3_rho_" + "theta_"+ str(theta) + "_p_"+ str(prob)+ ".png", dpi=800)
"""
