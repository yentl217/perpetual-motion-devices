# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:41:12 2020

@author: rheaa
note need CVXOPT ver >=1.2.3 installed (\w 1.2.0 'hermitian' requirement seemed to fuck things up)
"""

#NOTE: at this time we are just looking at ladder systems, so Aberg's results apply.

import numpy as np
#This is the SDP solver
import picos as pic
import matplotlib.pyplot as plt
import sys

### setting up relevant Hamiltonians ###
#setting up qubit Hamiltonian
H2 = [0,1]
#setting up d-dim system Hamiltonian
dim = int(sys.argv[1]) #first argument is the dimension of the input system
Hd = list()
for i in range(dim):
    Hd.append(i)
#total Hamiltonian (not forgetting H_R = -H_A^T)
H_total = [x - H2[0] for x in Hd] + [x - H2[1] for x in Hd]

### setting up useful qubit states ###
#TODO: pic.new_param vs np.array?
plus2 = pic.new_param('plus2', np.array([[.5,.5],[.5,.5]]))    # |+X+|
minus2 = pic.new_param('minus2', np.array([[.5,-.5],[-.5,.5]])) # |-X-|
zero2 = pic.new_param('zero2', np.array([[1.,0.],[0.,0.]]))    # |0X0|
one2  = pic.new_param('one2', np.array([[0.,0.],[0.,1.]]))     # |1X1|
I2 = pic.new_param('I2',np.eye(2))                             # Identity
X = pic.new_param('X', np.array([[0.,1.],[1.,0.]]))          # pauli X
Z =  pic.new_param('Z', np.array([[1.,0.],[0.,-1.]]))        # pauli Z
Y = pic.new_param('Y', np.array([[0.,-1j],[1j,0.]]))         # pauli Y

### setting up qudit basis states inside a list ###
qudit = list()
diag = [0]*dim
for i in range(dim):
    diag[i] = 1
    qudit.append(pic.diag(diag))
    diag[i] = 0

### setting up projectors onto total energy eigenspaces of qudit-qubit system ###    
#initialise dictionary of projectors onto total energy eigenspaces, indexed by total energy (qubit goes first)
projectors = dict()
#initalise list of unique energies
uniq_e = list()

for i in range(len(H_total)):
    e = H_total[i]
    #value of test determines whether e_total was achieved with ground or excited state on qubit
    test = i - dim
    if e not in uniq_e:
    #begin constructing projector onto eigenspace with total energy e
        uniq_e.append(e)
        if test > -1: #involves excited state
            projectors[e]=pic.kron(one2,qudit[test])
        else: #involves ground state
            projectors[e]=pic.kron(zero2,qudit[i])
    else:
    #if projector unto state with total energy e already exists, expand that projector to include new state with the same total energy e
        if test > -1:
            projectors[e]=projectors[e]+pic.kron(one2,qudit[test])
        else:
            projectors[e]=projectors[e]+pic.kron(zero2,qudit[i])

### def initial state rho ### 
rho = pic.new_param('rho', np.ones((dim,dim))*(1/dim))

### def Gibbs states (system 1=qudit, sys 2=qubit) ### 
b  = .5   # inverse temp

def g(x):
    return np.exp(-b * x)
gammad = list()
boltzmann_Z = g(Hd[0])+g(Hd[1])+g(Hd[2])
for i in range(dim):
    gammad.append(g(Hd[i])/boltzmann_Z)
gibbsd = pic.diag(gammad)

g2_0= g(H2[0])/(g(H2[0])+ g(H2[1]))
g2_1= g(H2[1])/(g(H2[0])+ g(H2[1]))
gibbs2 = pic.new_param('gamma', np.array([[g2_0,0.],[0.,g2_1]])) 

### dephasing map in total energy eigenbasis ###
def dephase2(M):  #qudit to qutrit
    dephase_M = pic.diag([0]*dim*2)
    for e in projectors.keys():
        dephase_M = dephase_M + projectors[e]*M*projectors[e]
    return dephase_M

### define SDP program we want to run ###
def f(rho1,sig2,gam1,gam2,cov):
    # rho -> sigma under GP (set cov=False)/ GPC map? Output \approx 1 --> yes, output<1 --> no.  
    #    dim(rho1) = dim(gam1)
    #    dim(sig2) = dim(gam2) = d2
           
    ## def problem & variables ##
    #TODO ask Rhea where general solution to this problem is
    p = pic.Problem()   
    X1 = p.add_variable('X1', (dim,dim), 'hermitian') 
    X2 = p.add_variable('X2', (2,2), 'hermitian')
    X3 = p.add_variable('X3', (2,2), 'hermitian')

    p.add_constraint((sig2|X2)+(gam2|X3)==1) 
    if cov==True:
        #time-covariance constraint included
        p.add_constraint(pic.kron(I2, X1) - dephase2(pic.kron(X2, rho1)) - pic.kron(X3, gam1) >> 0. )         
    else:
        #time-covariance constraint not included
        p.add_constraint(pic.kron(I2, X1) - pic.kron(X2, rho1) - pic.kron(X3, gam1) >> 0. )  

    p.add_constraint(X1 >> 0) 
    p.add_constraint(X2 >> 0) 
    p.add_constraint(X3 >> 0)  
    
    ## objective fn & solve ## 
    p.set_objective('min', pic.trace(X1))
    p.solve(verbose = 0,solver = 'cvxopt')   
    return p.obj_value().real

### create qubit state in X-Z great circle of Bloch sphere from these parameters ###
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
        #NOTE the max coherent state is always tested!
        t = np.pi * i/h 
        p = j / h
        if f(rho, sig(t,p),gibbsd,gibbs2,True) >= threshold:  
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


fig.savefig("CGP" + sys.argv[1]+ "_qudit_to_qubit_threshold" + str(threshold) + "beta_" +str(b)+".png", dpi=800)

np.savetxt('qudit_to_qubit_threshold' + str(threshold) + 'beta_' +str(b)+'.out', (x1,z1,n1))
