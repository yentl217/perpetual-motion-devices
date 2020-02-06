# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:41:12 2020
@author: rheaa
note need CVXOPT ver >=1.2.3 installed (\w 1.2.0 'hermitian' requirement seemed to fuck things up)
"""

import numpy as np
#from scipy import optimize
#import scipy
import picos as pic #SDP solver
#import cvxopt as cvx
#from pprint import pprint
import matplotlib.pyplot as plt

plus = pic.new_param('plus', np.array([[.5,.5],[.5,.5]]))    # |+X+|
minus = pic.new_param('plus', np.array([[.5,-.5],[-.5,.5]])) # |-X-|
zero = pic.new_param('zero', np.array([[1.,0.],[0.,0.]]))    # |0X0|
one  = pic.new_param('one', np.array([[0.,0.],[0.,1.]]))     # |1X1|
I = pic.new_param('I',np.eye(2))                             # id
X = pic.new_param('X', np.array([[0.,1.],[1.,0.]]))          # pauli X
Z =  pic.new_param('Z', np.array([[1.,0.],[0.,-1.]]))        # pauli Z
Y = pic.new_param('X', np.array([[0.,-1j],[1j,0.]]))         # pauli Y

### def Gibbs state ### 
#previous labelled gamma; relabelled to gibbs to avoid confusion with Gamma map
b = 1  # inverse temp
E0 = 0
E1 = 1 # energy level spacing
def g(x):
    return np.exp(-b * x)
g0= g(E0)/(g(E0)+ g(E1))
g1= g(E1)/(g(E0)+ g(E1))
gibbs = pic.new_param('gibbs', np.array([[g0,0.],[0.,g1]])) 

### G-twirling map on two qubits###
#Note that we take the Hamiltonian of the reference system, H_R, is -H_B^T, where H_B is the Hamiltonian of the qubit transformed onto. 
Pi0 = pic.new_param('Pi0', pic.kron(zero,zero)+pic.kron(one,one)) # |00X00|+|11X11|
Pim1 = pic.new_param('Pim1', pic.kron(one,zero)) # |10X10|
Pi1 = pic.new_param('Pi1', pic.kron(zero,one)) # |01X01|
def dephase(M):
    return (Pi0*M*Pi0) + (Pi1*M*Pi1) + (Pim1*M*Pim1) 

### Gibbs-preserving covariant constraint fn ###
#This is the equation 123 version of the SDP
def Gamma_GPC(a0,a1,a2,a3,rho1,gibbs1):
    return pic.kron(I,a1) - dephase(pic.kron(a2,rho1)) - pic.kron(a3,gibbs1) - a0
### Gibbs-preserving constraint fn ###
def Gamma_GP(a0,a1,a2,a3,rho1,gibbs1):
    return pic.kron(I,a1) - pic.kron(a2,rho1) - pic.kron(a3,gibbs1) - a0

def f(rho1, sigma1):
    """ rho -> sigma under GPC map? Output \approx 1 --> yes, output<1 --> no.  
        dim(rho) = dim(sigma) =2 assumed """
        
    ## def problem & variables ##
    p = pic.Problem()  
    # this is the zeta vector
    X0 = p.add_variable('X0', (4,4), 'hermitian') #eta
    X1 = p.add_variable('X1', (2,2), 'hermitian') #Z
    X2 = p.add_variable('X2', (2,2), 'hermitian') #rho 
    X3 = p.add_variable('X3', (2,2), 'hermitian') #gibbs
    
    ## constraints ##
    paulis = [I,X,Y,Z] 
    # this is the constraint Gamma(zeta) = 0; must also be doable just by setting each matrix element to 0...
    for i in range(4):
        for k in range(4):
            basis_ik = pic.kron(paulis[i], paulis[k])
            p.add_constraint(pic.trace(basis_ik*Gamma_GPC(X0,X1,X2,X3,rho1,gibbs))==0)
   ### equivalently, can get rid of X0 variable and just use the following (removing dephase for GP): 
   #p.add_constraint(pic.kron(id_d, X1) - dephase(pic.kron(X2, rho)) - pic.kron(X3, gibbs) >> 0. ) but this is much slower.
   #constraint on the positivity of eta
    p.add_constraint(X0 >> 0) 
    p.add_constraint(X1 >> 0) 
    p.add_constraint(X2 >> 0) 
    p.add_constraint(X3 >> 0) 
   #Tr(sigma zeta) = 1
    p.add_constraint((sigma1|X2)+(gibbs|X3)==1) 
    ## objective fn & solve ## 
    p.set_objective('min', pic.trace(X1))
    p.solve(verbose = 0,solver = 'cvxopt')   
    return p.obj_value().real

def sig(theta, p):
    """ sig(theta,p) = p |thetaXtheta| + (1-p) |theta_barXtheta_bar|
        |theta> = cos theta/2 |0> + sin theta/2 |1>, <theta|theta_bar>=0"""
    #Why's this bit necessary?
    if abs(theta - np.pi)< 0.000000001:
        return 0.5 * (I - (2*p-1)*Z)
    else:
        return 0.5 * (I + np.cos(theta)*(2*p-1)*Z + np.sin(theta)*(2*p-1)*X)

### def initial state rho ### 
#Would be good to have this as program inputs
ttt= 0.5
prob= 1.0
rho = sig(ttt* np.pi, prob)

### Generate some data + comparison to L1 norm conditions (for qubits) ### 
x1 = []
z1=[]
l1=[]
x=[]
z=[]
n=[]

threshold = 0.999 # set threshold for f(rho,sig) <=1
h = 20  # (h+1)^2 sigma points will be checked
m = 300  # no. of nec/suff conditions to check for L1 norm calc for each sigma point - ~300 is best
for i in range(0,h+1):
    for j in range(0,h+1):
        t = np.pi * i/h
        p = j / h
        t_list=[]
        for q in range(1,m+1):
            PP = np.matrix(((q/m)*rho-(1-(q/m))*gibbs).value)
            QQ = np.matrix(((q/m)*sig(t,p)-(1-(q/m))*gibbs).value)
            L1_rho_gam = sum(np.linalg.svd(PP)[1])  #L1-norm of p1 rho - p2 gibbs
            L1_sig_gam = sum(np.linalg.svd(QQ)[1])  #L1-norm of p1 sig - p2 gibbs
            t_list.append(L1_rho_gam/L1_sig_gam)
        #should the L_1 norm threshold here be different frm the f(rho,sig) one?
        if all(i >= 0.999999 for i in t_list):
            x1.append(np.sin(t)*(2*p-1))
            z1.append(np.cos(t)*(2*p-1))
            l1.append(1.0) #what does this do? I suspect there is = if conversion is possible...   
        if f(rho, sig(t,p)) >= threshold:  
            x.append(np.sin(t)*(2*p-1))
            z.append(np.cos(t)*(2*p-1))
            n.append(f(rho, sig(t,p)))

### plot results ###

#sets up grid plot
xx = np.linspace(-1.0, 1.0, 100)
yy = np.linspace(-1.0, 1.0, 100)
XX, YY = np.meshgrid(xx,yy)
#This defines X-Z circle of Bloch sphere
F = XX**2 + YY**2 - 1.0

fig = plt.figure()  
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(5.8,2))

#Draws X-Z cirle of Bloch sphere
ax1.contour(XX,YY,F,0, colors='k', linewidths=0.5)
#Plots accessible sigma according to analytic solution
ax1.scatter(x1,z1,1,c=l1,alpha=1, cmap="coolwarm")
#plots initial state
ax1.plot(np.sin(np.pi*ttt)*(2*prob-1), np.cos(np.pi*ttt)*(2*prob-1), marker='o',  markersize=5, color='black')
#plots Gibbs state
ax1.plot(0, 2*g0-1, marker='o',  markersize=5, color='gray')

#Draws X-Z circle of Bloch sphere
ax2.contour(XX,YY,F,0, colors='k', linewidths=0.5)
#Plots accessible sigma according to SDP
sc=ax2.scatter(x,z,1,c=n,alpha=1, cmap="coolwarm")
#Plots initial state
ax2.plot(np.sin(np.pi*ttt)*(2*prob-1), np.cos(np.pi*ttt)*(2*prob-1), marker='o',  markersize=5, color='black')
#Plots Gibbs state
ax2.plot(0, 2*g0-1, marker='o',  markersize=5, color='gray')

# get rid of top/right border 
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# extra spacing in between plots 
fig.subplots_adjust(wspace=0.7)

# axis labels/ coords
ax1.set_xlabel(r'$X$',fontsize = 15)
ax1.set_ylabel(r'$Z$',rotation=0,fontsize = 15)
ax1.xaxis.set_label_coords(1.08, 0.05)
ax1.yaxis.set_label_coords(0.01, 1.03)

ax2.set_xlabel(r'$X$',fontsize = 15)
ax2.set_ylabel(r'$Z$',rotation=0,fontsize = 15)
ax2.xaxis.set_label_coords(1.08, 0.05)
ax2.yaxis.set_label_coords(0.01, 1.03)

axes = plt.gca()
axes.set_xlim([-1.25,1.25])
axes.set_ylim([-1.25,1.25])

# colourbar 
fig.subplots_adjust(right=0.79,bottom=0.2)
cbar_ax = fig.add_axes([0.85, 0.2, 0.023, 0.67])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.ax.set_title('$f( \\rho, \\sigma )$',fontsize=15)


fig.savefig("CGP3_rho_" + "theta_"+ str(ttt) + "_p_"+ str(prob)+ ".png", dpi=800)

"""
### single plot ###
fig=plt.figure(figsize=(7,5)) 
plt.scatter(x,z,1,c=n,alpha=4, cmap="coolwarm")
plt.colorbar();
plt.plot(2*q-1, 0, marker='o',  markersize=5, color='black')
plt.plot(0, 2*g1-1, marker='o',  markersize=5, color='black')
ax = plt.subplot(1, 1, 1)
fig.savefig("bloch.png", dpi=800)
np.savetxt('data_bloch.out', (x,z,n)) 
np.savetxt('data_bloch_l1.out', (x1,z1,l1)) 
fig=plt.figure(figsize=(7,5)) 
plt.scatter(x1,z1,1,c=l1,alpha=3, cmap="coolwarm")
plt.colorbar();
plt.plot(2*q-1, 0, marker='o',  markersize=5, color='black')
plt.plot(0, 2*g1-1, marker='o',  markersize=5, color='black')
ax = plt.subplot(1, 1, 1)
fig.savefig("bloch_l1.png", dpi=800)
"""
