# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:41:12 2020

@author: rheaa
note need CVXOPT ver >=1.2.3 installed (\w 1.2.0 'hermitian' requirement seemed to fuck things up)
"""
import numpy as np
#from scipy import optimize
#import scipy
import picos as pic
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
b = 1  # inverse temp
E0 = 0
E1 = 1 # energy level spacing
def g(x):
    return np.exp(-b * x)
g0= g(E0)/(g(E0)+ g(E1))
g1= g(E1)/(g(E0)+ g(E1))
gamma = pic.new_param('gamma', np.array([[g0,0.],[0.,g1]])) 

### dephasing map ###
def dephase(M):
    return pic.diag(pic.diag_vect(M)) 

### Gibbs-preserving covariant constraint fn ###
def Gamma_GPC(a0,a1,a2,a3,rho1,gamma1):
    return pic.kron(I,a1) - dephase(pic.kron(a2,rho1)) - pic.kron(a3,gamma1) - a0
### Gibbs-preserving constraint fn ###
def Gamma_GP(a0,a1,a2,a3,rho1,gamma1):
    return pic.kron(I,a1) - pic.kron(a2,rho1) - pic.kron(a3,gamma1) - a0

def f(rho1, sigma1):
    """ rho -> sigma under GPC map? Output \approx 1 --> yes, output<1 --> no.  
        dim(rho) = dim(sigma) =2 assumed """
        
    ## def problem & variables ##
    p = pic.Problem()  
    X0 = p.add_variable('X0', (4,4), 'hermitian') 
    X1 = p.add_variable('X1', (2,2), 'hermitian') 
    X2 = p.add_variable('X2', (2,2), 'hermitian')
    X3 = p.add_variable('X3', (2,2), 'hermitian')
    ## constraints ##
    paulis = [I,X,Y,Z]    
    for i in range(4):
        for k in range(4):
            basis_ik = pic.kron(paulis[i], paulis[k])
            p.add_constraint(pic.trace(basis_ik*Gamma_GPC(X0,X1,X2,X3,rho1,gamma))==0)
   ### equivalently, can get rid of X0 variable and just use the following (removing dephase for GP): 
   #p.add_constraint(pic.kron(id_d, X1) - dephase(pic.kron(X2, rho)) - pic.kron(X3, gamma) >> 0. )         
    p.add_constraint(X0 >> 0) 
    p.add_constraint(X1 >> 0) 
    p.add_constraint(X2 >> 0) 
    p.add_constraint(X3 >> 0)     
    p.add_constraint((sigma1|X2)+(gamma|X3)==1) 
    ## objective fn & solve ## 
    p.set_objective('min', pic.trace(X1))
    p.solve(verbose = 0,solver = 'cvxopt')   
    return p.obj_value().real

def sig(thetaa, ppp):
    """ sig(t,p) = p |tXt| + (1-p) |t_barXt_bar|
        |t> = cos t/2 |0> + sin t/2 |1>, <t|t_bar>=0"""
    if abs(thetaa - np.pi)< 0.000000001:
        return 0.5 * (I - (2*ppp-1)*Z)
    else:
        return 0.5 * (I + np.cos(thetaa)*(2*ppp-1)*Z + np.sin(thetaa)*(2*ppp-1)*X)

### def initial state rho ### 
ttt= 0.5
prob= 0.8
rho = sig(ttt* np.pi, prob)

### Generate some data + comparison to L1 norm conditions### 
x1 = []
z1=[]
l1=[]
x=[]
z=[]
n=[]

threshold = 0.999 # set threshold for f(rho,sig) <=1
h = 20  # (h+1)^2 sigma points will be checked
m = 30  # no. of nec/suff conditions to check for L1 norm calc for each sigma point - ~300 is best
for i in range(0,h+1):
    for j in range(0,h+1):
        t = np.pi * i/h
        p = j / h
        t_list=[]
        for q in range(1,m+1):
            PP = np.matrix(((q/m)*rho-(1-(q/m))*gamma).value)
            QQ = np.matrix(((q/m)*sig(t,p)-(1-(q/m))*gamma).value)
            L1_rho_gam = sum(np.linalg.svd(PP)[1])  #L1-norm of p1 rho - p2 gamma
            L1_sig_gam = sum(np.linalg.svd(QQ)[1])  #L1-norm of p1 sig - p2 gamma
            t_list.append(L1_rho_gam/L1_sig_gam)
        if all(i >= 0.999999 for i in t_list):
            x1.append(np.sin(t)*(2*p-1))
            z1.append(np.cos(t)*(2*p-1))
            l1.append(1.0)    
        if f(rho, sig(t,p)) >= threshold:  
            x.append(np.sin(t)*(2*p-1))
            z.append(np.cos(t)*(2*p-1))
            n.append(f(rho, sig(t,p)))

### plot results ###

xx = np.linspace(-1.0, 1.0, 100)
yy = np.linspace(-1.0, 1.0, 100)
XX, YY = np.meshgrid(xx,yy)
F = XX**2 + YY**2 - 1.0

fig = plt.figure()  
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(5.8,2))

ax1.contour(XX,YY,F,0, colors='k', linewidths=0.5)
ax1.scatter(x1,z1,1,c=l1,alpha=1, cmap="coolwarm")
ax1.plot(np.sin(np.pi*ttt)*(2*prob-1), np.cos(np.pi*ttt)*(2*prob-1), marker='o',  markersize=5, color='black')
ax1.plot(0, 2*g0-1, marker='o',  markersize=5, color='gray')

ax2.contour(XX,YY,F,0, colors='k', linewidths=0.5)
sc=ax2.scatter(x,z,1,c=n,alpha=1, cmap="coolwarm")
ax2.plot(np.sin(np.pi*ttt)*(2*prob-1), np.cos(np.pi*ttt)*(2*prob-1), marker='o',  markersize=5, color='black')
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