# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:41:12 2020

@author: rheaa
note need CVXOPT ver >=1.2.3 installed (\w 1.2.0 'hermitian' requirement seemed to fuck things up)
"""
import numpy as np
import picos as pic
#import cvxopt as cvx
import matplotlib.pyplot as plt
import math

zero = pic.new_param('zero', np.array([[1.,0.],[0.,0.]]))    # |0X0|
one  = pic.new_param('one', np.array([[0.,0.],[0.,1.]]))     # |1X1|
I2 = pic.new_param('I2', np.eye(2))                          # id_2
X = pic.new_param('X', np.array([[0.,1.],[1.,0.]]))          # pauli X
Z =  pic.new_param('Z', np.array([[1.,0.],[0.,-1.]]))        # pauli Z
Y = pic.new_param('Y', np.array([[0.,-1j],[1j,0.]]))         # pauli Y

def dephase(M,H1,H2):
    ## dephase matrix M w.r.t. H_tot = id_d2 X H1 + HR X id_d1,
    ## HR = -H2
    ## H1, H2 lists of ordered energy eigenvalues of arb dim: low to high.
    d1 = len(H1)
    d2 = len(H2)
    id_d1 = pic.new_param('id_d1', np.eye(d1))
    id_d2 = pic.new_param('id_d2', np.eye(d2)) 
    dim = d1*d2
    HR = np.negative(H2)
    H_tot = np.diag(np.matrix((pic.kron(id_d2,pic.diag(H1))+pic.kron(pic.diag(HR),id_d1)).value))
    index = [np.argwhere(i==H_tot) for i in np.unique(H_tot)]
    dephased = pic.diag(np.zeros(dim))
    PVMs=[]
    for i in range(len(index)):
        a2 = np.zeros(dim)
        a2[index[i]]=1
        a2 = np.array(a2)
        PVMs.append(pic.diag(a2))
    for i in range(len(index)):
        dephased += PVMs[i] * M * PVMs[i]
    return dephased



def f(rho1,sig2,beta=1,H1=[0,1,2],H2=[0,1],cov=True,GP=True):
    ##   rho -> sigma under CPTP/ GP/ GPC/ Cov map? 
    ##   output \approx 1 --> yes, output < 1 --> no.  
    ##   dim(rho1) = dim(H1) = d1
    ##   dim(sig2) = dim(H2) = d2
    ##   default: qutrit-qubit CPG, equally-spaced energy levels, inv.temp=1
    d1 = len(H1)
    d2 = len(H2)
    id_d2 =pic.new_param('id_d2',np.eye(d2)) 

    ## def problem & variables ##
    p = pic.Problem()   
    X1 = p.add_variable('X1', (d1,d1), 'hermitian') 
    X2 = p.add_variable('X2', (d2,d2), 'hermitian')
    X3 = p.add_variable('X3', (d2,d2), 'hermitian')
    
    ## Gibbs-state:
    def g(x):
        return np.exp(-beta * x)
    exp_array1 = np.array(list(map(g,H1)))   
    exp_array2 = np.array(list(map(g,H2)))
    gam1 = pic.diag(np.true_divide(exp_array1, np.sum(exp_array1)))
    gam2 = pic.diag(np.true_divide(exp_array2, np.sum(exp_array2)))
    #g2_0 = gam2[0].value
    
    ### constraints ### 
    if GP == False: 
        p.add_constraint((sig2|X2)==1) 
        if cov == True:
            p.add_constraint(pic.kron(id_d2, X1) - dephase(pic.kron(X2, rho1),H1,H2) >> 0. )         
        else:
            p.add_constraint(pic.kron(id_d2, X1) - pic.kron(X2, rho1)  >> 0. )     
    else:
        p.add_constraint((sig2|X2)+(gam2|X3)==1) 
        p.add_constraint(X3 >> 0)
        if cov == True:
            p.add_constraint(pic.kron(id_d2, X1) - dephase(pic.kron(X2, rho1),H1,H2) - pic.kron(X3, gam1) >> 0. )         
        else:
            p.add_constraint(pic.kron(id_d2, X1) - pic.kron(X2, rho1) - pic.kron(X3, gam1) >> 0. )     
    p.add_constraint(X1 >> 0) 
    p.add_constraint(X2 >> 0) 
      
    ### objective fn & solve ### 
    p.set_objective('min', pic.trace(X1))
    p.solve(verbose = 0,solver = 'cvxopt')   
    return p.obj_value().real

def qubit_XZ(theta, p):
    ## qubit_XZ(t,p) = p |tXt| + (1-p) |t_barXt_bar|
    ## |t> = cos t/2 |0> + sin t/2 |1>, <t|t_bar>=0
    return 0.5 * (I2 + np.cos(theta)*(2*p-1)*Z + np.sin(theta)*(2*p-1)*X)

### Generate some data ### 
threshold = 0.9999 # set threshold for f(rho,sig) <=1
h = 200 # (h+1)^2 sigma points will be checked

x1=[]
z1=[]
n1=[]
fitting=[]

### def initial state rho ### 
#prob=0.75
#theta=0.50 
#rho =  qubit_XZ(theta*np.pi, prob)
rho = pic.new_param('rho', np.ones((3,3))*(1/3))

E1 = [0,1,5]
E2 = [0,1]
Cov = True
Gp  =  False
beta = 0.5
for i in range(0,h+1):
    print(i)
    for j in range(0,h+1):
        t = np.pi * i/h
        p = j / h
        if f(rho, qubit_XZ(t,p),beta,E1,E2,Cov,Gp) >= threshold: 
            x1.append(np.sin(t)*(2*p-1))
            r_z = np.cos(t)*(2*p-1)
            z1.append(r_z)
            n1.append(1.) #n1.append(f(rho, sig(t,p),gamma_1))
            if r_z < -1/3:
                fitting.append(2*math.sqrt((1+r_z)/6))
            elif r_z < 1/3:
                fitting.append(2/3)
            else:
                fitting.append(2*math.sqrt((1-r_z)/6))

### plot results ###
xx = np.linspace(-1.0, 1.0, 100)
yy = np.linspace(-1.0, 1.0, 100)
XX, YY = np.meshgrid(xx,yy)
F = XX**2 + YY**2 - 1.0

fig = plt.figure()  
fig, ax1= plt.subplots(1, 1, figsize=(4,4))

ax1.contour(XX,YY,F,0, colors='k', linewidths=0.5)
sc=ax1.scatter(x1,z1,0.1,c=n1,alpha=1, cmap="coolwarm")
ax1.scatter(fitting,z1,0.1,c='black',alpha=1)
#ax1.plot(np.sin(np.pi*theta)*(2*prob-1), np.cos(np.pi*theta)*(2*prob-1), marker='o',  markersize=5, color='black')

## Gibbs-state:
def g(x):
    return np.exp(-beta * x) 
exp_array2 = np.array(list(map(g,E2)))
gam2 = pic.diag(np.true_divide(exp_array2, np.sum(exp_array2)))
g2_0 = gam2[0].value
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


fig.savefig('dim_' + str(len(E1)) +'_to_' +str(len(E2))+ '_H1_' + str(E1) + '_threshold_' + str(threshold) + '_beta_' +str(beta)+ '_cov_' +str(Cov) + '_GP_' +str(Gp) +'.png', dpi=800)

np.savetxt('dim_' + str(len(E1)) +'_to_' +str(len(E2))+ '_H1_' + str(E1) +  '_threshold_' + str(threshold) + '_beta_' +str(beta)+ '_cov_' +str(Cov) + '_GP_' +str(Gp) + '.out', (x1,z1,n1))
