# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:41:12 2020

@author: rheaa
note need CVXOPT ver >=1.2.3 installed (\w 1.2.0 'hermitian' requirement seemed to fuck things up)
"""
import numpy as np
import picos as pic
import cvxopt as cvx
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
 
### params ### 
E1 = [0,1,3,4] # list of energy eigenvalues of input Hamiltonian
E2 = [0,1]     # list of energy eigenvalues of output Hamiltonian
d1 = len(E1)
d2 = len(E2)
threshold = 0.9999 # set threshold for f(rho,sig) <=1
Cov  = True  # covariant constraint
Gp   =  False # Gibbs-preserving constraint
beta = 1.    # inverse temperature 
rho = pic.new_param('rho', np.ones((d1,d1))*(1/d1)) # input state rho = maximally coherent state

## Gibbs-state:
def g(x):
    return np.exp(-beta * x)
exp_array1 = np.array(list(map(g,E1)))   
exp_array2 = np.array(list(map(g,E2)))
gam1 = pic.diag(np.true_divide(exp_array1, np.sum(exp_array1)))
gam2 = pic.diag(np.true_divide(exp_array2, np.sum(exp_array2)))
g2_0 = gam2[0].value

cbit =   pic.new_param('zero',np.array([[0.5,0.5],[0.5,0.5]])) # |+X+|
zero = pic.new_param('zero', np.array([[1.,0.],[0.,0.]]))      # |0X0|
one  = pic.new_param('one', np.array([[0.,0.],[0.,1.]]))       # |1X1|
I2 = pic.new_param('I2', np.eye(2))                            # id_2
X = pic.new_param('X', np.array([[0.,1.],[1.,0.]]))            # pauli X
Z =  pic.new_param('Z', np.array([[1.,0.],[0.,-1.]]))          # pauli Z
Y = pic.new_param('Y', np.array([[0.,-1j],[1j,0.]]))           # pauli Y


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

def dephase_H(M):
    return dephase(M,E1,E2)

id_d2 =pic.new_param('id_d2',np.eye(d2)) 

def f(rho1,sig2,beta,cov=True,GP=True):
    ##   rho -> sigma under CPTP/ GP/ GPC/ Cov map? 
    ##   output \approx 1 --> yes, output < 1 --> no.  
    ##   dim(rho1) = dim(H1) = d1
    ##   dim(sig2) = dim(H2) = d2
    ##   default: qutrit-qubit CGP, equally-spaced energy levels, inv.temp=1

    ## def problem & variables ##
    p = pic.Problem()   
    X1 = p.add_variable('X1', (d1,d1), 'hermitian') 
    X2 = p.add_variable('X2', (d2,d2), 'hermitian')
    X3 = p.add_variable('X3', (d2,d2), 'hermitian')
    
    
    ### constraints ### 
    if GP == False: 
        p.add_constraint((sig2|X2)==1) 
        if cov == True:
            p.add_constraint(pic.kron(id_d2, X1) - dephase_H(pic.kron(X2, rho1)) >> 0. )         
        else:
            p.add_constraint(pic.kron(id_d2, X1) - pic.kron(X2, rho1)  >> 0. )     
    else:
        p.add_constraint((sig2|X2)+(gam2|X3)==1) 
        p.add_constraint(X3 >> 0)
        if cov == True:
            p.add_constraint(pic.kron(id_d2, X1) - dephase_H(pic.kron(X2, rho1)) - pic.kron(X3, gam1) >> 0. )         
        else:
            p.add_constraint(pic.kron(id_d2, X1) - pic.kron(X2, rho1) - pic.kron(X3, gam1) >> 0. )     
    p.add_constraint(X1 >> 0) 
    p.add_constraint(X2 >> 0) 
      
    ### objective fn & solve ### 
    p.set_objective('min', pic.trace(X1))
    p.solve(verbose = 0,solver = 'cvxopt')   
    return p.obj_value().real


def state(x,z):
    return 0.5 * (I2 + z*Z + x*X)


### Generate some data ### 

# compute range of z-values to check #
dz = 0.01
dx = dz
zrange=[]
z=-1.
x=0.
while z<=1.:
    if f(rho, state(x,z),beta,Cov,Gp) >= threshold: 
        zrange.append(z)
    z+=dz

zmin = min(zrange)
zmax = max(zrange)

xvals=[]
zvals=[]

z= zmin-2*dz
while z<=zmax+2*dz:
    x = np.sqrt(1-z**2)
    print(z)
    while x>=-dx:
        if f(rho, state(x,z),beta,Cov,Gp) >= threshold: 
            xvals.append(x)
            zvals.append(z)
            break 
        x-=dx
    z+=dz
    
### save data ###
np.savetxt('beta'+ str(beta) +'_E1_' + str(E1) + '_E2_' + str(E2) + '_GP' + str(Gp) +'_cov' +str(Cov) +'.out', (xvals,zvals,g2_0*np.ones(len(xvals))))

### plot results ###
xx = np.linspace(-1.0, 1.0, 100)
yy = np.linspace(-1.0, 1.0, 100)
XX, YY = np.meshgrid(xx,yy)
F = XX**2 + YY**2 - 1.0

fig = plt.figure()  
fig, ax1= plt.subplots(1, 1, figsize=(2,2))

ax1.contour(XX,YY,F,0, colors='k', linewidths=0.5)
#sc=ax1.scatter(xvals,zvals,0.1,c=np.ones(len(xvals)),alpha=1, cmap="coolwarm")
ax1.fill_betweenx(np.array(zvals), -np.array(xvals), np.array(xvals),facecolor='lightsteelblue', alpha=1)
if Gp == True:
    ax1.plot(0, 2*g2_0-1, marker='o',  markersize=5, color='red')
ax1.tick_params(labelsize=14)


# get rid of top/right border 
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# axis labels/ coords
ax1.set_xlabel(r'$X$',fontsize = 12)
ax1.set_ylabel(r'$Z$',rotation=0,fontsize = 12)
ax1.xaxis.set_label_coords(1.08, 0.05)
ax1.yaxis.set_label_coords(0.01, 1.03)

ax1.yaxis.set_ticks(np.array([-1,-0.5, 0,0.5, 1]))
ax1.set_yticklabels(np.array(["-1"," ","0"," ","1"]),fontsize=12)
ax1.xaxis.set_ticks(np.array([-1,-0.5,0,0.5, 1]))
ax1.set_xticklabels(np.array(["-1"," ","0"," ","1"]),fontsize=12)

axes = plt.gca()
axes.set_xlim([-1.25,1.25])
axes.set_ylim([-1.25,1.25])

# colourbar 
fig.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=0, hspace=0)

# axis labels/ coords
ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
ax1.yaxis.set_minor_locator(MultipleLocator(0.1))

ax1.tick_params(which ='both',direction='in', width=0.5)
ax1.tick_params(which='major', length=3)
ax1.tick_params(which='minor', length=1.5)

for tick in ax1.get_xticklabels():
    tick.set_fontname("Arial")
    tick.set_fontsize(10)
for tick in ax1.get_yticklabels():
    tick.set_fontname("Arial")
    tick.set_fontsize(10)

fig.savefig('beta'+ str(beta) +'_E1_' + str(E1) + '_E2_' + str(E2) + '_GP' + str(Gp) +'_cov' +str(Cov) +".png", dpi=800)


