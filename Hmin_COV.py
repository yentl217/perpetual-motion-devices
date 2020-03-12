# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 19:41:12 2020

@author: rheaa
note need CVXOPT ver >=1.2.3 installed (\w 1.2.0 'hermitian' requirement seemed to fuck things up)
"""
import numpy as np
import picos as pic
import cvxopt as cvx
from numpy import loadtxt
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
plus =   pic.new_param('plus',np.array([[0.5,0.5],[0.5,0.5]]))    # |+X+|
minus=   pic.new_param('minus',np.array([[0.5,-0.5],[-0.5,0.5]])) # |-X-|
zero = pic.new_param('zero', np.array([[1.,0.],[0.,0.]]))         # |0X0|
one  = pic.new_param('one', np.array([[0.,0.],[0.,1.]]))          # |1X1|
I2 = pic.new_param('I2', np.eye(2))                               # id_2
X = pic.new_param('X', np.array([[0.,1.],[1.,0.]]))               # pauli X
Z =  pic.new_param('Z', np.array([[1.,0.],[0.,-1.]]))             # pauli Z
Y = pic.new_param('Y', np.array([[0.,-1j],[1j,0.]]))              # pauli Y

def state(x,z):
    return 0.5 * (I2 + z*Z + x*X)
def state_t(t,p):
    return 0.5 * (I2 + (2*p-1)*np.cos(np.pi*t)*Z +(2*p-1)*np.sin(np.pi*t)*X)

### params ### 
E1 = [0,1]       # list of energy eigenvalues of input Hamiltonian
E2 = [0,1]       # list of energy eigenvalues of output Hamiltonian
d1 = len(E1)
d2 = len(E2)
threshold =  0.99999999 # set threshold for f(rho,sig), <=1
x0 = 0.5 #0.25
z0 = 0.0 #0.5
rho = state(x0,z0) # def initial state rho
theta_ref = 1/3    # reference state angle to Z-axis/pi
p_ref = 1.            
reference = state_t(theta_ref,p_ref) # def reference state
xref=pic.trace(reference*X).value
zref=pic.trace(reference*Z).value

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

def Hmin(rho,ref):
    ### compute Hmin(A|B)_rho, covariant map, ref=reference state. ###
    # def problem & variables #
    p = pic.Problem()   
    X1 = p.add_variable('X1', (d1,d1), 'hermitian') 
    p.add_constraint(pic.kron(id_d2, X1) - dephase_H(pic.kron(ref, rho)) >> 0.)
    p.add_constraint(X1 >> 0) 
    # objective fn & solve # 
    p.set_objective('min', pic.trace(X1))
    p.solve(verbose = 0,solver = 'cvxopt') 
    return -np.log2(p.obj_value().real)

def ratio(rho,ref,x,z):
    return Hmin(state(x,z),ref)/Hmin(rho,ref)

### Generate some data ### 
#(Note: assumes accessible state space is symmetric about Z, this seems to be valid here.)
dz = 0.01
dx = dz
zrange=[]
z= -1.
x=  0.

x_list=[]
z_list=[]
hmin_list=[]

z= -1.
while z<=1.:
    x = np.sqrt(1-z**2)
    while x>=-dx:
        if  ratio(rho,reference,x,z) >= threshold:
            x_list.append(x)
            z_list.append(z)
            break 
        x-=dx
    z+=dz

#import SDP data for comparison
xvals, zvals= loadtxt("filename.out",float)

### plot results ###
xx = np.linspace(-1.0, 1.0, 100)
yy = np.linspace(-1.0, 1.0, 100)
XX, YY = np.meshgrid(xx,yy)
F = XX**2 + YY**2 - 1.0

fig = plt.figure()  
fig, ax1= plt.subplots(1, 1, figsize=(2,2))

ax1.contour(XX,YY,F,0, colors='k', linewidths=0.5)
#sc=ax1.scatter(x_list,z_list,0.1,c=hmin_list,alpha=1, cmap="coolwarm")
ax1.fill_betweenx(np.array(z_list), -np.array(x_list), np.array(x_list),facecolor='lightsteelblue', alpha=1)
ax1.plot(xvals, zvals,ls='-', lw=0.5, color='black')
ax1.plot(-np.array(xvals), np.array(zvals), ls='-', lw=0.5, color='black')
ax1.plot(xref, zref, marker='o',  markersize=5, color='blue')
ax1.plot(x0, z0, marker='o',  markersize=5, color='black')
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

fig.savefig("ref_theta_" +str(theta_ref) +"_p_" + str(p_ref) + ".png", dpi=800)

