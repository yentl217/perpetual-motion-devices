import numpy as np
import picos as pic

#qubit Hamiltonian
H_1 = [0,1]
#higher-dim system Hamiltonian
H_2 = [0,1,2]
#dimension of higher-dim system
dim_2 = len(H_2)
#total Hamiltonian
H_total = [x - H_1[0] for x in H_2] + [x - H_1[1] for x in H_2]

#setting up qubit basis states
zero2 = pic.diag([1,0]) # |0X0|
one2  = pic.diag([0,1])# |1X1|
qubit = [zero2,one2]

#Setting up qutrit basis states
zero3 = pic.diag([1,0,0])
one3 = pic.diag([0,1,0])
two3 = pic.diag([0,0,1])
qutrit = [zero3,one3,two3]

#initialise dictionary of projectors onto total energy eigenspaces, indexed by total energy (qubit goes first)
projectors = dict()
#initalise list of unique energies
uniq_e = list()

for i in range(len(H_total)):
    e = H_total[i]
    #value of test determines whether e_total was achieved with ground or excited state on qubit
    test = i - dim_2
    if e not in uniq_e:
    #begin constructing projector onto eigenspace with total energy e
        uniq_e.append(e)
        if test > -1: #involves excited state
            projectors[e]=pic.kron(qubit[1],qutrit[test])
        else: #involves ground state
            projectors[e]=pic.kron(,qubit[0],qutrit[i])
    else:
    #if projector unto state with total energy e already exists, expand that projector to include new state with the same total energy e
        if test > -1:
            projectors[e]=projectors[e]+pic.kron(,qubit[1],qutrit[test])
        else:
            projectors[e]=projectors[e]+pic.kron(qubit[0],qutrit[i])

for key,value in projectors.items():
    print(key,value)
