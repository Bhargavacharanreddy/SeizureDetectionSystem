from random import randrange, seed
from WalshHadamard import *
from numpy import linalg as LA

seed(2015)
import scipy.io
import numpy as np

#data = scipy.io.loadmat("S001R02_edfm.mat")

#for i in data:

		#np.savetxt((i+".csv"),data[i],delimiter=',')

#--------------------excel testing area--------------------#

from excel import *
path="test.xls"
X=open_file(path)
#----------------------------------------------------------#
#X=[ 10,14,11,-6,5,4,4,-1,7,10,4,6,6,7,7,3 ]


Y=[]
print("X = ")
print(X)
print("Slow WHT(X) = ")
print(WHT(X)[0])

print("Fast WHT(X) = ")

print(FWHT(X))
sums=0
for i in FWHT(X):
    sums=sums+i
sums=sums/16


diff=0
for i in FWHT(X):
    diff=diff+((sums-i)*(sums-i))
    
diff=sqrt(diff/16)
print("ARithmatic Mean of these values is "+str(sums))
print("Standard Deviation of these values is "+str(diff))

for i in FWHT(X):
    Y.append((i-sums)/diff)
    
print("Normalized values are :::")

print(Y)

from bispectrumd import *
#A = matrix( [[1,2,3,4],[11,12,13,14],[21,22,23,25],[40,42,44,46]])

#A=NP.matrix("1 2 3 4;11 12 13 14;21 22 23 24;40 42 44 46")

#print(A)
#qpc = sio.loadmat('rec_1m.mat')

#print qpc['val']


#sample = FWHT(qpc['val'])

#print sample

#dbic = bispectrumd(qpc['val'], 15,0,2000,0)

