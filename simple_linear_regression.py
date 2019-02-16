'''
----------------------------------------------------------------------------------
simple linear regression python code without using pandas and scikit library
with only one feature that is x
output is y
hypothesis function(predicted/model function) is h
parameters/coefficients theta0 and theta1
-------------------------prepared by Devaraj Nadiger------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
LEARNING_RATE=0.01
ITERATIONS=10

#input x and output y with hypothesis function h.
x=np.array([1,2,4,3,5])
y=np.array([1,3,3,2,5])
h=np.zeros(5)
m=len(x)
theta0=1
theta1=1
a=LEARNING_RATE
count=0

while(count<ITERATIONS):
    cost=[0]*ITERATIONS
   
    for i in range(0,m):
        h[i]=theta0+theta1*x[i]
    print('hypothesis function,h(theta)=',h)
    for i in range(0,m):
        s=np.sum((h[i]-y[i])**2)
        j=s/(2*m)
        #storing each iterated cost function
        cost[i]=j
        #calculation of parameters
        theta0=theta0-(a/m)*np.sum(h[i]-y[i])
        theta1=theta1-(a/m)*np.sum((h[i]-y[i])*x[i])
        
    print('cost function,j(theta)=',j)
    print('theta0=',theta0)
    print('theta1=',theta1)       
    print('\n')
    
    count=count+1
   
k=list(range(0,ITERATIONS,1))
plt.scatter(k,cost,color='r')
plt.plot(k,cost)
plt.title('cost function vs iterations ')
plt.xlabel('iterations')
plt.ylabel('cost function')
plt.show()


plt.scatter(x,y,color='r')
plt.plot(x,h)
plt.title('predicted(curve) and actual(dots) output ')
plt.xlabel('data set')
plt.ylabel('output')
plt.show()
