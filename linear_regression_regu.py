'''
----------------------------------------------------------------------------------
linear regression python code using pandas and numpy library with regularization
with only two feature that is x1 and x2
output is y
hypothesis function(predicted/model function) is h
parameters/coeffcients theta0,theta1,theta2
regularizatiion parameter LAMBDA
-------------------------prepared by Devaraj Nadiger------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
LEARNING_RATE=0.000001
ITERATIONS=100
LAMBDA=100

#hypothesis function
def hyp(X,theta):
    hyp=theta.T.dot(X)
    return hyp
#regularization term to add in cost function
def reg2(theta,m,g,l):
    theta=theta[1:]
    r=((np.sum((theta)**2))*l)/(2*m)
    return r
#regularization term to add in gradient function
def reg(theta,m,g,l,a):
    theta=theta[1:]
    o=(np.sum(theta))*(l)/m
    return o

#cost function
def cost_fun(X,Y,theta,m,g,l):
    j=np.sum((hyp(X,theta)-Y)**2)/(2*m)
    j=j+reg2(theta,m,g,l)
    return j

#gradient descent algorithm
def grad_des(X,Y,theta,a,itrns,m,theta_len,L):
    count=[0]*itrns
    d=[0]*itrns
    for i in range(itrns):
        h=hyp(X,theta)
        grad=X.dot(h-Y)/m
        #storing theta1 value to draw graph
        d[i]=theta[1]
        cost=cost_fun(X,Y,theta,m,theta_len,L)
        theta=theta-a*grad
        
        #calling regularization function and substarcting it from theta1 and theta2
        z=reg(theta,m,theta_len,L,a)
        theta[1]=theta[1]-z
        theta[2]=theta[2]-z
        
        count[i]=cost
    return theta,count,d

#main function
def main():
    data=pd.read_csv('channing.csv') #reading csv file
    x1=data["entry"].values
    x1=(x1-np.mean(x1))/(max(x1)-min(x1))
    x1d=x1
    x2=data["exit"].values
    x2d=x2
    x2=(x2-np.mean(x2))/(max(x2)-min(x2))
    y=data["time"].values 
    m=len(x1)
    x0=np.ones(m)
    X=np.array([x0,x1,x2])
    Y=np.array(y)
    theta=np.array([1,1,1])
    theta_len=len(theta)
    a=LEARNING_RATE
    itrns=ITERATIONS
    L=LAMBDA
    j=list(range(0,itrns,1))
    
    H=hyp(X,theta)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1d, x2d, y, c='r')
    ax.scatter(x1d, x2d, H, c='b')
    ax.set_xlabel('x1 feature')
    ax.set_ylabel('x2 feature')
    ax.set_zlabel('initial predicted(blue) vs actual(red) output')
    plt.show()

    new_theta,count,d=grad_des(X,Y,theta,a,itrns,m,theta_len,L)

    
    print('cost_function=',count)
    print('\n')
    
    plt.scatter(d,count,color='red')
    plt.plot(d,count)
    plt.xlabel('parameter')
    plt.ylabel('cost_function')
    plt.title('cost function curve ')
    plt.show()

    plt.scatter(j,d,color='red')
    plt.plot(j,d)
    plt.xlabel('iteration')
    plt.ylabel('parameter')
    plt.title('parameter vs iteration ')
    plt.show()
    
    plt.scatter(j,count,color='red')
    plt.plot(j,count)
    plt.xlabel('iterations')
    plt.ylabel('cost_function')
    plt.title('cost function curve ')
    plt.show()
    
    print('final_theta=',new_theta)
    
    pr=hyp(X,new_theta) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1d, x2d, y, c='r')
    ax.scatter(x1d, x2d, pr, c='b')
    ax.set_xlabel('x1 feature')
    ax.set_ylabel('x2 feature')
    ax.set_zlabel('predicted(blue) vs actual(red) output')
    plt.show()
    
if __name__=="__main__":
    main()

