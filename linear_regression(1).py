'''
-----------------------------------------------------------------------
linear regression python code using pandas and numpy library
with only one feature that is x1
output is y
hypothesis function(predicted/model function) is h
parameters/coefficients theta0 and theta1
-----------------prepared by Devaraj Nadiger---------------------------
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
LEARNING_RATE=0.01
ITERATIONS=100

#hypothesis function
def hyp(X,theta):
    hyp=theta.T.dot(X)
    return hyp

#cost function
def cost_fun(X,Y,theta,m):
    j=np.sum((theta.T.dot(X)-Y)**2)/(2*m)
    return j

#gradient descent algorithm
def grad_des(X,Y,theta,a,itrns,m):
    cost_count=[0]*itrns
    t=[0]*itrns
    for i in range(itrns):
        h=hyp(X,theta)
        #print(h)
        loss=h-Y
        grad=X.dot(loss)/m
        cost=cost_fun(X,Y,theta,m)
        theta=theta-a*grad
        cost_count[i]=cost #storing every iterated value of cost function to plot graph
        t[i]=theta[1] #storing every iterated value of theta1 to plot graph
    return theta,cost_count,t

#main function
def main():
    data=pd.read_csv('my_data.csv') #reading csv file
    x1=data["sr"].values
    x1=(x1-np.mean(x1))/(max(x1)-min(x1))
    y=data["amp"].values 
   
    m=len(y)
    x0=np.ones(m)
    X=np.array([x0,x1])
    Y=np.array(y)
    theta=np.array([0,0])
    a=LEARNING_RATE
    itrns=ITERATIONS
    j=list(range(0,ITERATIONS,1)) #creating array of iterations to plot graph

    #gradient function calling which returns iterated value of theta,cost functiion and theta1
    new_theta,cost_count,t=grad_des(X,Y,theta,a,itrns,m)

    print('cost_function=',cost_count)


    plt.scatter(j,cost_count,color='red')
    plt.plot(j,cost_count)
    plt.xlabel('iterations')
    plt.ylabel('cost function')
    plt.title('cost function vs iterations curve ')
    plt.show()

    
    plt.scatter(t, cost_count, color='red')
    plt.plot(t,cost_count)
    plt.xlabel('parameter')
    plt.ylabel('cost function')
    plt.title('cost function vs parameter ')
    plt.show()
    
    print('final theta values=',new_theta)
    #calculating model function value with latest values of theta
    pr=hyp(X,new_theta) 
    print('predicted_values=',pr)


    plt.scatter(x1,y,color='r')
    plt.plot(x1,pr)
    plt.xlabel('data set')
    plt.ylabel('output')
    plt.title('predicted(line) and Actual(dot) output')
    plt.show()
    
if __name__=="__main__":
    main()
