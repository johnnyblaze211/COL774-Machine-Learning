import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import argparse

#sigmoid function
def sigmoid(A):
    return 1/(1+np.exp(-A))

#h(theta) for logistic regression
def h_theta(theta, X, y):
    Z = X@theta
    h = sigmoid(Z)
    return h

#calculates Hessian matrix of J(theta) for given values of theta, X, y
def hessian(theta, X, y):
    h = h_theta(theta, X, y)
    prod = h*(1-h)
    prod = np.reshape(prod, (prod.shape[0], ))
    diag = np.diag(prod)
    
    hess = np.dot(np.dot(X.T, diag), X)/y.shape[0]

    return hess

    

    

    
# cost function J for logistic regression
def J_theta(theta, X, y):
    h = h_theta(theta, X, y)
    L = -(y*np.log(h) + (1-y)*np.log(1-h))
    J = np.mean(L, axis = 0)
    
    return J

#returns gradient of J(theta) for given values of theta, X, y
def grad_J(theta, X, y):
    h = h_theta(theta, X, y)
    prod = np.multiply((y - h), X)
    grad = (-1) * np.reshape(np.mean(prod, axis = 0), (X.shape[1], 1))
    return grad

#logistic regression with Newton's update
def logistic_regression_newton(theta_init, X, y, min_error, max_iter = 100000, plot2D_iter_skip = 1):
    cost = []
    iter_arr = []
    theta_arr = []

    iter = 0
    theta = theta_init
    while (iter < 2 or (np.abs(cost[-1] - cost[-2]) > min_error and iter < max_iter)):
        J = J_theta(theta, X, y)
        grad = grad_J(theta, X, y)
        H = hessian(theta, X, y)
        H_inv = np.linalg.inv(H)

        if(iter%plot2D_iter_skip == 0):
            cost.append(J)
            iter_arr.append(iter)
            theta_arr.append(theta)
        iter+=1

        #update of theta at each step
        theta = theta - np.dot(H_inv, grad)
    
    return cost, theta_arr, iter_arr, theta, iter

 
if __name__ == '__main__':
    #argparse arguments
    parser = argparse.ArgumentParser(description = 'Implementation of Logistic Regression using Newton\'s Method')
    parser.add_argument('fileX', help = 'File for input X')
    parser.add_argument('fileY', help = 'File for input Y')
    parser.add_argument('--error', '-e', help = 'threshold for convergence')
    args = parser.parse_args()

    #reading and preprocessing X input
    df_x = pd.read_csv(args.fileX, header = None)
    X = df_x.to_numpy()
    mean = np.average(X, axis = 0)
    std_dev = np.std(X, axis = 0)
    X = (X - mean)/std_dev  #normalization
    x0 = np.ones((X.shape[0], 1))
    X = np.hstack([x0, X])

    #reading y input
    y = pd.read_csv(args.fileY, header = None).to_numpy()

    #set hyperparameter values
    theta_init = np.zeros((X.shape[1], 1))
    min_error = 1e-5
    if(args.error):
        min_error = float(args.error)

    #get results from log_reg function
    J_arr, theta_arr, iter_arr, theta, iter = logistic_regression_newton(theta_init, X, y, min_error)
    print(f'Error_threshold: {min_error}')
    print(f'Iterations: {iter}')
    print(f'theta: {theta}')

    ####################
    ###code for plots###
    ####################

    #plotting assuming theta vector has 3 dimensions(including intercept coefficient)
    x1 = X[:, 1]
    x2 = X[:, 2]
    markers = []
    colors = []
    labels = ['Value = 0', 'Value = 1']
    for i in y:
        if i == 1:
            markers.append('+')
            colors.append('r')
        else:
            markers.append('o')
            colors.append('g')

    #plot input data points
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    bool0 = False
    bool1 = False
    for i, (x_, y_, m, c) in enumerate(zip(x1, x2, markers, colors)):
        label = None
        if(y[i] == 1 and not bool1):
            label = 'Value: 1'
            bool1 = True
        elif(y[i] == 0 and not bool0):
            label = 'Value: 0'
            bool0 = True
        plt.scatter(x_, y_, marker = m, c = c, label = label)
    
    #get meshgrids for contour plots
    x1_arr = np.arange(x1.min(), x1.max(), 0.01*(x1.max() - x1.min()))
    x2_arr = np.arange(x2.min(), x2.max(), 0.01*(x2.max() - x2.min()))
    x1_mesh, x2_mesh = np.meshgrid(x1_arr, x2_arr)
    Z = sigmoid(theta[0] + theta[1]*x1_mesh + theta[2] * x2_mesh)

    #plot linear separator on 2d scatter plot
    plt.contour(x1_mesh, x2_mesh, Z, [0.5], colors = 'purple')

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    plt.title('Logistic Regression using Newton\'s method(Normalized)')
    plt.legend()

    #save image
    fig.savefig('log_reg.png')
    plt.show()

    

