import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import argparse

#normalize X
def normalize(X):
    m = np.mean(X, axis = 0)
    s = np.std(X, axis = 0)
    normalized_X = (X - m)/s
    return normalized_X

#calculate phi
def phi(y):
    return np.mean(y, axis = 0)
#calculate mean vector for class with value = 0
def mu_0(X, y):
    return np.dot(X.T, 1 - y)/np.sum(1-y, axis = 0)
# calculate mean vector for class with value = 1
def mu_1(X, y):
    return np.dot(X.T, y)/np.sum(y, axis = 0)

#calculate covariance matrix, assuming both distribution have same sigma
def sigma(X, y):
    m0 = mu_0(X, y)
    m1 = mu_1(X, y)
    m0_extended = np.ones((y.shape[0], 1))@(m0.T)
    m1_extended = np.ones((y.shape[0], 1))@(m1.T)
    X_shifted_m0 = X - m0_extended
    X_shifted_m0_with_zeros = np.multiply(X_shifted_m0, 1 - y)
    X_shifted_m1 = X - m1_extended
    X_shifted_m1_with_zeros = np.multiply(X_shifted_m1, y)

    X_shifted = X_shifted_m0_with_zeros + X_shifted_m1_with_zeros
    sigma = np.dot(X_shifted.T, X_shifted)/X.shape[0]
    return sigma

# calculate covariance matrix for class with value = 0
def sigma0(X, y):
    m0 = mu_0(X, y)
    m0_extended = np.ones((y.shape[0], 1))@(m0.T)
    #y_extended = np.hstack([y, y])
    X_shifted = X - m0_extended
    X_shifted_with_zeros = np.multiply(X_shifted, 1 - y)
    result = X_shifted_with_zeros.T@X_shifted
    result = result/np.sum(1-y, axis = 0)
    return result

# calculate covariance matrix for class with value = 1
def sigma1(X, y):
    m1 = mu_1(X, y)
    m1_extended = np.ones((y.shape[0], 1))@(m1.T)
    #y_extended = np.hstack([y, y])
    X_shifted = X - m1_extended
    X_shifted_with_zeros = np.multiply(X_shifted, y)
    result = X_shifted_with_zeros.T@X_shifted
    result = result/np.sum(y, axis = 0)
    return result

#In below functions, A = (P(x|y=0;theta)P(y=0;theta))/(P(x|y=1;theta)P(y=1;theta))
#As discussed in class, decision boundary is given by (1/(1+A)) = 0.5 or log(A) = 0
#The functions calculate value of log(A). Contour is detected when log(A) = 0.

#log(A) expression assuming sigma0 = sigma1 = sigma
def log_A_linear(x, sigma, mu0, mu1, phi):
    c1 = np.log((1-phi)/phi)
    sigma_inv = np.linalg.inv(sigma)
    c2 = 0.5*((mu1.T)@sigma_inv@mu1) - 0.5*((mu0.T)@sigma_inv@mu0)
    c3 = (-1)*((mu1.T@sigma_inv - mu0.T@sigma_inv)@x)

    return c1+c2+c3

#log(A) expression assuming sigma0 != sigma1
def log_A_quadratic(x, sigma0, sigma1, mu0, mu1, phi):
    det0 = np.linalg.det(sigma0)
    det1 = np.linalg.det(sigma1)
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    c1 = np.log(np.sqrt(det1)/np.sqrt(det0))
    c2 = np.log((1-phi)/phi)
    c3 = 0.5*((x - mu1).T@(sigma1_inv)@(x-mu1))
    c4 = (-0.5)*((x - mu0).T@(sigma0_inv)@(x-mu0))
    return c1 + c2 + c3 + c4


#returns mesh for X1, X2, Z when sigma0 = sigma1 = sigma
def boundary_linear(X, sigma, mu0, mu1, phi):
    x1 = X[:, 0]
    x2 = X[:, 1]
    max1 = np.max(x1)
    min1 = np.min(x1)
    max2 = np.max(x2)
    min2 = np.min(x2)
    x1_arr = np.arange(min1, max1, 0.005*(max1 - min1))
    x2_arr = np.arange(min2, max2, 0.005*(max2 - min2))
    x1_mesh, x2_mesh = np.meshgrid(x1_arr, x2_arr)
    mat = np.dstack((x1_mesh, x2_mesh))

    Z = np.empty((mat.shape[0], mat.shape[1]))

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            x_ij = np.reshape(mat[i][j], (2, 1))
            Z[i][j] = log_A_linear(x_ij, sigma, mu0, mu1, phi)
    
    return x1_mesh, x2_mesh, Z

#returns mesh for X1, X2, Z when sigma0 != sigma1
def boundary_quadratic(X, sigma0, sigma1, mu0, mu1, phi):
    x1 = X[:, 0]
    x2 = X[:, 1]
    max1 = np.max(x1)
    min1 = np.min(x1)
    max2 = np.max(x2)
    min2 = np.min(x2)
    x1_arr = np.arange(min1, max1, 0.005*(max1 - min1))
    x2_arr = np.arange(min2, max2, 0.005*(max2 - min2))
    x1_mesh, x2_mesh = np.meshgrid(x1_arr, x2_arr)
    mat = np.dstack((x1_mesh, x2_mesh))

    Z = np.empty((mat.shape[0], mat.shape[1]))

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            x_ij = np.reshape(mat[i][j], (2, 1))
            Z[i][j] = log_A_quadratic(x_ij, sigma0, sigma1, mu0, mu1, phi)
    
    return x1_mesh, x2_mesh, Z


    
    

if __name__ == '__main__':
    #argparse arguments
    parser = argparse.ArgumentParser(description='Implementation for GDA')
    parser.add_argument('fileX', help = 'File for X input')
    parser.add_argument('fileY', help = 'File for Y input')
    args = parser.parse_args()

    #reading and preprocessing X and y data
    X = pd.read_csv(args.fileX, sep = '  ', header = None, engine='python')
    X = X.to_numpy()
    X = normalize(X)
    y = pd.read_csv(args.fileY, sep = '  ', header = None, engine = 'python')
    y = y.to_numpy()
    y = [0 if y1 == 'Alaska' else 1 for y1 in y]
    y = np.reshape(y, (len(y), 1))

    # calculates all parameters in GDA
    phi = phi(y)
    sigma = sigma(X, y)
    sigma0 = sigma0(X, y)
    sigma1 = sigma1(X, y)
    mu0 = mu_0(X, y)
    mu1 = mu_1(X, y)

    # print parameters to command line
    print (f'\nphi: {phi}')
    print(f'\nmu0: {mu0}')
    print(f'\nmu1: {mu1}\n')
    print('\nAssuming sigma0 == sigma1:')
    print(f'\nsigma: {sigma}\n')

    print('\nAssuming sigma0 != sigma1')
    print(f'\nsigma0: {sigma0}')
    print(f'\nsigma1: {sigma1}')
    
    #get 2D mesh values when sigma0 = sigma1 = sigma
    x1_mesh, x2_mesh, Z_mesh = boundary_linear(X, sigma, mu0, mu1, phi)


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    #plot linear boundary for sigma0 = sigma1 = sigma
    ax.contour(x1_mesh, x2_mesh, Z_mesh, [0.], colors = 'green')

    #get 2D mesh values when sigma0 != sigma1, plot quadratic boundary
    x1_mesh, x2_mesh, Z_mesh = boundary_quadratic(X, sigma0, sigma1, mu0, mu1, phi)
    ax.contour(x1_mesh, x2_mesh, Z_mesh, [0. ], colors = 'red')


    x1 = X[:, 0]
    x2 = X[:, 1]

    #Alaska assumed to have value = 0, Canada has value = 1
    x1_Alaska = [v for i, v in enumerate(x1) if y[i] == 0]
    x2_Alaska = [v for i, v in enumerate(x2) if y[i] == 0]

    x1_Canada = [v for i, v in enumerate(x1) if y[i] == 1]
    x2_Canada = [v for i, v in enumerate(x2) if y[i] == 1]

    #scatter plot for datapoints
    scatter1 = ax.scatter(x1_Alaska, x2_Alaska, marker = 'o', label = 'Alaska')
    scatter2 = ax.scatter(x1_Canada, x2_Canada, marker = 'x', label = 'Canada')

    
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title(r'Plot for normalized $x_1$ and $x_2$ values')
    legend_elements = [Line2D([], [], color = 'green', label = 'Label1'), Line2D([], [], color = 'red', label = 'Label2')]
    ax.legend([scatter1, scatter2, legend_elements[0], legend_elements[1]], ['Alaska', 'Canada', r'Linear Separator: $\Sigma_0 = \Sigma_1 = \Sigma$', r'Quadratic Separator: $\Sigma_0 \neq\Sigma_1$'])

    #save plot
    fig.savefig('gda_plot.png')
    plt.show()
    


