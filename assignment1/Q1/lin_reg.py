import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.offsetbox import AnchoredText
import argparse



#function to normalize X
def normalize(X):
    m = np.mean(X, axis = 0)
    s = np.std(X, axis = 0)
    return (X - m)/s

#function to calculate J(theta)
def J_cost(y, X, theta):
    return 0.5*np.average((y-np.matmul(X, theta)) ** 2, axis = 0)

#function to calculate grad(J(theta))
def grad_J(y, X, theta):
    return (-1)*np.reshape(np.average(np.multiply((y - np.matmul(X, theta)), X), axis = 0), (2, 1))

#gradient descent with theta initiazlized to zero
#eta is the learning_rate
#min_error is the error threshold for convergence
#max_iter is the maximum number of iterations the loop can run if not converged
def grad_descent(X, y, eta, min_error, max_iter = 100000):
    cost = []
    theta_arr = []
    theta = np.zeros((X.shape[1], 1))
    theta_arr.append(theta)
    J = 0.5*np.average((y - np.matmul(X, theta)) ** 2, axis = 0)
    cost.append(J[0])
    J_prev = J
    iter = 0
    while (iter == 0 or (iter > 0 and iter <= max_iter and np.linalg.norm(J - J_prev, ord = 2) > min_error)):
        grad = grad_J(y, X, theta)
        theta = theta - eta * grad
        theta_arr.append(theta)
        J_prev = J
        J = 0.5*np.average((y - np.matmul(X, theta)) ** 2, axis = 0)
        cost.append(J[0])
        iter+=1
        norm1 = np.linalg.norm(J - J_prev, ord = 2)
    return cost, theta_arr, theta
#Note: in above function, (cost, theta_arr) store (J, theta) at every iteration, to be used by plot functions


#function for animation of 3d surface and contour plots
def anim_func(n, dataPoints, line, line2, iter_text, J_text, framesize):
    line.set_data(dataPoints[0:2, 0:n*framesize])
    line.set_3d_properties(dataPoints[2, 0:n*framesize])
    line2.set_data(dataPoints[0:2, 0:n*framesize])
    iter_text.set_text('Iteration: ' + str(n*framesize))
    J = dataPoints[2][n*framesize]
    J_text.set_text('J(\u03B8): ' + str(J))
    theta0 = '%.3f' % dataPoints[0][n*framesize]
    theta1 = '%.3f' % dataPoints[1][n*framesize]
    
    return line


if __name__ == '__main__':

    #set command line arg parser
    parser = argparse.ArgumentParser(description = 'Implementation of batch linear regression using gradient descent')
    parser.add_argument('fileX', help = 'csv file for X input')
    parser.add_argument('fileY', help = 'csv file for Y input')
    parser.add_argument('--l_rate', '-l', type = float, help = 'Learning Rate for gradient Descent')
    parser.add_argument('--error', '-e', type = float, help = 'threshold for min_error')
    parser.add_argument('--animation_framesize', '-f', help = 'iterations covered in one frame of animation')
    args = parser.parse_args()
    
    #set hyperparameters
    learning_rate = 1.7
    if args.l_rate:
        learning_rate = args.l_rate
    
    error = 1e-8
    if args.error:
        error = args.error

    #set scales (sc_x, sc_y) for viewing 3D plot
    #do not change for given plot
    sc_x = 1
    sc_y = 1000

    #framesize is the no. of iterations skipped in every frame during animation. To view every iteration, set to 1
    framesize = 1
    if(args.animation_framesize):
        framesize = int(args.animation_framesize)

    #reading from CSV
    df_x = pd.read_csv(args.fileX, header = None)
    X = np.reshape(df_x, (df_x.shape[0], 1))
    df_y = pd.read_csv(args.fileY, header = None)
    y = np.reshape(df_y.to_numpy(), (df_y.shape[0], 1))

    #preprocessing(normalization and adding intercept)
    X = normalize(X)
    ones = np.ones((X.shape[0], 1))
    X = np.hstack([ones, X])          #X is the training data of shape(m, 2) with first column set to 1.

    
    #call to grad_descent function
    cost, theta_arr, theta = grad_descent(X, y, eta = learning_rate, min_error = error)

    #print final values on command line
    
    print(f'Iterations per animation frame: {framesize}')
    print(f'learning_rate: {learning_rate}')
    print(f'Error threshold: {error}')
    print(f'Iterations: {len(cost)}')
    print(f'theta0:  {theta[0]}')
    print(f'theta1: {theta[1]}')
    
    #code to plot cost curve
    fig_cost = plt.figure()
    plt.plot(cost)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost function: J(\u03B8)')

    #plot regression line and datapoints
    fig_plot = plt.figure()
    plt.scatter(df_x, df_y)
    plt.xlabel('Acidity of Wine')
    plt.ylabel('Density')
    plt.plot(df_x, X@theta, color='red', label = 'Best Fit Line')
    plt.legend()
    

    ####################
    ###code for plots###
    ####################
    fig = plt.figure(figsize = plt.figaspect(2.0))
    
    ax = fig.add_subplot(2,1,1, projection='3d')
    
    
    theta_arr = np.reshape(theta_arr, (len(theta_arr), 2))
    theta0_arr = theta_arr[:, 0]
    theta1_arr = theta_arr[:, 1]

    #creating 2D mesh for theta0, theta1
    theta0_mesh_arr = np.arange(theta[0] - sc_x*(theta[0] - theta0_arr[0]), theta[0] + sc_x*(theta[0] - theta0_arr[0]), 2*sc_x*0.01*(theta[0] - theta0_arr[0]))
    theta1_mesh_arr = np.arange(theta[1] - sc_y*(theta[1] - theta1_arr[0]), theta[1] + sc_y*(theta[1] - theta1_arr[0]), 2*sc_y*0.01*(theta[1] - theta1_arr[0]))
    J_arr = cost
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_mesh_arr, theta1_mesh_arr)

    t1 = np.dstack((theta0_mesh, theta1_mesh))
    Z = np.empty((t1.shape[0], t1.shape[1]))
    
    #calculating Z values for 3D suface and contour
    for i in range (t1.shape[0]):
        for j in range(t1.shape[1]):
            theta_ij = np.reshape(t1[i][j], (2,1))
            Z[i][j] = J_cost(y, X, theta_ij)
    
    
    #plotting of 3D surface
    ax.plot_surface(theta0_mesh, theta1_mesh, Z, alpha = 0.5)
    line = ax.plot([], [], [], c = 'r', linestyle='solid')[0]
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'$J(\theta)$')
    dataPoints = np.array([theta0_arr, theta1_arr, J_arr])
    n_points = len(cost)
    
    
    #plotting of contours
    ax2 = fig.add_subplot(2,1,2)
    ax2.set_title(r'Contour Plot for $\theta_1$ and $\theta_2$')
    contour_plot = ax2.contour(theta0_mesh, theta1_mesh, Z, sorted(J_arr))
    ax2.set_xlabel(r'$\theta_1$')
    ax2.set_ylabel(r'$\theta_2$')
    dots = ax2.plot(theta0_arr, theta1_arr, c = 'r', marker = 'o', lw = 0, markersize = 5, alpha = 0.5)[0]
    ax2.text(0.03, 0.95, 'Learning Rate: ' + str(learning_rate), transform =  ax2.transAxes, size = 8)
    iter_text = ax2.text(0.03, 0.90, '', transform=ax2.transAxes, size = 8)
    J_text = ax2.text(0.03, 0.85, '', transform=ax2.transAxes, size = 8)


    #saving animation and 2D plots
    fig_cost.savefig(f'lin_reg_learning_rate_{learning_rate}_cost_curve.png')
    fig_plot.savefig(f'lin_reg_learning_rate_{learning_rate}_plot.png')
    anim = animation.FuncAnimation(fig, anim_func, frames = int(n_points/framesize), interval = 200, fargs=(dataPoints, line, dots, iter_text, J_text, framesize))
    videowriter = animation.FFMpegWriter(fps=5)
    anim.save(f'lin_reg_learning_rate_{learning_rate}_animation.mp4', writer = videowriter)
    plt.show()


