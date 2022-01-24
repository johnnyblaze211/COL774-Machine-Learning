import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import time
import argparse

#function to return random sample data X, y
#mean_var_X is array of (mean, variance) pairs for x_1, x_2 .... x_n respectively
#var_noise is the variance of the gaussian nois
#size is sample size
def random_sample(theta, mean_var_X, var_noise, size):
    n = theta.shape[0]
    assert(mean_var_X.shape[0] == n-1)
    X = np.ones((size,1))
    for i in range(n-1):
        normal_arr = np.random.normal(mean_var_X[i][0], np.sqrt(mean_var_X[i][1]), (size, 1))
        X = np.hstack([X, normal_arr])
    yhat = np.matmul(X, theta)
    noise = np.random.normal(0, np.sqrt(var_noise), (size, 1))
    y = yhat + noise
    y = np.reshape(y, (y.shape[0], 1))
    
    return X, y

# function returns gradient for given theta and batch_X, batch_y
def grad_batch(theta, batch_X, batch_y):
    assert(batch_X.shape[0] == batch_y.shape[0])
    A = batch_y - np.reshape(np.matmul(batch_X, theta), (batch_X.shape[0], 1))
    B = np.multiply(A, batch_X)
    grad = (-1)*np.reshape(np.average(B, axis = 0), (theta.shape[0], 1))
    return grad

# returns cost of theta with batch: (batch_X, batch_y)
def J_theta(theta, batch_X, batch_y):
    assert(batch_X.shape[0] == batch_y.shape[0])
    np.reshape(batch_y, (batch_y.shape[0], 1))
    n = batch_X.shape[0]
    dot = np.matmul(batch_X, theta)
    A = (batch_y - np.reshape(dot, (batch_X.shape[0], 1)))
    A = A**2
    J =  0.5*np.average(A, axis = 0)
    return J

#code for minibatch gradient descent
#init_theta is the initial theta(zero vector by our assumptions)
#eta is the learning_rate
#min_error is the error threshold for convergence
#We check for convergence by averaging the cost of n_check_conv consecutive batches and comparing the change with average of previous n_check_conv consecutive batches
#plot2D_iter_skip = i such that we plot the results of every ith iteration on all our plots
def minibatch_gradient_descent(init_theta, X, y, eta, min_error, batch_size, n_check_conv, max_iter = 5000000, plot2D_iter_skip = 1):
    theta = init_theta
    n = X.shape[0]
    X_batches = np.split(X, int(n/batch_size))
    y_batches = np.split(y, int(n/batch_size))

    #store cost, iteration number for every plot2D_iter_skip iteration
    cost = []
    theta_arr = []
    iter_arr = []

    #avg: average of current n_check_conv batches
    #avg_prev: average of previous set of n_check_conv batches
    #iter: iteration number
    #sum_cost: stores sum of costs upto n_check_conv batches
    iter, sum_cost, avg_prev, avg = 0,0,0,0

    #iterates over all batches until convergence
    while True:
        for i, (batch_X, batch_y) in enumerate(zip(X_batches, y_batches)):
            #check every n_check_conv iterations if converged
            if(iter % n_check_conv == 0):
                avg = sum_cost/n_check_conv
                #if convergence satisfied return values, else set sum_cost to zero
                if((iter > n_check_conv and np.abs(avg_prev - avg) < min_error) or iter >= max_iter):
                    print(f'\niter:{iter}, \ntheta:{theta}\n', end = '') #commented
                    print(f'cost_diff:{avg_prev - avg}\n') #commented
                    return cost, theta_arr, iter_arr, theta, iter
                else:
                    print(f'\niter:{iter}, \ntheta:{theta}\n', end = '') #commented
                    print(f'cost_diff:{avg_prev - avg}\n') #commented

                    avg_prev = avg
                    sum_cost = 0
            
            J = J_theta(theta, batch_X, batch_y)
            sum_cost += J
            n = len(cost)
            k = 10
            if(iter%plot2D_iter_skip == 0):
                cost.append(J)
                theta_arr.append(theta)
                iter_arr.append(iter)

            #theta update step for a single batch
            grad = grad_batch(theta, batch_X, batch_y)
            theta = theta - eta*grad

            iter += 1

#function for matplotlib animation
def anim_func(n, dataPoints, scatter, line):
    scatter.set_data(dataPoints[0][n], dataPoints[1][n])
    scatter.set_3d_properties(dataPoints[2][n])
    line.set_data(dataPoints[0][0:n], dataPoints[1][0:n])
    line.set_3d_properties(dataPoints[2][0:n])
    return scatter, line
            



if __name__ == '__main__':

    #argparse arguments
    parser = argparse.ArgumentParser(description='implementation of SGD')
    parser.add_argument('file_test', help = 'csv test file')
    parser.add_argument('--l_rate', '-l', help = 'Learning Rate for SGD')
    parser.add_argument('--batch_size', '-b', help = 'Batch size for SGD')
    parser.add_argument('--n_check_conv', '-n', help = 'Checks convergence after every n batches')
    parser.add_argument('--error', '-e', help = 'threshold for convergence')
    parser.add_argument('--plot_iter_to_skip', '-s', help = 'Plot data for every s iterations')
    args = parser.parse_args()

    #calculate random sample with gaussian noise
    sample_theta = np.reshape(np.array([3, 1, 2]), (3,1))
    mean_var_X = np.array([[3, 4], [-1, 4]])
    var_noise = 2.0
    size = int(1e6)
    X, y = random_sample(sample_theta, mean_var_X, var_noise, size)
    print(f'var(y): {np.var(y, axis=0)}')
    print(f'mean(y): {np.mean(y, axis = 0)}')


    #preprocess X and y test datasets
    test_data = pd.read_csv(args.file_test)
    x1_test = np.reshape(test_data["X_1"].to_numpy(), (-1, 1))
    x2_test = np.reshape(test_data["X_2"].to_numpy(), (-1, 1))
    test_X = np.hstack([x1_test, x2_test])
    test_X = np.hstack([np.ones((x1_test.shape[0], 1)), test_X])
    test_y = np.reshape(test_data["Y"].to_numpy(), (-1, 1))
    

    # set hyperparameter values
    init_theta = np.reshape(np.array([0., 0., 0.]), (3,1))
    learning_rate = 0.001
    if args.l_rate:
        learning_rate = float(args.l_rate)
    min_error = 3.e-3
    if args.error:
        min_error = float(args.error)
    batch_size = 1
    if args.batch_size:
        batch_size = int(args.batch_size)
    n_check_conv = 17963
    if args.n_check_conv:
        n_check_conv = int(args.n_check_conv)
    plot2D_iter_skip = 691
    if args.plot_iter_to_skip:
        plot2D_iter_skip = int(args.plot_iter_to_skip)

    #call to SGD function
    start = time.time()
    cost, theta_arr, iter_arr, theta, iterations = minibatch_gradient_descent(init_theta, X, y, learning_rate, min_error, batch_size, n_check_conv, plot2D_iter_skip = plot2D_iter_skip)
    end = time.time()

    #print results to command line
    print('\n\nResults: ')
    print(f'Error of test data with sample_theta: {J_theta(sample_theta, test_X, test_y)}')
    print(f'Learning_rate: {learning_rate}')
    print(f'Batch_size: {batch_size}')
    print(f'n_check_conv: {n_check_conv}')
    print(f'error threshold: {min_error}')
    print(f'Plot every ith iteration: i = {plot2D_iter_skip}\n')
    print(f'\nfinal theta: {theta}')
    print(f'Time for procedure: {end - start} seconds')
    print(f'Error for test_data: {J_theta(theta, test_X, test_y)}')
    print(f'No. of iterations: {iterations} ')

    ####################
    ###code for plots###
    ####################

    #3d line plot of theta
    fig1 = plt.figure()
    ax0 = fig1.add_subplot(1, 1, 1, projection = '3d')
    
    theta_arr = np.reshape(theta_arr, (len(theta_arr), 3))
    theta0_arr = theta_arr[:, 0]
    theta1_arr = theta_arr[:, 1]
    theta2_arr = theta_arr[:, 2]
    dataPoints = np.array([theta0_arr, theta1_arr, theta2_arr])

    ax0.set_xlabel(r'$\theta_0$')
    ax0.set_ylabel(r'$\theta_1$')
    ax0.set_zlabel(r'$\theta_2$')
    

    line = ax0.plot([], [], [], color = 'grey', alpha = 0.3, lw = 1)[0]
    scatter  = ax0.plot([], [], [], c = 'r', marker = 'o', alpha = 0.5)[0]
    sc1 = ax0.scatter([3], [1], [2], color = 'g', alpha = 0.5, label = r'expected $\theta\  =\  [3\ 1\ 2]$')
    plt.legend([sc1, scatter], [r'expected $\theta\  =\  [3\ 1\ 2]$', f'\u03B8 at every {plot2D_iter_skip}th iteration'])

    #2D plot of cost curve
    fig2 = plt.figure()
    ax = fig2.add_subplot(1,1,1)
    ax.plot(iter_arr, cost)
    plt.xlabel('No. of batch iterations')
    plt.ylabel('Cost: J(\u03B8)')

    #save animation and image
    ax.set_title(f'Cost function for SGD(batch size = {batch_size})')
    anim = animation.FuncAnimation(fig1, anim_func, frames = int(len(iter_arr)), interval = 200, fargs = (dataPoints, scatter, line), blit = True)
    fig2.savefig(f'sgd_batch_{batch_size}.png')
    videowriter = animation.FFMpegWriter(fps=10)
    anim.save(f'sgd_batch_{batch_size}_animation.mp4', writer = videowriter)
    plt.show()

