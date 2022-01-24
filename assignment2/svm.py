import pandas as pd
import numpy as np
import time
import sys
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import cvxopt as cvx
from cvxopt import matrix as cvxmat
from libsvm.svmutil import *
import argparse

thresh = 1e-4
const = 1
const_gamma = 0.05

label_neg1 = 8
label_pos1 = 9




def df_normalize_bin(df_x, df_y, label_plus = label_pos1, label_minus = label_neg1):
    assert(df_x.shape[0] == df_y.shape[0] and df_y.shape[1] == 1)
    X = df_x.to_numpy()
    X = X/255
    y = np.reshape(df_y.to_numpy(), (-1, 1))
    y[y == label_minus] = -1
    y[y == label_plus] = 1

    return X, y
def df_normalize_multi(df_x, df_y):
    assert(df_x.shape[0] == df_y.shape[0] and df_y.shape[1] == 1)
    X = df_x.to_numpy()
    X = X/255
    y = np.reshape(df_y.to_numpy(), (-1, 1))
    return X, y
    
def preprocess_bin(train_df, test_df, label_plus = label_pos1, label_minus = label_neg1, mode = 'binary'):
    train_df_bin = train_df.loc[train_df[784].isin([label_plus, label_minus]), :]
    test_df_bin = test_df.loc[test_df[784].isin([label_plus, label_minus]), :]


    if mode == 'binary':

        m = train_df.shape[0]
        #Following 80:20 validation split
        m_train = int(0.8*m)
        train_df_data = train_df_bin.loc[:(m_train - 1), :783]
        train_df_label = train_df_bin.loc[:(m_train - 1), 784:]
        validation_df_data = train_df_bin.loc[m_train:, :783]
        validation_df_label = train_df_bin.loc[m_train:, 784:]
        test_df_data = test_df_bin.loc[:, :783]
        test_df_label = test_df_bin.loc[:, 784:]


        
        #Normalize and adjust y
        X_train, y_train = df_normalize_bin(train_df_data, train_df_label)
        X_val, y_val = df_normalize_bin(validation_df_data, validation_df_label)
        X_test, y_test = df_normalize_bin(test_df_data, test_df_label)

        return X_train, y_train, X_val, y_val, X_test, y_test
    
    elif mode == 'multi':
        train_df_data = train_df_bin.loc[:, :783]
        train_df_label = train_df_bin.loc[:, 784:]
        test_df_data = test_df.loc[:, :783]
        test_df_label = test_df.loc[:, 784:]

        X_train, y_train = df_normalize_bin(train_df_data, train_df_label, label_plus = label_plus, label_minus = label_minus)
        X_test, y_test = df_normalize_multi(test_df_data, test_df_label)

        return X_train, y_train, X_test, y_test
    
    elif mode == 'multi_libsvm':
        train_df_data = train_df.loc[:, :783]
        train_df_label = train_df.loc[:, 784:]
        test_df_data = test_df.loc[:, :783]
        test_df_label = test_df.loc[:, 784:]

        X_train, y_train = df_normalize_multi(train_df_data, train_df_label)
        X_test, y_test = df_normalize_multi(test_df_data, test_df_label)

        return X_train, y_train, X_test, y_test

    

def get_cvxopt_params(X, y, gaussian = False):
    assert(y.shape[0] == X.shape[0])
    m = y.shape[0]
    if not gaussian:
        Q = np.multiply(y, X)
        P = cvxmat(Q@(Q.T))
    else:
        X2 = X@X.T
        x2_diag = np.diag(X2).reshape((m, 1))
        x2_diag_ext = np.hstack([x2_diag for i in range(m)])
        M = np.exp(-const_gamma*(x2_diag_ext + x2_diag_ext.T - 2*X2))
        P1 = (y@y.T)*M
        P = cvxmat(P1)


    q = cvxmat((-1)*np.ones((m, 1)))

    A = cvxmat(y.T, tc = 'd')

    b = cvxmat(0, tc = 'd')

    i1 = np.identity(m)
    i2 = -1*i1
    G = cvxmat(np.vstack([i1, i2]))

    j1 = const * np.ones((m, 1))
    j2 = np.zeros((m, 1))
    h = cvxmat(np.vstack([j1, j2]))

    return P, q, G, h, A, b


def solve_cvx_binary_linear(X, y):
    start = time.time()
    assert(y.shape[0] == X.shape[0])
    P, q, G, h, A, b = get_cvxopt_params(X, y)
    solver = cvx.solvers.qp(P, q, G, h, A, b, options = {'show_progress': False})
    alpha = np.array(solver['x'])
    
    support_vector_arr = []
    m = X.shape[0]
    n = X.shape[1]
    w = np.zeros((n,1))
    for i in range(m):
        if alpha[i] > thresh:
            w += (alpha[i]*y[i]) * np.reshape(X[i], (n, 1))
            support_vector_arr.append(i)

    gamma = const - alpha
    b = 0.
    bCount = 0
    bTotal = 0.
    bool1 = False
    min1 = np.inf
    max1 = -np.inf


    for i in range(m):
        if alpha[i] > thresh and gamma[i] > thresh:
            bool1 = True
            bval = y[i] - w.T @ X[i]
            bTotal+= bval
            bCount += 1


        if not bool1:
            if alpha[i] < thresh:
                if int(y[i]) == -1:
                    min1 = min(min1, w.T @ X[i])
                else:
                    max1 = max(max1, w.T @ X[i])
    if bool1:
        b = bTotal/bCount
    else:
        b = -(max1 + min1)/2
    
    end = time.time()
    
    return support_vector_arr, alpha, w, b, end - start

def solve_cvx_binary_gaussian(X, y):
    start = time.time()
    P, q, G, h, A, b = get_cvxopt_params(X, y, gaussian = True)
    solver = cvx.solvers.qp(P, q, G, h, A, b, options = {'show_progress': True})
    alpha = np.array(solver['x'])
    gamma = const - alpha
    support_vector_arr = []
    m = X.shape[0]
    n = X.shape[1]
    for i in range(m):
        if alpha[i] > thresh:
            support_vector_arr.append(i)
    

    bool1 = False
    min1 = np.inf
    max1 = -np.inf
    bTotal = 0
    bCount = 0

    X2 = X@X.T
    diagX2 = np.diag(X2).reshape((-1, 1))
    ext_diag_X2 = np.hstack([diagX2 for i in range(m)])
    M = ext_diag_X2 + ext_diag_X2.T - 2*X2
    exp_M = np.exp(-const_gamma*M)
    alpha_y = np.multiply(alpha, y)
    wt_X = exp_M@alpha_y
    for i in range(m):
        if alpha[i] > thresh and gamma[i] > thresh:
            bool1 = True
            bval = y[i] - wt_X[i]
            bCount+=1
            bTotal+=bval

        if not bool1:
            if alpha[i] < thresh:
                sum1 = 0.
                if int(y[i]) == -1:
                    min1 = min(min1, wt_X[i])
                else:
                    max1 = max(max1, wt_X[i])
    
    if bool1:
        b = bTotal/bCount
    else:
        b = -(max1 + min1)/2
    end = time.time()

    return support_vector_arr, alpha, b, end - start


    
            

        







def prediction_bin_linear(w, b, X_test, y_test):
    assert(X_test.shape[0] == y_test.shape[0] and y_test.shape[1] == 1)
    y_pred = X_test@w + b
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = -1

    acc = (np.sum(y_pred == y_test))/y_test.shape[0]

    return y_pred, acc

def prediction_bin_gaussian(alpha, X_train, y_train, b, X_test, y_test):

    m1 = X_test.shape[0]
    m2 = X_train.shape[0]

    X2 = X_train
    X1 = X_test
    alpha_y = np.multiply(alpha, y_train)


    X2_sq = X2@(X2.T)
    diag_X2_sq = np.diag(X2_sq).reshape((-1, 1))
    ext_diag_X2_sq = np.hstack([diag_X2_sq for i in range(m1)])
    M1 = ext_diag_X2_sq.T

    X1_sq = X1@(X1.T)
    diag_X1_sq = np.diag(X1_sq).reshape((-1, 1))
    ext_diag_X1_sq = np.hstack([diag_X1_sq for i in range(m2)])
    M2 = ext_diag_X1_sq

    M3 = X1@(X2.T)

    exp_M = np.exp(-const_gamma*(M1 + M2 - 2*M3))
    y_pred = exp_M@alpha_y + b
    y = y_pred.copy()
    y_pred[y_pred>=0] = 1
    y_pred[y_pred<0] = -1

    acc = (np.sum(y_pred == y_test))/y_test.shape[0]

    return y, y_pred, acc


def libsvm_bin_prediction(X, y, X_test, y_test, gaussian = False):
    y = np.reshape(y, (-1, ))
    y_test = np.reshape(y_test, (-1, ))
    start_time = time.time()
    svm_prob = svm_problem(y, X)
    if gaussian:
        svm_param = svm_parameter('-q -s 0 -t 2 -c 1 -g 0.05')
    else:
        svm_param = svm_parameter('-q -t 0 -c 1')
    model = svm_train(svm_prob, svm_param)
    end_time = time.time()
    y_pred, acc, val = svm_predict(y_test, X_test, model, '-q')

    bias = - model.rho[0]
    nSV = model.l
    return y_pred, bias, nSV, acc[0]/100, end_time - start_time

def cvx_multi_prediction(train_df, test_df):

    start_time = time.time()
    y1 = train_df.loc[:, 784:].to_numpy()
    min_label = y1.min()
    max_label = y1.max()
    votes = np.zeros((test_df.shape[0], max_label - min_label + 1))
    values = np.zeros((test_df.shape[0], max_label - min_label + 1))
    class_pred = np.zeros((test_df.shape[0], 1))
    class_value = np.zeros((test_df.shape[0], 1))

    z = np.zeros((test_df.shape[0], 1))
    for l1 in range(min_label, max_label):
        for l2 in range(l1+1, max_label+1):
            X_train, y_train, X_test, y_test = preprocess_bin(train_df, test_df, label_plus = l2, label_minus = l1, mode = 'multi')
            SVs, alpha, b, time_train = solve_cvx_binary_gaussian(X_train, y_train)
            y_mag, y_pred, _ = prediction_bin_gaussian(alpha, X_train, y_train, b, X_test, y_test)
            y_pred_l1 = y_pred.copy()
            y_pred_l1[y_pred_l1 == 1] = 0
            y_pred_l1[y_pred_l1 == -1] = 1
            y_pred_l2 = y_pred.copy()
            y_pred_l2[y_pred == -1] = 0

            y_mag_l1 = y_mag.copy()
            y_mag_l1[y_mag_l1>=0] = 0
            y_mag_l1 = -y_mag_l1
            y_mag_l2 = y_mag.copy()
            y_mag_l2[y_mag_l2<0] = 0

            bool1, bool2, bool3 = False, False, False
            if l1 - min_label > 0: 
                bool1 = True
                Z1 = np.hstack([z for i in range(l1 - min_label)])
            if l2 - l1 - 1 > 0: 
                bool2 = True
                Z2 = np.hstack([z for i in range(l2 - l1 - 1)])

            if max_label - l2 > 0: 
                bool3 = True
                Z3 = np.hstack([z for i in range(max_label - l2)])

            Y_pred = y_pred_l1.copy()
            Y_mag = y_mag_l1.copy()
            if bool1: 
                Y_pred = np.hstack([Z1, Y_pred])
                Y_mag = np.hstack([Z1, Y_mag])
            if bool2:
                Y_pred = np.hstack([Y_pred, Z2])
                Y_mag = np.hstack([Y_mag, Z2])
            Y_pred = np.hstack([Y_pred, y_pred_l2])
            Y_mag = np.hstack([Y_mag, y_mag_l2])
            if bool3:
                Y_pred = np.hstack([Y_pred, Z3])
                Y_mag = np.hstack([Y_mag, Z3])

            votes = votes + Y_pred
            values = values + Y_mag

            print(f'Done Classifier: ({l1}, {l2})')

    c1 = np.max(votes, axis = 1).reshape((-1, 1))
    bool_mat = (c1 == votes)

    list1 = np.argwhere(bool_mat == True)
    for [ex_i, label_i] in list1:
        v1 = values[ex_i][label_i]
        if(v1 > class_value[ex_i]):
            class_value[ex_i] = v1
            class_pred[ex_i] = label_i
    
    accuracy = (class_pred == y_test).sum()/y_test.shape[0]
    end_time = time.time()
    return class_pred, accuracy, end_time - start_time

def libsvm_multi_prediction(X_train, y_train, X_test, y_test):
    y_train2 = np.reshape(y_train, (-1, ))
    y_test2 = np.reshape(y_test, (-1, ))
    start_time = time.time()

    prob = svm_problem(y_train2, X_train)
    param = svm_parameter('-q -s 0 -t 2 -c 1 -g 0.05')
    model = svm_train(prob, param)

    y_pred, acc, val = svm_predict(y_test2, X_test, model, '-q')
    end_time = time.time()
    return y_pred, acc[0]/100, end_time - start_time

def get_confusion_matrix(y, y_pred, savepath = None):
    fig, ax1 = plt.subplots()
    conf_mat = confusion_matrix(y, y_pred)
    sns.heatmap(conf_mat, annot = True, ax = ax1, fmt = 'g')
    ax1.set_xlabel('Predicted Labels')
    ax1.set_ylabel('Actual Labels')
    ax1.set_title('Confusion matrix')
    if savepath != None:
        fig.savefig(savepath)
    return fig, ax1

def get_confusion_images(X, y, y_pred):
    y1 = np.reshape(y, (-1, ))
    yp = np.reshape(y_pred, (-1, ))
    t = (y1 == yp)
    l1 = np.argwhere(t == False)
    sample = random.sample(list(l1), 10)
    fig, axs = plt.subplots(2, 5)
    for i, ex in enumerate(sample):
        img = X[ex, :].reshape((28, 28))
        r = int(i/5)
        c = i%5
        axs[r, c].imshow(img, cmap = 'gray')
        axs[r, c].set_title('Label: ' + str(int(y1[ex]))+', Predicted: '+str(int(yp[ex])), fontsize = 6)
        axs[r, c].axis('off')
    return fig, axs

def k_fold_cross_validation(X_train, y_train, X_test, y_test, C_list = [10, 5, 1, 1e-3, 1e-5], k = 5):
    m = int(X_train.shape[0]/k)
    best_val_acc_list = []
    test_acc_list = []
    for c in C_list:
        param = svm_parameter('-q -h 0 -s 0 -t 2 -c '+str(c)+' -g 0.05')
        y_final = np.zeros((m, 1))
        best_acc = 0.
        best_idx = -1
        best_model = None
        for i in range(k):
            X_val = X_train[i*m:(i+1)*m, :]
            y_val = y_train[i*m:(i+1)*m, :]
            X_train2 = np.vstack([X_train[:i*m, :], X_train[(i+1)*m:, :]])
            y_train2 = np.vstack([y_train[:i*m, :], y_train[(i+1)*m:, :]])
            prob = svm_problem(y_train2.reshape((-1, )), X_train2)
            model = svm_train(prob, param)

            y_pred, acc, val = svm_predict(y_val.reshape((-1, )), X_val, model, '-q')
            print(f'Done validation fold {i+1}/{k} for c = {c}')
            if (acc[0] > best_acc):
                y_final = y_pred
                best_acc = acc[0]
                best_idx = i
                best_model = model
        print(f'K-fold validation accuracy for C = {c}: {best_acc/100}')
        best_val_acc_list.append(best_acc/100)

        prob = svm_problem(y_train.reshape((-1, )), X_train)
        model = svm_train(prob, param)
        y_pred, acc, val = svm_predict(y_test.reshape((-1, )), X_test, model, '-q')
        print(f'\nTest accuracy for C = {c} on full training data: {acc[0]/100}\n')
        test_acc_list.append(acc[0]/100)

    fig, ax = plt.subplots()
    ax.plot(C_list, best_val_acc_list, color = 'r', label = 'K-fold validation accuracy')
    ax.plot(C_list, test_acc_list, color = 'b', label = 'Test Accuracy')
    ax.set_xscale('log')
    ax.set_xlabel('Values for C')
    ax.set_ylabel('Accuracy')
    ax.set_title('K-fold cross validation')
    ax.legend()
    fig.savefig('plots/K_fold_cross_validation.png')

    return fig, ax





            


















    






            
            
            




            



















if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Implementation for SVMs')
    parser.add_argument('train_file', metavar = 'file1', help = 'Training file in CSV format')
    parser.add_argument('test_file', metavar = 'file2', help = 'Test file in CSV format')
    parser.add_argument('binary_or_multi', metavar = 'type', type = int, help = '0 for binary, 1 for multiclass')
    parser.add_argument('part', metavar = 'p', help = 'part number for Q2(a) or Q2(b)')
    args = parser.parse_args()

    try:
        train_df = pd.read_csv(args.train_file, header=None)
    except:
        print('Unable to read train file')
        sys.exit(1)
    try:
        test_df = pd.read_csv(args.test_file,header=None)
    except:
        print('Unable to read test file')
        sys.exit(1)

    p_type = args.binary_or_multi
    try:
        assert(p_type == 0 or p_type == 1)
    except:
        print('Error: Enter 0 or 1 for type')
        sys.exit(1)
    part = args.part


    if p_type == 0:
    #preprocessing for binary classification: (d = 8, d+1 = 9) mod 10
        X_train_bin, y_train_bin, X_val_bin, y_val_bin, X_test_bin, y_test_bin = preprocess_bin(train_df, test_df)

        if part == 'a':
            support_vectors, alpha, w, b, time1 = solve_cvx_binary_linear(X_train_bin, y_train_bin)
            _, acc_test = prediction_bin_linear(w, b, X_test_bin, y_test_bin)
            _, acc_val = prediction_bin_linear(w, b, X_val_bin, y_val_bin)

            print('\nResults for linear kernel using CVXOPT module:')
            print(f'bias, nSV: {b, len(support_vectors)}')
            print(f'Training Time: {round(time1, 3)}s')
            print(f'Validation Accuracy: {acc_val}')
            print(f'Test Accuracy: {acc_test}')


        elif part == 'b':
            support_vectors2, alpha2, b2, time2 = solve_cvx_binary_gaussian(X_train_bin, y_train_bin)
            _, _, acc_val2 = prediction_bin_gaussian(alpha2, X_train_bin, y_train_bin, b2, X_val_bin, y_val_bin)
            _, _, acc_test2 = prediction_bin_gaussian(alpha2, X_train_bin, y_train_bin, b2, X_test_bin, y_test_bin)
            print('\nResults for gaussian kernel using CVXOPT module:')
            print(f'bias, nSV: {b2, len(support_vectors2)}')
            print(f'Training Time: {round(time2, 3)}s')
            print(f'Validation Accuracy: {acc_val2}')
            print(f'Test Accuracy: {acc_test2}')
    

        elif part == 'c':
            _, b3, nSV3, acc_test3, time3 = libsvm_bin_prediction(X_train_bin, y_train_bin, X_test_bin, y_test_bin)
            _, b4, nSV4, acc_test4, time4 = libsvm_bin_prediction(X_train_bin, y_train_bin, X_test_bin, y_test_bin, gaussian = True)


            print('\nResults for LIBSVM:')
            print(f'Linear Kernel bias, nSV: {b3, nSV3}')
            print(f'Training Time for Linear Kernel: {round(time3, 3)}s')
            print(f'Test Accuracy for Linear Kernel: {acc_test3}')
            print(f'\nGaussian Kernel bias, nSV: {b4, nSV4}')
            print(f'Training Time for Gaussian Kernel: {round(time4, 3)}s')
            print(f'Test Accuracy for Gaussian Kernel: {acc_test4}')
        else:
            print('Invalid part for binary classifier in Q2')
            sys.exit(1)
    elif(p_type == 1): 
        X_train_full, y_train_full, X_test_full, y_test_full = preprocess_bin(train_df, test_df, mode = 'multi_libsvm')
        if part == 'a' or part == 'c':
            y_pred5, acc5, time5 = cvx_multi_prediction(train_df, test_df)
            print(f'Multiclass using CVXOPT:')
            print(f'Accuracy for multiclass one vs one model: {acc5}')
            print(f'Time taken for multiclass one vs one model: {round(time5, 3)}s')
            if part =='c':
                fig1, ax1 = get_confusion_matrix(y_test_full, y_pred5)
                ax1.set_title('Multiclass using CVXOPT')
                fig1.savefig('plots/multi_cvxopt_confusion.png')
                fig_v1, axs_v1 = get_confusion_images(X_test_full, y_test_full, y_pred5)
                fig_v1.suptitle('Multiclass using CVXOPT')
                fig_v1.savefig('plots/multi_cvxopt_misclassified_images.png')
        if part == 'b' or part =='c':
            y_pred6, acc6, time6 = libsvm_multi_prediction(X_train_full, y_train_full, X_test_full, y_test_full)
            print(f'Multiclass using LIBSVM:')
            print(f'Accuracy for multiclass one vs one model: {acc6}')
            print(f'Time taken for multiclass one vs one model: {round(time6, 3)}s')
            if part == 'c':
                fig2, ax2 = get_confusion_matrix(y_test_full, y_pred6)
                ax2.set_title('MultiClass using LIBSVM')
                fig2.savefig('plots/multi_libsvm_confusion.png')
                fig_v2, axs_v2 = get_confusion_images(X_test_full, y_test_full, y_pred6)
                fig_v2.suptitle('Multiclass using LIBSVM')
                fig_v2.savefig('plots/multi_libsvm_misclassified.png')
        if part == 'c':
            plt.show()

        if part == 'd':
            fig3, ax3 = k_fold_cross_validation(X_train_full, y_train_full, X_test_full, y_test_full)
            plt.show()
        if part not in ['a', 'b', 'c', 'd']:
            print('Invalid part number for multi classification')
            sys.exit(1)



























    





