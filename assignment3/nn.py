import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
from sklearn.neural_network import MLPClassifier

np.random.seed(0)

def one_hot_transform(df_train, df_test, print_df = False):
    assert(list(df_train.keys()) == list(df_test.keys()))
    df_train_y = df_train['y']
    np_train_y = df_train_y.to_numpy().reshape((-1,1))
    df_train_X = df_train.drop(columns = ['y'])

    df_test_y = df_test['y']
    np_test_y = df_test_y.to_numpy().reshape((-1,1))
    df_test_X = df_test.drop(columns = ['y'])

    ohe = OneHotEncoder()
    ohe.fit(df_train_X)
    ohe2 = OneHotEncoder()
    ohe2.fit(np_train_y)


    df_train_X_onehot = pd.DataFrame(ohe.transform(df_train_X).toarray(), columns = ohe.get_feature_names_out())
    df_test_X_onehot = pd.DataFrame(ohe.transform(df_test_X).toarray(), columns = ohe.get_feature_names_out())
    df_train_y_onehot = pd.DataFrame(ohe2.transform(np_train_y).toarray(), columns = ohe2.get_feature_names_out())
    df_test_y_onehot = pd.DataFrame(ohe2.transform(np_test_y).toarray(), columns = ohe2.get_feature_names_out())
    
    if print_df:
        print(df_train_X_onehot.info())
        print(df_train_y_onehot.info())
        print(df_test_X_onehot.info())
        print(df_test_y_onehot.info())

    return df_train_X_onehot.to_numpy(), df_train_y_onehot.to_numpy(), df_test_X_onehot.to_numpy(), df_test_y_onehot.to_numpy()

def extend(X):
    return np.vstack([np.ones((1, X.shape[1])), X])
def normalize(X):
    m = np.mean(X, axis = 0); s = np.std(X, axis = 0)
    return (X - m)/s

def sigmoid(X):
    return 1/(1 + np.exp(-X))
def relu(X):
    return X*(X>0)


class NeuralNetwork:
    '''
    M: mini-batch size
    n: number of features
    r: number of output classes
    hidden_layers: array of hidden layer sizes
    '''
    def __init__(self, M, n, hidden_layers, r, activation = 'sigmoid'):
        self.minibatch_size = M
        self.layer_sizes = [n] + hidden_layers + [r]
        self.b = [None] + [np.zeros((self.layer_sizes[l], 1)) for l in range(1, len(self.layer_sizes))]
        self.W = [None] + [np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * np.sqrt(2/(self.layer_sizes[l-1])) for l in range(1, len(self.layer_sizes))]
        #self.W = [None] + [np.zeros((self.layer_sizes[l], self.layer_sizes[l-1])) for l in range(1, len(self.layer_sizes))]
        self.X = [None] * (len(self.layer_sizes))
        self.activated_X = [None] * len(self.layer_sizes)
        self.grad = [None] + [np.zeros(self.W[l].shape) for l in range(1, len(self.layer_sizes))]
        self.delta = [None for i in range(len(self.layer_sizes))]
        self.output_final = None
        self.grad_b = [None] * (len(self.layer_sizes))
        self.activation = activation

    def forward_prop(self, X, debug = False):
        assert(X.shape[0] == self.layer_sizes[0])
        self.activated_X[0] = X
        if debug: print(f'activated_X[0]: \n{self.activated_X[0]}')
        #self.activated_X[0] = sigmoid(X)
        L = len(self.layer_sizes)
        for l in range(1, L-1):
            self.X[l] = (self.W[l] @ self.activated_X[l-1] + self.b[l])
            #assert(self.X[l].shape == (self.W[l].shape[0], self.activated_X[l-1].shape[1])); assert(self.b[l].shape[0] == self.X[l].shape[0])
            if self.activation == 'sigmoid': self.activated_X[l] = sigmoid(self.X[l])
            elif self.activation == 'relu': self.activated_X[l] = relu(self.X[l])
            else: raise ValueError("Could not recognise activation function. Enter relu or sigmoid")
            if debug: print(f'activated_X[{l}]: \n{self.activated_X[l]}')

        #Last Layer is always sigmoid
        self.X[L-1] = (self.W[L-1] @ self.activated_X[L-2] + self.b[L-1])
        self.activated_X[L-1] = sigmoid(self.X[L-1])
        
            
        self.output_final = self.activated_X[-1]
        if debug: print(f'output_final: \n{self.output_final}')
    
    def backward_prop(self, y, debug = False):
        assert(y.shape[0] == self.layer_sizes[-1])
        L = len(self.layer_sizes)
        m = self.minibatch_size

        #for sigmoid
        self.delta[L-1] = (y - self.output_final)*self.output_final*(1 - self.output_final)
        self.grad[L-1] = -(self.delta[L-1] @ self.activated_X[L-2].T)/m

        if debug: print(f'self.delta[{L-1}] \n{self.delta[L-1] }')
        if debug: print(f'self.grad[{L-1}] \n{self.grad[L-1] }')
        for l in range(L-2, 0, -1):
            if self.activation == 'sigmoid': self.delta[l] = (self.W[l+1].T @ self.delta[l+1]) * (self.activated_X[l]) * (1 - self.activated_X[l])
            elif self.activation == 'relu': self.delta[l] = (self.W[l+1].T @ self.delta[l+1]) * np.where(self.activated_X[l]>0, 1, 0)
            self.grad[l] = -(self.delta[l] @ self.activated_X[l-1].T)

            if debug: print(f'self.delta[{l}] \n{self.delta[l] }')
            if debug: print(f'self.grad[{l}] \n{self.grad[l] }')
        for l in range(L-1, 0, -1): 
            self.grad_b[l] = -np.sum(self.delta[l], axis = 1).reshape((-1, 1))
            if debug: print(f'grad_b[{l}]: \n{self.grad_b[l]}')
        
    def update_params(self, learning_rate, debug = False, adaptiveLR = False, epoch = 1):
        #assert(len(self.grad) == len(self.layer_sizes))
        for l in range(1, len(self.grad)):
            if debug:
                print(f'Before')
                print(f'self.W[{l}]: \n{self.W[l]}')
                print(f'self.b[{l}]: \n{self.b[l]}')
            if adaptiveLR:
                self.W[l] = self.W[l] - (learning_rate/np.sqrt(epoch))*self.grad[l]
                self.b[l] = self.b[l] - (learning_rate/np.sqrt(epoch))*self.grad_b[l]
            else:
                self.W[l] = self.W[l] - learning_rate*self.grad[l]
                self.b[l] = self.b[l] - learning_rate*self.grad_b[l]
            if debug:
                print(f'After')
                print(f'self.W[{l}]: \n{self.W[l]}')
                print(f'self.b[{l}]: \n{self.b[l]}')
        
    
    def test_model(self, X1, y1):
        X = (X1)
        y = y1
        assert(X.shape[0] == y.shape[0])
        m = y.shape[0]
        Xt = X.T
        
        self.forward_prop(Xt)
        y_pred = self.output_final
        y_pred1 = y_pred.T
        y1_fin = np.argmax(y, axis = 1)
        y_pred_fin = np.argmax(y_pred1, axis = 1)

        #with np.printoptions(threshold = np.inf): print(y[:10000])

        acc = (y1_fin == y_pred_fin).sum()/m
        return acc, y1_fin, y_pred_fin

    def train_model(self, X1, y1, epochs, learning_rate, thresh, adaptiveLR = False, debug = False):
        X = (X1)
        y = y1
        assert(X.shape[0] == y.shape[0])
        
        Xt1 = X.T
        yt1= y.T
        m = X.shape[0]
        M = self.minibatch_size
        n_batches = m // M
        
        acc_arr = []
        prev_sum_error = 0
        for epoch in range(epochs):
            sum_error = 0
            if debug:
                print(f'epoch: {epoch}')
            #acc_arr.append(self.test_model(X, y))
            s = np.random.permutation(m)
            Xt = Xt1#[:, s]
            yt = yt1#[:, s]
            Xt_batches = [Xt[:, i*M:(i+1)*M] for i in range(n_batches)]
            yt_batches = [yt[:, i*M:(i+1)*M] for i in range(n_batches)]
            J_sum = np.zeros(yt_batches[0].shape)
            for batch in range(n_batches):
                self.forward_prop(Xt_batches[batch], debug = debug)
                
                self.backward_prop(yt_batches[batch], debug = debug)
                if adaptiveLR: self.update_params(learning_rate, debug = debug, adaptiveLR = adaptiveLR, epoch = epoch+1)
                else: self.update_params(learning_rate, debug = debug)
                sum_error += np.sum(np.mean((yt_batches[batch] - self.output_final)**2, axis = 1), axis = 0)/2
            
            print(f'Done epoch {epoch+1}/{epochs}      ', end = '\r')

            sum_error = sum_error/n_batches

            #print(f'epoch: {epoch}: {sum_error}')
            if np.abs(prev_sum_error - sum_error) < thresh: break
            prev_sum_error = sum_error
            #print()
        print()
        #print(acc_arr)

def part_A(train_df, test_df):
    one_hot_transform(train_df, test_df, print_df = True)

def part_B(np_train_X, np_train_Y, np_test_X, np_test_Y):
    nn = NeuralNetwork(M = 100, n = np_train_X.shape[1], hidden_layers = [50], r = np_train_Y.shape[1])
    nn.train_model(np_train_X, np_train_Y, thresh = 1e-4, epochs = 1000, learning_rate = 0.1)
    acc, _, _ = nn.test_model(np_test_X, np_test_Y)
    print(f'Test Accuracy for default parameters: {acc}')

        
    
            


def part_C(np_train_X, np_train_Y, np_test_X, np_test_Y):
    hidden_arr = [5, 10, 15, 20, 25]
    train_acc_arr = []
    test_acc_arr = []

    train_time_arr = []
    for idx, i in enumerate(hidden_arr):
        nn = NeuralNetwork(M = 100, n = np_train_X.shape[1], hidden_layers = [i], r = np_train_Y.shape[1])
        t1 = time.time()
        nn.train_model(np_train_X, np_train_Y, epochs = 10000, thresh = 1e-5, learning_rate=0.1)
        t2 = time.time()
        train_time_arr.append(t2 - t1)


        train_acc, y_actual_train, y_pred_train = nn.test_model(np_train_X, np_train_Y)
        test_acc, y_actual_test, y_pred_test = nn.test_model(np_test_X, np_test_Y)
        conf_mat_test = confusion_matrix(y_actual_test, y_pred_test)
        

        fig2, ax2 = plt.subplots()
        ax2.set_title(f'Test Confusion Matrix; Hidden layer size {i}')
        sns.heatmap(conf_mat_test, annot = True, ax = ax2, fmt = 'g')
        ax2.set_xlabel('Predicted class')
        ax2.set_ylabel('Actual class')
        fig2.savefig('plots/nn_part_c_confMat_'+str(i)+'_test.png')

        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)
        print(f'Done Iteration {idx+1}/{len(hidden_arr)}')
        print(f'Training accuracy: {train_acc}')
        print(f'Test accuracy: {test_acc}')
        print(f'Time to Train: {(t2 - t1):.4f}s')

    
    fig, ax = plt.subplots()
    ax.set_xlabel('Hidden layer size')
    ax.set_ylabel('Accuracy')
    ax.plot(hidden_arr, train_acc_arr,  label = 'Training accuracy')
    ax.plot(hidden_arr, test_acc_arr, label = 'Test accuracy')
    ax.legend()
    fig.savefig('plots/nn_part_c_accuracy_plot.png')

    fig_a, ax_a = plt.subplots()
    ax_a.set_xlabel('Hidden layer size')
    ax_a.set_ylabel('Time taken for training(in s)')
    ax_a.plot(hidden_arr, train_time_arr)
    fig_a.savefig('plots/nn_part_c_time_plot.png')
    plt.show()

def part_D(np_train_X, np_train_Y, np_test_X, np_test_Y):
    hidden_arr = [5, 10, 15, 20, 25]
    train_acc_arr = []
    test_acc_arr = []

    train_time_arr = []
    for idx, i in enumerate(hidden_arr):
        nn = NeuralNetwork(M = 100, n = np_train_X.shape[1], hidden_layers = [i], r = np_train_Y.shape[1])
        t1 = time.time()
        nn.train_model(np_train_X, np_train_Y, epochs = 10000, thresh = 1e-5, learning_rate=0.1, adaptiveLR=True)
        t2 = time.time()
        train_time_arr.append(t2 - t1)


        train_acc, y_actual_train, y_pred_train = nn.test_model(np_train_X, np_train_Y)
        test_acc, y_actual_test, y_pred_test = nn.test_model(np_test_X, np_test_Y)
        conf_mat_test = confusion_matrix(y_actual_test, y_pred_test)
        

        fig2, ax2 = plt.subplots()
        ax2.set_title(f'Test Confusion Matrix; Hidden layer size {i}')
        sns.heatmap(conf_mat_test, annot = True, ax = ax2, fmt = 'g')
        ax2.set_xlabel('Predicted class')
        ax2.set_ylabel('Actual class')
        fig2.savefig('plots/nn_part_d_confMat_'+str(i)+'_test.png')

        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)
        print(f'Done Iteration {idx+1}/{len(hidden_arr)}')
        print(f'Training accuracy: {train_acc}')
        print(f'Test accuracy: {test_acc}')
        print(f'Time to Train: {(t2 - t1):.4f}s')

    
    fig, ax = plt.subplots()
    ax.set_xlabel('Hidden layer size')
    ax.set_ylabel('Accuracy')
    ax.plot(hidden_arr, train_acc_arr,  label = 'Training accuracy')
    ax.plot(hidden_arr, test_acc_arr, label = 'Test accuracy')
    ax.legend()
    fig.savefig('plots/nn_part_d_accuracy_plot.png')

    fig_a, ax_a = plt.subplots()
    ax_a.set_xlabel('Hidden layer size')
    ax_a.set_ylabel('Time taken for training(in s)')
    ax_a.plot(hidden_arr, train_time_arr)
    fig_a.savefig('plots/nn_part_d_time_plot.png')



def part_E(np_train_X, np_train_Y, np_test_X, np_test_Y):

    ## relu with [100, 100]
    nn1 = NeuralNetwork(M = 100, n = np_train_X.shape[1], hidden_layers = [100, 100], r = np_train_Y.shape[1], activation = 'relu')
    t1 = time.time()
    print('Training relu model with hidden layers [100, 100]')
    nn1.train_model(np_train_X, np_train_Y, epochs = 10000, thresh = 1e-6, learning_rate = 0.1, adaptiveLR=True)
    t2 = time.time()
    print(f'Time to train: {(t2 - t1):.4f}s')
    train_acc1, _, _ = nn1.test_model(np_train_X, np_train_Y)
    test_acc1, y_actual_test1, y_pred_test1 = nn1.test_model(np_test_X, np_test_Y)

    print(f'Training Accuracy: {train_acc1}')
    print(f'Test Accuracy: {test_acc1}')
    conf_mat_test1 = confusion_matrix(y_actual_test1, y_pred_test1)
    fig1, ax1 = plt.subplots()
    ax1.set_title(f'Test Confusion Matrix for relu; Hidden layer sizes [100, 100]')
    sns.heatmap(conf_mat_test1, annot = True, ax = ax1, fmt = 'g')
    ax1.set_xlabel('Predicted class')
    ax1.set_ylabel('Actual class')
    fig1.savefig('plots/nn_part_e_confMat_Relu_100_100_test.png')

    ##sigmoid with [100, 100]
    nn1 = NeuralNetwork(M = 100, n = np_train_X.shape[1], hidden_layers = [100, 100], r = np_train_Y.shape[1], activation = 'sigmoid')
    t1 = time.time()
    print('Training sigmoid model with hidden layers [100, 100]')
    nn1.train_model(np_train_X, np_train_Y, epochs = 10000, thresh = 1e-6, learning_rate = 0.1, adaptiveLR=True)
    t2 = time.time()
    print(f'Time to train: {(t2 - t1):.4f}s')
    train_acc1, _, _ = nn1.test_model(np_train_X, np_train_Y)
    test_acc1, y_actual_test1, y_pred_test1 = nn1.test_model(np_test_X, np_test_Y)

    print(f'Training Accuracy: {train_acc1}')
    print(f'Test Accuracy: {test_acc1}')
    conf_mat_test2 = confusion_matrix(y_actual_test1, y_pred_test1)
    fig2, ax2 = plt.subplots()
    ax2.set_title(f'Test Confusion Matrix for sigmoid; Hidden layer sizes [100, 100]')
    sns.heatmap(conf_mat_test2, annot = True, ax = ax2, fmt = 'g')
    ax2.set_xlabel('Predicted class')
    ax2.set_ylabel('Actual class')
    fig2.savefig('plots/nn_part_e_confMat_Sigmoid_100_100_test.png')

    ##relu with [50]
    nn1 = NeuralNetwork(M = 100, n = np_train_X.shape[1], hidden_layers = [50], r = np_train_Y.shape[1], activation = 'relu')
    t1 = time.time()
    print('Training relu model with hidden layers [50]')
    nn1.train_model(np_train_X, np_train_Y, epochs = 10000, thresh = 1e-6, learning_rate = 0.1, adaptiveLR=True)
    t2 = time.time()
    print(f'Time to train: {(t2 - t1):.4f}s')
    train_acc1, _, _ = nn1.test_model(np_train_X, np_train_Y)
    test_acc1, y_actual_test1, y_pred_test1 = nn1.test_model(np_test_X, np_test_Y)

    print(f'Training Accuracy: {train_acc1}')
    print(f'Test Accuracy: {test_acc1}')

    conf_mat_test3 = confusion_matrix(y_actual_test1, y_pred_test1)
    fig3, ax3 = plt.subplots()
    ax3.set_title(f'Test Confusion Matrix for relu; Hidden layer sizes [50]')
    sns.heatmap(conf_mat_test3, annot = True, ax = ax3, fmt = 'g')
    ax2.set_xlabel('Predicted class')
    ax2.set_ylabel('Actual class')
    fig3.savefig('plots/nn_part_e_confMat_Relu_50_test.png')

    ##sigmoid with [50]
    nn1 = NeuralNetwork(M = 100, n = np_train_X.shape[1], hidden_layers = [50], r = np_train_Y.shape[1], activation = 'sigmoid')
    t1 = time.time()
    print('Training sigmoid model with hidden layers [50]')
    nn1.train_model(np_train_X, np_train_Y, epochs = 10000, thresh = 1e-6, learning_rate = 0.1, adaptiveLR=True)
    t2 = time.time()
    print(f'Time to train: {(t2 - t1):.4f}s')
    train_acc1, _, _ = nn1.test_model(np_train_X, np_train_Y)
    test_acc1, y_actual_test1, y_pred_test1 = nn1.test_model(np_test_X, np_test_Y)

    print(f'Training Accuracy: {train_acc1}')
    print(f'Test Accuracy: {test_acc1}')

    conf_mat_test4 = confusion_matrix(y_actual_test1, y_pred_test1)
    fig4, ax4 = plt.subplots()
    ax4.set_title(f'Test Confusion Matrix for sigmoid; Hidden layer sizes [50]')
    sns.heatmap(conf_mat_test4, annot = True, ax = ax4, fmt = 'g')
    ax4.set_xlabel('Predicted class')
    ax4.set_ylabel('Actual class')
    fig4.savefig('plots/nn_part_e_confMat_Sigmoid_50_test.png')

    plt.show()

def part_F(np_train_X, np_train_Y, np_test_X, np_test_Y):
    clf = MLPClassifier(hidden_layer_sizes = (100, 100, ), solver = 'sgd', activation = 'relu', learning_rate = 'adaptive', learning_rate_init=0.1)
    print('Training relu MLP Classifier for hidden layers [100, 100]...')
    t1 = time.time()
    clf.fit(np_train_X, np_train_Y)
    t2 = time.time()
    print(f'Time to train: {(t2 - t1):.4f}s')
    acc = clf.score(np_test_X, np_test_Y)
    print(f'Test accuracy Score: {acc}')
    



















        



if __name__ == '__main__':
    cols = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'y']
    input_train_file = sys.argv[1]
    input_test_file = sys.argv[2]
    
    try:
        df_train = pd.read_csv(input_train_file, sep = ',', names = cols)
    except:
        print('Could not read/parse training file')
    try:
        df_test = pd.read_csv(input_test_file, sep=',', names = cols)
    except:
        print('Could not read/parse test file')

    np_train_X, np_train_Y, np_test_X, np_test_Y = one_hot_transform(df_train, df_test)
    assert(np_train_X.shape[1] == np_test_X.shape[1])
    assert(np_train_Y.shape[1] == np_test_Y.shape[1])

    part = sys.argv[3]
    if part not in ['a', 'b', 'c', 'd', 'e', 'f']:
        print('Part value must be among c,d,e,f')
        sys.exit(1)
    if part == 'a':
        part_A(df_train, df_test)
    elif(part == 'b'):
        part_B(np_train_X, np_train_Y, np_test_X, np_test_Y)
    elif part == 'c':
        part_C(np_train_X, np_train_Y, np_test_X, np_test_Y)
    elif part == 'd':
        part_D(np_train_X, np_train_Y, np_test_X, np_test_Y)
    elif part == 'e':
        part_E(np_train_X, np_train_Y, np_test_X, np_test_Y)
    elif part == 'f':
        part_F(np_train_X, np_train_Y, np_test_X, np_test_Y)

    #part_E(np_train_X, np_train_Y, np_test_X, np_test_Y)



    


