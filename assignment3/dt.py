import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from collections import deque

'''
Categories:
age: N
job: Y
marital: Y
education: Y
default: Y
balance: N
housing: Y
loan: Y
contact: Y
day: N?
month: Y
duration: N
campaign: N
pdays: N
previous: N
poutcome: Y

y:binary


'''
def breakpt():
    print('BOOOOO')
    sys.exit()

def preprocess(df):
    df2 = df.copy()
    df2['y'] = df2['y'].map({'yes':1, 'no':0})
    return df2
def bin_entropy(m0, m1):
    if(m0==0 or m1==0):
        return 0
    else:
        p0 = m0/(m0+m1)
        p1 = m1/(m0+m1)
        return -(p0*np.log(p0) + p1*np.log(p1))
def bool_categorical(df):
    l1 = []
    c = set(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])
    n = set(['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'])
    for idx, i in enumerate(df.keys()):
        if i == 'y':
            l1.append(None)
        if i in n:
            l1.append(False)
        else:
            l1.append(True)
    return l1
def bool_categorical_dict(df):
    d1 = {}
    c = set(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])
    n = set(['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'])
    for i, k in enumerate(df.keys()):
        if k == 'y':
            d1[k] = None
        elif k in n:
            d1[k] = False
        else:
            d1[k] = True
    return d1

def transform_oneHot(df):
    new_df = df.copy()
    y = df['y']
    new_df = new_df.drop(columns = 'y')
    b1 = bool_categorical_dict(df)
    for i, k in enumerate(b1.keys()):
        if b1[k]:
            vals = pd.unique(df[k])
            new_df = new_df.drop(k, 1)
            for i1, val in enumerate(list(vals)):
                new_df[str(k)+'_'+str(val)] = (df[k] == val)
    new_df = new_df.sort_index(axis = 1)
    new_df['y'] = y
    return new_df

def extend_oneHot(df, train_df):
    y1 = df['y']
    y2 = train_df['y']
    k0 = df.keys()[0]
    F_arr = df[k0] != df[k0]
    for k in train_df.keys():
        if k not in df.keys():
            df[k] = F_arr
    df1 = df.sort_index(axis=1)
    df2 = train_df.sort_index(axis=1)
    assert(list(df1.keys()) == list(df2.keys()))
    df1.drop(columns = 'y')
    df2.drop(columns = 'y')
    df1['y'] = y1
    df2['y'] = y2


    return df1

def get_sklearn_encodings(df):
    c = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    df_new = df.copy()
    label_encoders = {}
    for k in df.keys():
        le = LabelEncoder()
        le.fit(df[k])
        label_encoders[k] = le
        df_new[k] = le.transform(df[k])
    df_rem = df_new.drop(columns = c)
    df_cat = df_new[c]
    oneHotEnc = OneHotEncoder()
    oneHotEnc.fit(df_cat)
    df_oneHot_cat = pd.DataFrame(oneHotEnc.transform(df_cat).toarray(), columns = oneHotEnc.get_feature_names_out())
    df_fin = pd.concat([df_oneHot_cat, df_rem], axis = 1)
    return df_fin, label_encoders, oneHotEnc

def transform_sklearn(df, label_encoders, oneHotEnc):
    c = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    assert(list(label_encoders.keys()) == list(df.keys()))
    df_new = df.copy()
    for k in df_new.keys():
        if k in c:
            le = label_encoders[k]
            df_new[k] = le.transform(df_new[k])
    df_rem = df_new.drop(columns = c)
    df_cat = df_new[c]
    df_oneHot_cat = pd.DataFrame(oneHotEnc.transform(df_cat).toarray(), columns = oneHotEnc.get_feature_names_out())
    df_fin = pd.concat([df_oneHot_cat, df_rem], axis = 1)

    return df_fin

    



            

class DT_Node:
    def __init__(self, data, attrs, col_types_bool_arr):
        self.prune_val_data = None
        self.parent = None
        self.best_attr = None
        self.isNumeric = False
        self.median = None
        self.col_types = col_types_bool_arr     #True for categorical, False for numerical
        self.attrs = attrs
        self.data = data
        self.children = {}
        self.isLeaf = True
        #print(data.shape[0], len(list(self.attrs)))
        m1 = (data[:, -1] == 1).sum()
        m0 = (data[:, -1] == 0).sum()
        self.leafVal = 1 if (m1>=m0) else 0

    def __str__(self):
        print('Node Information:')
        print(self.data[:, -1])
        print(f'best_attr {self.best_attr}')
        print(f'self.isNumeric')
        print(f'attrs {self.attrs}')
        print(f'self.isLeaf {self.isLeaf}')
        print(f'self.leafVal {self.leafVal}')
        print(f'self.children.keys {self.children.keys()}')
        return ''
    def expand(self):
        if self.data.shape[0] == 0:
            self.leafVal = np.random.randint(2)
            return
        self.isLeaf = False
        m = self.data.shape[0]
        req_data = self.data
        y_req = req_data[:, -1]
        m0 = (y_req == 0).sum()
        m1 = (y_req == 1).sum()
        if(m0 == m):
            self.isLeaf = True
            self.leafVal = 0
            return
        elif(m1 == m):
            self.isLeaf = True
            self.leafVal = 1
            return
        elif(len(self.attrs) == 0):
            self.isLeaf = True
            if(m0>m1):
                self.leafVal = 0
            else: self.leafVal = 1
            return
        entropy = bin_entropy(m0, m1)
        
        min_sum_entropy = np.inf
        best_attr = -1
        best_attr_new_rows = {}
        for idx, attr in enumerate(self.attrs):
            attr_data = {}
            sum_entropy = 0
            if self.col_types[attr] == None:
                continue
            if self.col_types[attr]:
                column_arr = req_data[:, attr]
                vals = np.unique(column_arr)
                for val in vals:
                    rows = column_arr == val
                    attr_data[val] = req_data[rows, :]
                    mc = rows.shape[0]
                    mc0 = attr_data[val].shape[0]
                    mc1 = mc - mc0
                    sum_entropy+=(mc/m)*bin_entropy(mc0, mc1)
                if sum_entropy<min_sum_entropy:
                    self.isNumeric = False
                    min_sum_entropy = sum_entropy
                    best_attr = attr
                    best_attr_new_rows = attr_data
            
            else:
                column_arr = req_data[:, attr]
                median = np.median(column_arr)
                rows1 = column_arr < median
                rows2 = column_arr >= median
                
                data1 = req_data[rows1, :]
                data2 = req_data[rows2, :]
                
                
                attr_data['lt'] = data1
                attr_data['ge'] = data2

                m_data1 = data1.shape[0]
                m_data1_0 = (data1[:, -1] == 0).sum()
                m_data1_1 = m_data1 - m_data1_0
                sum_entropy+=(m_data1/m)*bin_entropy(m_data1_0, m_data1_1)

                m_data2 = data2.shape[0]
                m_data2_0 = (data2[:, -1] == 0).sum()
                m_data2_1 = m_data2 - m_data2_0
                sum_entropy+=(m_data2/m)*bin_entropy(m_data2_0, m_data2_1)
                if sum_entropy<min_sum_entropy:
                    self.median = median
                    self.isNumeric = True
                    min_sum_entropy = sum_entropy
                    best_attr = attr
                    best_attr_new_rows = attr_data

                
        
        new_attrs = list(self.attrs).copy()
        new_attrs.remove(best_attr)
        self.best_attr = best_attr
        for k in (best_attr_new_rows.keys()):
            newnode = DT_Node(best_attr_new_rows[k], new_attrs, self.col_types)
            self.children[k] = newnode
            newnode.parent = self
        

    

            
            
                    



class Tree:
    def __init__(self, train_df, test_df, val_df, oneHot = False):
        #attrs = np.array(range(data.shape[1] - 1))
        self.oneHot = False
        if oneHot:
            self.oneHot = True
            train_df = transform_oneHot(train_df)
            val_df = transform_oneHot(val_df)
            val_df = extend_oneHot(val_df, train_df)
            test_df = transform_oneHot(test_df)
            test_df = extend_oneHot(test_df, train_df)
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        #print(train_df.info())
        self.data = self.train_df.to_numpy()
        self.test_data = self.test_df.to_numpy()
        self.val_data = self.val_df.to_numpy()
        #self.cols = list(train_df.keys())
        col_types = bool_categorical(train_df)
        rows = (train_df[train_df.keys()[0]]!=None)
        attrs = [i for i, k in enumerate(train_df.keys()) if k!='y']
        attr_vals = [k for k in train_df.keys() if k!='y']
        #print(f'attr_vals: {attr_vals}')
        assert(list(train_df.keys())[-1] == 'y' )
        self.root = DT_Node(self.data, attrs, col_types)
        self.nodeQueue = deque()
        self.nodeQueue.append(self.root)
        self.treeSize = 1
        self.val_acc = None
        self.prune_test_acc_arr = []
        self.prune_train_acc_arr = []
        self.prune_val_acc_arr = []
        self.prune_tree_size_arr = []
        self.prune_counter = 0

    def checkModel(self, node_train_data, node_test_data, node_val_data, node):
        if node.isLeaf:
            req_rows_train_y = node_train_data[:, -1]
            req_rows_test_y = node_test_data[:, -1]
            req_rows_val_y = node_val_data[:, -1]

            m_train = (req_rows_train_y == node.leafVal).sum()
            m_test = (req_rows_test_y == node.leafVal).sum()
            m_val = (req_rows_val_y == node.leafVal).sum()

            return [m_train, m_test, m_val]

        else:
            if node.isNumeric:
                median = node.median
                attr = node.best_attr
                train_lt = node_train_data[node_train_data[:, attr] < median, :]
                train_ge = node_train_data[node_train_data[:, attr] >= median, :]
                test_lt = node_test_data[node_test_data[:, attr] < median, :]
                test_ge = node_test_data[node_test_data[:, attr] >= median, :]
                val_lt = node_val_data[node_val_data[:, attr] < median, :]
                val_ge = node_val_data[node_val_data[:, attr] >= median, :]

                m1 = self.checkModel(train_lt, test_lt, val_lt, node.children['lt'])
                m2 = self.checkModel(train_ge, test_ge, val_ge, node.children['ge'])


                return [m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2]]
            else:
                attr = node.best_attr
                train_dict = {}
                test_dict = {}
                val_dict = {}

                sum = [0, 0, 0]
                for k in node.children.keys():
                    train_dict[k] = node_train_data[node_train_data[:, attr] == k, :]
                    test_dict[k] = node_test_data[node_test_data[:, attr] == k, :]
                    val_dict[k] = node_val_data[node_val_data[:, attr] == k, :]
                    m = self.checkModel(train_dict[k], test_dict[k], val_dict[k], node.children[k])
                    sum[0]+=m[0]
                    sum[1]+=m[1]
                    sum[2]+=m[2]
                return sum




                


        

    def growTreeOnce(self):
        if not self.nodeQueue:
            return False
        node = self.nodeQueue.popleft()
        node.expand()
        while(len(node.children) == 0 and self.nodeQueue):
            node = self.nodeQueue.popleft()
            node.expand()
        self.treeSize += len(node.children)
        for c in node.children:
            self.nodeQueue.append(node.children[c])
        if not self.nodeQueue: return False
        return True
    
    def growTree(self, maxIter = 1e5, checkInterval = 100):
        train_acc = []
        test_acc = []
        val_acc = []
        tree_size_arr = []

        counter = 0
        while(self.growTreeOnce() and counter < maxIter):
            counter+=1
            print(f'Iteration: {counter}', end='\r')
            #if not counter:
                #print(self.root)
                #counter-=1

            if counter%checkInterval == 0:
                res = self.checkModel(self.data, self.test_data, self.val_data, self.root)
                train_acc.append(res[0]/self.data.shape[0])
                test_acc.append(res[1]/self.test_data.shape[0])
                val_acc.append(res[2]/self.val_data.shape[0])
                tree_size_arr.append(self.treeSize)
        res = self.checkModel(self.data, self.test_data, self.val_data, self.root)
        train_acc.append(res[0]/self.data.shape[0])
        test_acc.append(res[1]/self.test_data.shape[0])
        val_acc.append(res[2]/self.val_data.shape[0])
        self.val_acc = val_acc
        tree_size_arr.append(self.treeSize)


        return train_acc, test_acc, val_acc, tree_size_arr

    def set_prune_data(self, node, data):
        node.prune_val_data = data
        if node.isLeaf:
            return
        attr = node.best_attr
        if node.isNumeric:
            median = node.median
            node1 = node.children['lt']
            node2 = node.children['ge']
            data1 = data[data[:, attr] < median, :]
            data2 = data[data[:, attr] >= median, :]
            self.set_prune_data(node1, data1)
            self.set_prune_data(node2, data2)
        else:
            for c in node.children:
                data_new = data[data[:, attr] == c, :]
                self.set_prune_data(node.children[c], data_new)

    def post_pruning(self, node, sampleInterval = 10):
        return_val = None
        if node.isLeaf:
            m = (node.prune_val_data[:, -1] == node.leafVal).sum()
            return_val = m
        else:
            sum = 0
            for c in node.children:
                node_c = node.children[c]
                res = self.post_pruning(node_c)
                #sum+=res
                sum += res
            arr0 = (node.prune_val_data[:, -1] == 0)
            arr1 = (node.prune_val_data[:, -1] == 1)
            m0 = int(arr0.sum()) if arr0.size>0 else 0
            m1 = int(arr1.sum()) if arr1.size>0 else 0
            if max(m0, m1) > sum:
                node.isLeaf = True
                remove_cnt = len(node.children)
                self.treeSize -= remove_cnt
                node.children = []
                if m0>m1: node.leafVal = 0
                else: node.leafVal = 1
                self.prune_counter+=1
                print(f'Prune Iteration: {self.prune_counter}', end='\r')
                if self.prune_counter%sampleInterval == 0:
                    acc1, acc2, acc3 = self.checkModel(self.data, self.test_data, self.val_data, self.root)
                    train_acc = acc1/self.data.shape[0]
                    test_acc = acc2/self.test_data.shape[0]
                    val_acc = acc3/self.val_data.shape[0]
                    self.prune_train_acc_arr.append(train_acc)
                    self.prune_test_acc_arr.append(test_acc)
                    self.prune_val_acc_arr.append(val_acc)
                    self.prune_tree_size_arr.append(self.treeSize)
                return_val =  max(m0, m1)
            else:
                return_val =  sum
        
        return return_val
    
    def prune_tree(self, sampleInterval = 10):
        self.set_prune_data(self.root, self.val_data)
        self.post_pruning(self.root, sampleInterval = sampleInterval)
        self.prune_counter = 0


            
            
        



def part_a(df_train, df_test, df_val, oneHot = False):
    df_train = preprocess(df_train)
    df_test = preprocess(df_test)
    df_val = preprocess(df_val)
    assert(list(df_train.keys()) == list(df_test.keys()) == list(df_val.keys()))

    print(f'Using Multiple values per class')
    decision_Tree = Tree(df_train, df_test, df_val, oneHot=False)
    train_acc, test_acc, val_acc, tree_size_arr = decision_Tree.growTree()
    root = decision_Tree.root
    #print(root)

    print(f'Final Train Accuracy: {train_acc[-1]}')
    print(f'Final Validation Accuracy: {val_acc[-1]}')
    print(f'Final Test Accuracy: {test_acc[-1]}')

    fig, ax = plt.subplots()
    ax.plot(tree_size_arr, train_acc, label = 'Training accuracy')
    ax.plot(tree_size_arr, test_acc, label = 'Test Accuracy')
    ax.plot(tree_size_arr, val_acc, label = 'Validation Accuracy')
    ax.set_xlabel('Number of nodes in Tree')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Tree Nodes plot')
    ax.legend()
    fig.savefig('plots/dt_growth_multi.png')



    ##With oneHot


    print(f'Using OneHot Encoding')
    decision_Tree = Tree(df_train, df_test, df_val, oneHot=True)
    train_acc, test_acc, val_acc, tree_size_arr = decision_Tree.growTree()
    root = decision_Tree.root
    #print(root)

    print(f'Final Train Accuracy: {train_acc[-1]}')
    print(f'Final Validation Accuracy: {val_acc[-1]}')
    print(f'Final Test Accuracy: {test_acc[-1]}')

    fig2, ax2 = plt.subplots()
    ax2.plot(tree_size_arr, train_acc, label = 'Training accuracy')
    ax2.plot(tree_size_arr, test_acc, label = 'Test Accuracy')
    ax2.plot(tree_size_arr, val_acc, label = 'Validation Accuracy')
    ax2.set_xlabel('Number of nodes in Tree')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Tree Nodes plot')
    ax2.legend()
    fig2.savefig('plots/dt_growth_onehot.png')

    return fig, ax, fig2, ax2


    #add to partB
    '''
    if pruning:
        decision_Tree.prune_tree()
        train_acc_arr = decision_Tree.prune_train_acc_arr
        test_acc_arr = decision_Tree.prune_test_acc_arr
        val_acc_arr = decision_Tree.prune_val_acc_arr
        tree_size_arr_prune = decision_Tree.prune_tree_size_arr
        print(train_acc_arr)
        print(test_acc_arr)
        print(val_acc_arr)
        print(tree_size_arr_prune)
        

        
        print('\nAfter pruning')
        print(f'Final Train Accuracy: {train_acc_arr[-1]}')
        print(f'Final Validation Accuracy: {val_acc_arr[-1]}')
        print(f'Final Test Accuracy: {test_acc_arr[-1]}')
    '''


def part_b(df_train, df_test, df_val, oneHot = True):
    df_train = preprocess(df_train)
    df_test = preprocess(df_test)
    df_val = preprocess(df_val)
    assert(list(df_train.keys()) == list(df_test.keys()) == list(df_val.keys()))

    
    decision_Tree = Tree(df_train, df_test, df_val, oneHot=oneHot)
    train_acc, test_acc, val_acc, tree_size_arr = decision_Tree.growTree()
    root = decision_Tree.root

    fig, ax = plt.subplots()
    ax.plot(tree_size_arr, train_acc, label = 'Training accuracy')
    ax.plot(tree_size_arr, test_acc, label = 'Test Accuracy')
    ax.plot(tree_size_arr, val_acc, label = 'Validation Accuracy')
    ax.set_xlabel('Number of nodes in Tree')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Tree Nodes plot')
    ax.legend()
    fig.savefig('plots/dt_growth.png')

    #print(root)

    print(f'Train Accuracy before pruning: {train_acc[-1]}')
    print(f'Validation Accuracy before pruning: {val_acc[-1]}')
    print(f'Test Accuracy before pruning: {test_acc[-1]}')

    #add to partB
    decision_Tree.prune_tree()
    train_acc_arr = decision_Tree.prune_train_acc_arr
    test_acc_arr = decision_Tree.prune_test_acc_arr
    val_acc_arr = decision_Tree.prune_val_acc_arr
    tree_size_arr = decision_Tree.prune_tree_size_arr

    fig2, ax2 = plt.subplots()
    ax2.plot(tree_size_arr, train_acc_arr, label = 'Training accuracy')
    ax2.plot(tree_size_arr, test_acc_arr, label = 'Test Accuracy')
    ax2.plot(tree_size_arr, val_acc_arr, label = 'Validation Accuracy')
    ax2.set_xlabel('Number of nodes in Tree(while pruning)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Tree Nodes plot for pruning process')
    ax2.legend()
    fig2.savefig('plots/dt_pruning.png')

    

    
    print('\nAfter pruning')
    print(f'Final Train Accuracy: {train_acc_arr[-1]}')
    print(f'Final Validation Accuracy: {val_acc_arr[-1]}')
    print(f'Final Test Accuracy: {test_acc_arr[-1]}')

    return fig, ax, fig2, ax2
    




def part_c(df_train, df_test, df_val):
    df_train_X = df_train.drop(columns = ['y'])
    df_train_X2, labelEncs, oneHotEnc = get_sklearn_encodings(df_train_X)
    df_train_y = df_train['y'].map({'yes':1, 'no':0})

    df_test_X = df_test.drop(columns = ['y'])
    df_test_X2 = transform_sklearn(df_test_X, labelEncs, oneHotEnc)
    df_test_y = df_test['y'].map({'yes':1, 'no':0})

    df_val_X = df_val.drop(columns = ['y'])
    df_val_X2 = transform_sklearn(df_val_X, labelEncs, oneHotEnc)
    df_val_y = df_val['y'].map({'yes':1, 'no':0})

    n_estimators_arr = [i for i in range(50, 500, 100)]
    max_features_arr = [i/10 for i in range(1, 10, 2)]
    min_samples_split_arr = [i for i in range(2, 11, 2)]

    assert(list(df_train_X2.keys()) == list(df_test_X2.keys()) == list(df_val_X2.keys()))
    best_oob_score = -1
    best_rfc = None
    best_n_estimators =  -1
    best_max_features = -1
    best_min_samples_split = -1
    for i in n_estimators_arr:
        for j in max_features_arr:
            for k in min_samples_split_arr:
                
                rfc1 = RandomForestClassifier(n_estimators = i, max_features = j, min_samples_split=k, bootstrap=True, oob_score=True, n_jobs = -1)
                rfc1.fit(df_train_X2.to_numpy(), df_train_y.to_numpy())
                if rfc1.oob_score_ > best_oob_score:
                    best_oob_score = rfc1.oob_score_
                    best_rfc = rfc1
                    best_n_estimators = i
                    best_max_features = j
                    best_min_samples_split = k
                print(f'DONE n_estimators: {i}, max_features: {j}, min_samples_split: {k}')
    


    print(f'Best Parameters: n_estimators: {best_n_estimators}, max_features: {best_max_features}, min_samples_split: {best_min_samples_split}')
    print(f'Best Parameter OOB Score: {best_oob_score}')
    print(f'Training Accuracy: {best_rfc.score(df_train_X2.to_numpy(), df_train_y.to_numpy())}')
    print(f'Test Accuracy: {best_rfc.score(df_test_X2.to_numpy(), df_test_y.to_numpy())}')
    print(f'Validation Accuracy: {best_rfc.score(df_val_X2.to_numpy(), df_val_y.to_numpy())}')

    return best_n_estimators, best_max_features, best_min_samples_split

def part_d(df_train, df_test, df_val, reCompute = False):
    df_train_X = df_train.drop(columns = ['y'])
    df_train_X2, labelEncs, oneHotEnc = get_sklearn_encodings(df_train_X)
    df_train_y = df_train['y'].map({'yes':1, 'no':0})

    df_test_X = df_test.drop(columns = ['y'])
    df_test_X2 = transform_sklearn(df_test_X, labelEncs, oneHotEnc)
    df_test_y = df_test['y'].map({'yes':1, 'no':0})

    df_val_X = df_val.drop(columns = ['y'])
    df_val_X2 = transform_sklearn(df_val_X, labelEncs, oneHotEnc)
    df_val_y = df_val['y'].map({'yes':1, 'no':0})
    assert(list(df_train_X2.keys()) == list(df_test_X2.keys()) == list(df_val_X2.keys()))

    np_train_X2 = df_train_X2.to_numpy()
    np_test_X2 = df_test_X2.to_numpy()
    np_val_X2 = df_val_X2.to_numpy()
    np_train_y = df_train_y.to_numpy()
    np_test_y = df_test_y.to_numpy()
    np_val_y = df_val_y.to_numpy()


    best_ne, best_mf, best_mss = 350, 0.5, 10
    if reCompute:
        best_ne, best_mf, best_mss = part_c(df_train, df_val, df_test)
    
    ne_arr = [i for i in range(50, 500, 100)]
    oob_score_ne = []
    train_acc_ne = []
    test_acc_ne = []
    val_acc_ne = []

    print('Parameter Sensitivity Analysis:')
    for ne in ne_arr:
        print(f'DONE n_estimators: {ne}, max_features: {best_mf}, min_samples_split: {best_mss}')
        rfc = RandomForestClassifier(n_estimators = ne, max_features=best_mf, min_samples_split=best_mss, bootstrap=True, oob_score=True, n_jobs = -1)
        rfc.fit(np_train_X2, np_train_y)
        oob_score_ne.append(rfc.oob_score_)
        train_acc_ne.append(rfc.score(np_train_X2, np_train_y))
        test_acc_ne.append(rfc.score(np_test_X2, np_test_y))
        val_acc_ne.append(rfc.score(np_val_X2, np_val_y))

    mf_arr = [i/10 for i in range(1, 10, 2)]
    oob_score_mf = []
    train_acc_mf = []
    test_acc_mf = []
    val_acc_mf = []

    for mf in mf_arr:
        print(f'DONE n_estimators: {best_ne}, max_features: {mf}, min_samples_split: {best_mss}')
        rfc = RandomForestClassifier(n_estimators = best_ne, max_features=mf, min_samples_split=best_mss, bootstrap=True, oob_score=True, n_jobs = -1)
        rfc.fit(np_train_X2, np_train_y)
        oob_score_mf.append(rfc.oob_score_)
        train_acc_mf.append(rfc.score(np_train_X2, np_train_y))
        test_acc_mf.append(rfc.score(np_test_X2, np_test_y))
        val_acc_mf.append(rfc.score(np_val_X2, np_val_y))

    mss_arr = [i for i in range(2, 11, 2)]
    oob_score_mss = []
    train_acc_mss = []
    test_acc_mss = []
    val_acc_mss = []

    for mss in mss_arr:
        print(f'DONE n_estimators: {best_ne}, max_features: {best_mf}, min_samples_split: {mss}')
        rfc = RandomForestClassifier(n_estimators = best_ne, max_features=best_mf, min_samples_split=mss, bootstrap=True, oob_score=True, n_jobs = -1)
        rfc.fit(np_train_X2, np_train_y)
        oob_score_mss.append(rfc.oob_score_)
        train_acc_mss.append(rfc.score(np_train_X2, np_train_y))
        test_acc_mss.append(rfc.score(np_test_X2, np_test_y))
        val_acc_mss.append(rfc.score(np_val_X2, np_val_y))

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('n_estimators')
    ax1.set_ylabel('accuracy')
    ax1.set_title('Varying parameter n_estimators')
    ax1.plot(ne_arr, test_acc_ne, label = 'Test Accuracy')
    ax1.plot(ne_arr, val_acc_ne, label = 'Validation Accuracy')
    ax1.plot(ne_arr, oob_score_ne, label = 'OOB score')
    ax1.legend()
    fig1.savefig('plots/n_estimators_sensitivity.png')

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('max_features')
    ax2.set_ylabel('accuracy')
    ax2.set_title('Varying parameter max_features')
    ax2.plot(mf_arr, test_acc_mf, label = 'Test Accuracy')
    ax2.plot(mf_arr, val_acc_mf, label = 'Validation Accuracy')
    ax2.plot(mf_arr, oob_score_mf, label = 'OOB score')
    ax2.legend()
    fig2.savefig('plots/max_features_sensitivity.png')

    fig3, ax3 = plt.subplots()
    ax3.set_xlabel('min_samples_split')
    ax3.set_ylabel('accuracy')
    ax3.set_title('Varying parameter min_samples_split')
    ax3.plot(mss_arr, test_acc_mss, label = 'Test Accuracy')
    ax3.plot(mss_arr, val_acc_mss, label = 'Validation Accuracy')
    ax3.plot(mss_arr, oob_score_mss, label = 'OOB score')
    ax3.legend()
    fig3.savefig('plots/min_samples_split_sensitivity.png')

    return fig1, ax1, fig2, ax2, fig3, ax3



    




    



if __name__ == '__main__':
    help_str = 'Help: python3 dt.py <train_file> <test_file> <validation_file> <part (one of {a,b,c,d})>' 
    try:
        train_file = sys.argv[1]
        test_file = sys.argv[2]
        val_file = sys.argv[3]
        part = sys.argv[4]
    except:
        print('Please provide correct input format')
        print(help_str)
        sys.exit(1)
    

    try:
        df_train = pd.read_csv(train_file, sep=';')
    except:
        print('Could not read/parse training file')
        sys.exit(1)
    try:
        df_test = pd.read_csv(test_file, sep=';')
    except:
        print('Could not read/parse test file')
        sys.exit(1)
    try:
        df_val = pd.read_csv(val_file, sep=';')
    except:
        print('Could not read/parse validation file')
        sys.exit(1)

    assert(list(df_train.keys()) == list(df_test.keys()) == list(df_val.keys()))

    if(part not in ['a', 'b', 'c', 'd']):
        print('part value must be a,b,c or d')
        sys.exit(1)
    
    res = None
    if(part == 'a'):
        res = part_a(df_train, df_test, df_val)
    elif(part == 'b'):
        res = part_b(df_train, df_test, df_val)
    elif(part == 'c'):
        res = part_c(df_train, df_test, df_val)
    elif(part == 'd'):
        res = part_d(df_train, df_test, df_val)
    plt.plot()
    
    #part_a(df_train, df_test, df_val, oneHot=False, pruning = True)
    #f1, a1, f2, a2, f3, a3 = part_d(df_train, df_test, df_val, reCompute=True)

    

