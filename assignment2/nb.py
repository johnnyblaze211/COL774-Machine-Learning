import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import time
import sys
import random
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from collections import OrderedDict
import matplotlib.pyplot as plt
import argparse

alpha = 1
def wordcountdict(words_list):
    return dict(nltk.FreqDist(words_list))

def listFromDict(dict):
    return list(dict.keys())

def log_phi_y(df_label):
    min_label = df_label.min()
    max_label = df_label.max()
    dict1 = {}
    for label_value in range(min_label, max_label+1):
        dict1[label_value] = np.log((df_label.value_counts()[label_value])/len(df_label))
    return dict1


def sw_stemmed_list(wordList):
    ps = PorterStemmer()
    sw_set = set(stopwords.words('english'))
    res = [ps.stem(word) for word in wordList if (word not in sw_set)]
    return res

def sw_stemmed_dict(wordDict):
    dict1 = OrderedDict()
    ps = PorterStemmer()
    sw_set = set(stopwords.words('english'))
    counter = 0
    for word in wordDict.keys():
        if word not in sw_set:
            new_word = ps.stem(word)
            if dict1.get(new_word) == None:
                dict1[new_word] = counter
                counter += 1
    
    return dict1

def log_theta(vocab_dict, word_list_dict, vocab_freq_dict_i, alpha = alpha, tfidf = False, doc_count_dict = None, N = None):
    if(tfidf):
        assert(doc_count_dict != None and N!=None)
    vocab_list = listFromDict(vocab_dict)
    log_theta_dict = {}
    L = len(vocab_list)
    for k in word_list_dict.keys():
        log_theta_k = {}
        list1 = []
        sum1 = 0

        if not tfidf:
            for l in range(L):
                print(f'Calculating theta {int(l/2) + 1}/{L} for label {k}', end = '\r')
                nl = vocab_freq_dict_i[k].get(vocab_list[l])
                if not nl: nl = 0
                list1.append(nl)
                sum1+=nl
            for l in range(L):
                print(f'Calculating theta {int((L+l)/2) + 1}/{L} for label {k}', end = '\r')
                log_theta_l_k = np.log((list1[l] + alpha)/(sum1 + L*alpha))
                log_theta_k[vocab_list[l]] = (log_theta_l_k)
            log_theta_dict[k] = log_theta_k
        else:
            for l in range(L):
                print(f'Calculating theta {int(l/2) + 1}/{L} for label {k}', end = '\r')
                tf = vocab_freq_dict_i[k].get(vocab_list[l])
                df = doc_count_dict.get(vocab_list[l])
                if(tf!=None and df!=None): nl = tf*np.log(N/df)
                else: nl = 0
                list1.append(nl)
                sum1 += nl
            for l in range(L):
                print(f'Calculating theta {int((L+l)/2) + 1}/{L} for label {k}', end = '\r')
                log_theta_l_k = np.log((list1[l] + alpha)/(sum1 + L*alpha))
                log_theta_k[vocab_list[l]] = log_theta_l_k
            log_theta_dict[k] = log_theta_k
        print()
    return log_theta_dict


def model_test_accuracy(test_df, vocab_dict, log_phi_y, log_theta, stemming = False, bigrams = False, debug = False, summary = False):

    #Comment below
    #test_df = test_df[:5000]

    df_labels = test_df['overall']
    #if debug: print(test_df.info())

    if summary: df_texts = test_df['summary']
    else: df_texts = test_df['reviewText']


    #if stemming: vocab_dict = sw_stemmed_dict(vocab_dict)
    predictions = []

    correct_pred_count = 0
    labels = df_labels.to_numpy()
    texts = df_texts.to_numpy()

    L = len(vocab_dict)
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    for idx, text1 in enumerate(texts):
        if debug: print(f'Predicting test example: {idx+1}/{df_labels.shape[0]}', end = '\r')
        text = text1.lower()
        word_list = tokenizer.tokenize(text)
        if stemming: word_list = sw_stemmed_list(word_list)
        if bigrams:
            bgs_list = list(nltk.bigrams(word_list))
            word_list.extend(bgs_list)
        max_label = None
        max_value = -1e9
        for label in log_phi_y.keys():
            assert(log_phi_y[label])
            sum_log_P_xj = 0
            for word in word_list:
                if vocab_dict.get(word):
                    l = vocab_dict[word]
                    #print(f'Label, l: {label, l}')
                    sum_log_P_xj += (log_theta[label].get(word))

                        
                else:
                    sum_log_P_xj += np.log(1/L)
                    #sum_log_P_xj += np.log(1/L)
            LL_value = log_phi_y[label] + sum_log_P_xj
            if LL_value > max_value:
                max_label = label
                max_value = LL_value
        
        predictions.append(max_label)
        if max_label == labels[idx]:
            correct_pred_count+=1
    if debug: print()

    return predictions, correct_pred_count/labels.shape[0]

def random_guess_test_accuracy(train_df, test_df):
    min_label = train_df['overall'].min()
    max_label = train_df['overall'].max()
    labels = test_df['overall'].to_numpy()
    correct_pred_count = 0
    for i in range(labels.shape[0]):
        if(random.randint(min_label, max_label) == labels[i]): correct_pred_count+=1
    
    return correct_pred_count/labels.shape[0]
def majority_guess_test_accuracy(train_df, test_df):
    v = train_df['overall'].value_counts().keys().tolist()[0]
    y_test = test_df['overall'].to_numpy()
    return (v == y_test).sum()/len(test_df)


def get_confusion_matrix(y, y_pred):
    max1 = np.max(y)
    min1 = np.min(y)


    conf_mat = confusion_matrix(y, y_pred)

    fig1, ax1 = plt.subplots()
    sns.heatmap(conf_mat, annot = True, ax = ax1, fmt = 'g')
    ax1.set_xlabel('Predicted Labels')
    ax1.set_ylabel('Actual Labels')
    ax1.set_title('Confusion matrix')
    ax1.xaxis.set_ticklabels([i for i in range(min1, max1+1)])
    ax1.yaxis.set_ticklabels([i for i in range(min1, max1+1)])
    return fig1, ax1


def preprocess(texts, labels, bigrams = False, stemming = False):
    min_label = labels.min()
    max_label = labels.max()
    text_dict = {}
    text_list2 = []
    for i in range(min_label, max_label + 1, 1):
        text_dict[i] = ''

    vocab_dict_doc_count = {}
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    for j in range(len(texts)):
        l1 = tokenizer.tokenize(texts[j])
        if stemming:
            l1 = sw_stemmed_list(l1)
        if bigrams:
            bgs = list(nltk.bigrams(l1))
            l1 += bgs
        text_list2.append(l1)
        dict1 = wordcountdict(l1)
        for w in dict1:
            if vocab_dict_doc_count.get(w):
                vocab_dict_doc_count[w]+=1
            else:
                vocab_dict_doc_count[w] = 1
        text_dict[labels[j]] += (texts[j].lower() + ' ')
        print(f'Preprocessing training example {j+1}/{labels.shape[0]}', end = '\r')
    print()
    return vocab_dict_doc_count, text_dict, text_list2


def process(train_df, test_df, summary = False, stemming = False, bigrams = False, tfidf = False, train_acc = False):
    train_df = train_df
    test_df = test_df
    if summary: df_texts = train_df['summary']
    else: df_texts = train_df['reviewText']
    df_labels = train_df['overall']
    texts = df_texts.to_numpy()
    labels = df_labels.to_numpy()
    
    test_labels = test_df['overall'].to_numpy()
    min_label = labels.min()
    max_label = labels.max()

    doc_count_dict, text_dict, text_list2 = preprocess(texts, labels, bigrams = bigrams, stemming = stemming)

    word_list_dict = {}
    vocab_dict = OrderedDict()
    vocab_freq_dict_i = {}
    #vocab_freq_dict_full = {}

    dict_counter = 0
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    for label in range(min_label, max_label + 1, 1):
        word_list_dict[label] = tokenizer.tokenize(text_dict[label])
        if stemming: word_list_dict[label] = sw_stemmed_list(word_list_dict[label])
        if bigrams:
            bgs = list(nltk.bigrams(word_list_dict[label]))
            word_list_dict[label] += bgs
        vocab_freq_dict_i[label] = wordcountdict(word_list_dict[label])
        for word in word_list_dict[label]: 
            if vocab_dict.get(word) == None:

                vocab_dict[word] = dict_counter
                
                dict_counter+=1
        print(f'done label {label}')
    '''for k in range(min_label, max_label+1):
        word_list_dict[k] = []
        vocab_freq_dict_i[k] = {}
    for i in range(len(text_list2)):
        print(i)
        dict1 = wordcountdict(text_list2[i])
        for word in dict1.keys():
            if not vocab_dict.get(word):
                vocab_dict[word] = 1
            if vocab_freq_dict_i[labels[i]].get(word):
                vocab_freq_dict_i[labels[i]][word] +=dict1[word]
            else:
                vocab_freq_dict_i[labels[i]][word] = dict1[word]'''

    #s = []
    #for l in range(min_label, max_label+1):
    #    s.extend(word_list_dict[l])
    #vocab_freq_dict_full = wordcountdict(s)

    log_phi_y_dict = log_phi_y(df_labels)
    if not tfidf:log_theta_dict = log_theta(vocab_dict, word_list_dict, vocab_freq_dict_i)
    else: log_theta_dict = log_theta(vocab_dict, word_list_dict, vocab_freq_dict_i, tfidf = True, N = train_df.shape[0], doc_count_dict=doc_count_dict)

    if train_acc:
        y_pred_tr, tr_acc = model_test_accuracy(train_df, vocab_dict, log_phi_y_dict, log_theta_dict, summary = summary, stemming = stemming, bigrams = bigrams, debug = True)
        F1_score_tr = f1_score(labels, y_pred_tr, average = None)
        F1_macro_tr = f1_score(labels, y_pred_tr, average = 'macro')

    y_pred, acc = model_test_accuracy(test_df, vocab_dict, log_phi_y_dict, log_theta_dict, summary = summary, stemming = stemming, bigrams = bigrams, debug = True)
    F1_score = f1_score(test_labels, y_pred, average = None)
    F1_macro = f1_score(test_labels, y_pred, average = 'macro')
    if train_acc:
        return y_pred_tr, tr_acc, F1_score_tr, F1_macro_tr, y_pred, acc, F1_score, F1_macro
    return y_pred, acc, F1_score, F1_macro









        
            
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Naive Bayes Text Classifier')
    parser.add_argument('train_file', metavar = 'file1', help = 'Training data in json')
    parser.add_argument('test_file', metavar = 'file2', help = 'Test data in json')
    parser.add_argument('part', metavar = 'p', help = 'Part of question 1')

    args = parser.parse_args()
    train_df = pd.read_json(args.train_file, lines = True)
    try:
        test_df = pd.read_json(args.test_file, lines = True)
    except:
        print('Unable to read Test file')
        sys.exit(1)
    
    p = args.part

    #Comment below to get original size
    if(p == 'a'):
        print(f'Part a:')
        _, acc_tr, F1_tr, F1_macro_tr, _, acc, F1_score, F1_macro = process(train_df, test_df, train_acc=True)
        print(f'accuracy for training set: {acc_tr}')
        print(f'Macro F1 score for training set: {F1_macro_tr}')
        print(f'F1 scores: {F1_tr}')
        print()
        
        print(f'accuracy for test set: {acc}')
        print(f'Macro F1 score for test set: {F1_macro}')
        print(f'F1 scores: {F1_score}')
        print()

    elif(p == 'b'):
        print(f'Part b:')
        acc = random_guess_test_accuracy(train_df, test_df)
        print(f'Random guess accuracy: {acc}')
        acc = majority_guess_test_accuracy(train_df, test_df)
        print(f'Majority guess accuracy: {acc}')
        print()

    elif(p == 'c'):
        print(f'Part c:')
        y_pred, acc, F1_score, F1_macro = process(train_df[:], test_df[:])
        f, ax = get_confusion_matrix(test_df[:]['overall'].to_numpy(), y_pred)
        f.savefig('plots/Confusion_matrix_nb_part_c')
        print(f'accuracy for test set: {acc}')
        print(f'Macro F1 score for test set: {F1_macro}')
        print(f'F1 scores: {F1_score}')
        print()
        plt.show()

    elif(p == 'd'):
        print(f'Part d:')
        y_pred, acc, F1_score, F1_macro = process(train_df, test_df, stemming = True)
        print(f'Test accuracy with stemming: {acc}')
        print(f'Macro F1 score with stemming: {F1_macro}')
        print(f'F1 scores: {F1_score}')
        print()

    elif(p == 'e'):
        print(f'Part e:')
        y_pred, acc, F1_score, F1_macro = process(train_df, test_df, stemming = True, bigrams = True, tfidf = True)
        print(f'Test accuracy with unigrams + bigrams + Tfidf: {acc}')
        print(f'Macro F1 score: {F1_macro}')
        print(f'F1 scores: {F1_score}')

    elif(p == 'g'):
        print(f'Part g:')
        y_pred, acc, F1_score, F1_macro = process(train_df, test_df, summary = True, stemming = True, bigrams = True, tfidf = True)
        print(f'Test accuracy with unigrams for summarized review: {acc}')
        print(f'Macro F1 score: {F1_macro}')
        print(f'F1 scores: {F1_score}')

    else:
        print('Enter a valid part number')
        sys.exit(1)






        
        












    

    
    

    





    