import glob
import random
import pandas as pd
import numpy as np
from scipy.io import loadmat

def load_physionet(dir_path, test=0.2,vali=0, shuffle=True):
    "return train_X, train_y, test_X, test_y, valid_X, valid_y"
    if dir_path[-1]!='/': dir_path = dir_path+'/'
    ref = pd.read_csv(dir_path+'REFERENCE.csv',header=None)
    label_id = {'N':0, 'A':1, 'O':2, '~':3 }#Normal, AF, Other, Noisy
    X = []
    y = []
    test_X = None
    test_y = None
    valid_X = None
    valid_y = None
    
    for index, row in ref.iterrows():
        file_prefix = row[0]
        mat_file = dir_path+file_prefix+'.mat'
        hea_file = dir_path+file_prefix+'.hea'
        data = loadmat(mat_file)['val']

        data = data.squeeze()
        data = np.nan_to_num(data)
        data = data-np.mean(data)
        data = data/np.std(data)

        
        X.append( data )
        y.append( label_id[row[1]] )
    data_n = len(y)
    print(data_n)

    X = np.array(X)
    y = np.array(y)
        
    if shuffle:
        shuffle_idx = list(range(data_n))
        random.shuffle(shuffle_idx)
        X = X[shuffle_idx]
        y = y[shuffle_idx]
   
    valid_n = int(vali*data_n)  
    test_n = int(test*data_n)
    assert (valid_n+test_n <= data_n) , "Dataset has no enough samples!"

    if vali>0:
        valid_X = X[0:valid_n]
        valid_y = y[0:valid_n]
        
    if test>0:
        test_X = X[valid_n: valid_n+test_n]
        test_y = y[valid_n: valid_n+test_n]
    
    if vali>0 or test>0:
        X = X[valid_n+test_n: ]
        y = y[valid_n+test_n: ]
        
    #print('Train: %d, Test: %d, Validation: %d   (%s)'%((data_n-valid_n-test_n), test_n, valid_n, 'shuffled' if shuffle else 'unshuffled'))
    return np.squeeze(X), np.squeeze(y), np.squeeze(test_X), np.squeeze(test_y), np.squeeze(valid_X), np.squeeze(valid_y)