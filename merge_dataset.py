import numpy as np
from scipy.io import savemat
from physionet import load_physionet

import argparse



def merge_data(dir_path, test=0.2, train_file='train',test_file='test',shuffle=True):
    train_X, train_y, test_X, test_y, _, _ = load_physionet(dir_path=dir_path, test=test, vali=0, shuffle=True)

    train_data = {'data': train_X, 'label':train_y}
    test_data = {'data': test_X, 'label':test_y}
    savemat(train_file,train_data)
    savemat(test_file, test_data)
    
    print("[!] Train set saved as %s.mat"%(train_file))
    print("[!] Test set saved as %s.mat"%(test_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',type=str,default='training2017',help='the directory of dataset')
    parser.add_argument('--test_set',type=float,default=0.2,help='The percentage of test set')
    args = parser.parse_args()

    merge_data(args.dir, test=args.test_set)

if __name__=='__main__':
    main()