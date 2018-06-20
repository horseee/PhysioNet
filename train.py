import tensorflow as tf 
import numpy as np
import argparse
import sys, os
import random
from physionet import load_physionet
from model import ResNet
from scipy.io import loadmat


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def cut_and_pad(X, cut_size):
    n = len(X)
    X_cut = np.zeros(shape=(n, cut_size))
    for i in range(n):
        data_len = X[i].squeeze().shape[0]
        # cut if too long / padd if too short
        X_cut[i, :min(cut_size, data_len )] = X[i][0,  : min(cut_size, data_len)]
    return X_cut

def to_one_hot(y, class_num=4):
    if isinstance(y, int):
        y_onehot = np.zeros((1,class_num))
        y_onehot[y] = 1
        return y_onehot
    elif isinstance(y, np.ndarray):
        y_onehot = np.zeros((y.shape[0],class_num))
        for i in range(y.shape[0]):
            y_onehot[i, y[i]] = 1
        return y_onehot

def get_sub_set(X, y, k, K_folder_or_not):
    if not K_folder_or_not:
        k_dataset_len = int(len(X) * 0.9)
        train_X = X[ : k_dataset_len ]
        train_y = y[ : k_dataset_len ]
        valid_X = X[ k_dataset_len :]
        valid_y = y[ k_dataset_len :]
    else:
        k_dataset_len = int(len(X) / 5)
        if k == 0:
            valid_X = X[ : k_dataset_len ]
            valid_y = y[ : k_dataset_len ]
            train_X = X[ k_dataset_len :]
            train_y = y[ k_dataset_len :]
        else:
            print(k*k_dataset_len)
            valid_X = X[ k*k_dataset_len : (k+1)*k_dataset_len ]
            valid_y = y[ k*k_dataset_len : (k+1)*k_dataset_len ]
            train_X = np.concatenate((X[ : k*k_dataset_len] , X[(k+1)*k_dataset_len: ]), axis=0)
            train_y = np.concatenate((y[ : k*k_dataset_len] , y[(k+1)*k_dataset_len: ]), axis=0)
    return train_X, train_y, valid_X, valid_y

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate',type=float,default=0.0000002,help='learning rate')
parser.add_argument('--epochs',type=int,default=30000,help='epoch number')
parser.add_argument('--batch_size',type=int,default=16, help='batch size')
parser.add_argument('--k_folder', type=bool, default=False, help='If open kfolder validation')
args = parser.parse_args()

class_num = 4

training_set = loadmat('train.mat')
X = training_set['data'][0]
y = training_set['label'][0].astype('int32')

#cut_size_start = 300 * 3
cut_size = 300 * 30

X = cut_and_pad(X, cut_size)

#import matplotlib.pyplot as plt
#plt.plot(range(cut_size),X[0])
#plt.show()


# k-fold / train
if args.k_folder:
    low_border = 0
    high_border = 5
    F1_valid = np.zeros(5)
else:
    low_border = 0
    high_border = 1

for k in range(low_border,high_border):
    # get validation set
    train_X, train_y, valid_X, valid_y = get_sub_set(X, y, k, args.k_folder)
    y_onehot = to_one_hot(train_y)

    if args.k_folder:
        print("[!] kfolder_iter: %d, train: %d, validation: %d"%(k, len(train_X),len(valid_X)))
    else:
        print("[!] Training: %d, validation: %d" % (len(train_X),len(valid_X)))

    data_input = tf.placeholder(dtype='float32',shape=(None,cut_size,1))
    label_input = tf.placeholder(dtype='float',shape=(None))

    # build model
    logits = ResNet(data_input, class_num=class_num)
    loss = tf.losses.softmax_cross_entropy(label_input, logits)
    opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    try: os.mkdir('checkpoints')
    except: pass 
    sess.run(tf.global_variables_initializer())
    saver =  tf.train.Saver(tf.global_variables())

    if not args.k_folder:
        try:
            if os.path.exists('checkpoints'):
                saver.restore(sess, 'checkpoints/model')
                print('Model restored from checkpoints')
            else: print('Restore failed, training new model!')
        except: print('Restore failed, training new model!')


    batch_size = args.batch_size
    epochs = args.epochs
    train_X = train_X.reshape(-1,cut_size,1)
    valid_X = valid_X.reshape(-1,cut_size,1)
    ep = 0
    while True:
        total_loss = []
        ep = ep + 1
        for itr in range(0,len(train_X),batch_size):
            # prepare data bactch
            if itr+batch_size>=len(train_X):
                cat_n = itr+batch_size-len(train_X)
                cat_idx = random.sample(range(len(train_X)),cat_n)
                batch_inputs = np.concatenate((train_X[itr:],train_X[cat_idx]),axis=0)
                batch_labels = np.concatenate((y_onehot[itr:],y_onehot[cat_idx]),axis=0)
            else:
                batch_inputs = train_X[itr:itr+batch_size]        
                batch_labels = y_onehot[itr:itr+batch_size]

            _, cur_loss = sess.run([opt, loss], {data_input: batch_inputs, label_input: batch_labels})
            total_loss.append(cur_loss)
            #if itr % 10==0:
            #    print('   iter %d, loss = %f'%(itr, cur_loss))
            #    saver.save(sess, args.ckpt)
        print('[*] epoch %d, average loss = %f'%(ep, np.mean(total_loss)))
        if not args.k_folder:
            saver.save(sess, 'checkpoints/model')

        # validation
        if ep % 5 ==0: #and ep!=0:
            err = 0
            n = np.zeros(class_num)
            N = np.zeros(class_num)
            correct = np.zeros(class_num)
            valid_n = len(valid_X)
            for i in range(valid_n):
                res = sess.run([logits], {data_input: valid_X[i].reshape(-1, cut_size,1)})
                # print(valid_y[i])
                # print(res)
                predicts  = np.argmax(res[0],axis=1)
                n[predicts] = n[predicts] + 1   
                N[valid_y[i]] = N[valid_y[i]] + 1
                if predicts[0]!= valid_y[i]:
                    err+=1
                else:
                    correct[predicts] = correct[predicts] + 1
            print("[!] %d validation data, accuracy = %f"%(valid_n, 1.0 * (valid_n - err)/valid_n))
            res = 2.0 * correct / (N + n)
            print("[!] Normal = %f, Af = %f, Other = %f, Noisy = %f" % (res[0], res[1], res[2], res[3]))
            print("[!] F1 accuracy = %f" % np.mean(2.0 * correct / (N + n)))
            if args.k_folder:
                F1_valid[k] = np.mean(res)
        
        if np.mean(total_loss) < 0.2 and ep % 5 == 0:
            break

if args.k_folder:
    print("\n\n[!] k-folder finished!! The F1 score for each folder is :")
    print("[!] 1: %f, 2: %f, 3: %f, 4: %f, 5: %f" % (F1_valid[0], F1_valid[1], F1_valid[2], F1_valid[3], F1_valid[4]))
    print("[!] Average is %f" % (np.mean(F1_valid)))
