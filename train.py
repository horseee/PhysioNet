import tensorflow as tf 
import numpy as np
import argparse
import sys, os
import random
from physionet import load_physionet
from model import ResNet
from scipy.io import loadmat


def cut_and_pad(X, cut_size):
    n = len(X)
    X_cut = np.zeros(shape=(n, cut_size))
    for i in range(n):
        data_len = X[i].squeeze().shape[0]
        # cut if too long / padd if too short
        X_cut[i, :min(cut_size, data_len)] = X[i][0, :min(cut_size, data_len)]
    return X_cut

def to_one_hot(y, class_num=4):
    if isinstance(y, int):
        y_onehot = np.zeros((1,class_num))
        y_onehot[y] = 1
        return y_onehot
    elif isinstance(y, np.ndarray):
        y_onehot = np.zeros((y.shape[0],class_num))
        y_onehot[:, y] = 1
        return y_onehot

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate',type=float,default=0.0000000001,help='learning rate')
parser.add_argument('--epochs',type=int,default=3000,help='epoch number')
parser.add_argument('--ckpt',type=str,default='checkpoints/model',help='epoch number')
parser.add_argument('--batch_size',type=int,default=16,help='batch size')
args = parser.parse_args()

class_num = 4

training_set = loadmat('train.mat')
X = training_set['data'][0]
y = training_set['label'][0].astype('int32')

cut_size = 300*30
X = cut_and_pad(X, cut_size)


# get validation set
valid_n = int(0.1*len(X))
valid_X = X[:valid_n]
valid_y = y[:valid_n]
X = X[valid_n:]
y = y[valid_n:]

y_onehot = to_one_hot(y)

#import matplotlib.pyplot as plt
#plt.plot(range(cut_size),X[0])
#plt.show()

print("[!] train: %d, validation: %d"%(len(X),len(valid_X)))


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

try:
    if os.path.exists('checkpoints'):
        saver.restore(sess, args.ckpt)
        print('Model restored from %s'%(args.ckpt))
    else: print('Restore failed, training new model!')
except: print('Restore failed, training new model!')

batch_size = args.batch_size
epochs = args.epochs
X = X.reshape(-1,cut_size,1)
valid_X = valid_X.reshape(-1,cut_size,1)
for ep in range(epochs+1):
    total_loss = []

    for itr in range(0,len(X),batch_size):
        # prepare data bactch
        if itr+batch_size>=len(X):
            cat_n = itr+batch_size-len(X)
            cat_idx = random.sample(range(len(X)),cat_n)
            batch_inputs = np.concatenate((X[itr:],X[cat_idx]),axis=0)
            batch_labels = np.concatenate((y_onehot[itr:],y_onehot[cat_idx]),axis=0)
        else:
            batch_inputs = X[itr:itr+batch_size]        
            batch_labels = y_onehot[itr:itr+batch_size]

        _, cur_loss = sess.run([opt, loss], {data_input: batch_inputs, label_input: batch_labels})
        total_loss.append(cur_loss)
        #if itr % 10==0:
        #    print('   iter %d, loss = %f'%(itr, cur_loss))
        #    saver.save(sess, args.ckpt)
    print('[*] epoch %d, average loss = %f'%(ep, np.mean(total_loss)))
    saver.save(sess, args.ckpt)

    # validation
    if ep%5==0: #and ep!=0:
        err = 0
        for i in range(valid_n):
            res = sess.run([logits], {data_input: valid_X[i].reshape(-1, cut_size,1)})
            predicts  = np.argmax(res[0],axis=1)
            if predicts!= valid_y[i]:
                err+=1
        print("[!] %d validation data, accuracy = %s"%(valid_n, (valid_n-err)/valid_n))


    
        






