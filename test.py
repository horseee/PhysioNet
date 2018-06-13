import tensorflow as tf 
import numpy as np
import argparse
import sys, os

from model import ResNet
from scipy.io import loadmat

training_set = loadmat('test.mat')
X = training_set['data'][0]
y = training_set['label'][0].astype('int32')

cut_size = 300 * 30
n = len(X)
X_cut = np.zeros(shape=(n, cut_size))
for i in range(n):
    data_len = X[i].squeeze().shape[0]
    X_cut[i, :min(cut_size, data_len)] = X[i][0, :min(cut_size, data_len)]
X = X_cut

class_num = 4

# reconstruct model
test_input = tf.placeholder(dtype='float32',shape=(None,cut_size,1))
res_net = ResNet(test_input, class_num=class_num)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

sess.run(tf.global_variables_initializer())
saver =  tf.train.Saver(tf.global_variables())

# restore model
if os.path.exists('checkpoints'):
    saver.restore(sess, 'checkpoints/model')
    print('Model successfully restore from checkpoints/model')
else: print('Restore failed. No model found!')

test_len = len(X)
label_class = {0:'N', 1:'A', 2:'O', 3:'~'}#Normal, AF, Other, Noisy
PreCount = np.zeros(class_num)
RealCount = np.zeros(class_num)
CorrectCount = np.zeros(class_num)
for i in range(test_len):
    res = sess.run([res_net], {test_input: X[i].reshape(-1, cut_size,1)})
    predicts  = np.argmax(res[0],axis=1)
    #print('case %d: class = %s, predict = %s, ' % (i, label_class[y[i]], label_class[predicts[0]]))
    PreCount[predicts] = PreCount[predicts] + 1   
    RealCount[y[i]] = RealCount[y[i]] + 1
    if (predicts[0] == y[i]):
        CorrectCount[predicts] = CorrectCount[predicts] + 1

# F1
print('F1 = %f'%np.mean(CorrectCount * 2/ (PreCount + RealCount)))
# Accuracy
print('Accuracy = %f' % (np.sum(CorrectCount) / test_len))
# Precision
precision_rate = CorrectCount / PreCount
print('Precision: N = %f, A = %f, O = %f, ~ = %f' % (precision_rate[0], precision_rate[1], precision_rate[2], precision_rate[3]))
# Recall
recall_rate = CorrectCount / RealCount
print('Recall: N = %f, A = %f, O = %f, ~ = %f' % (recall_rate[0], recall_rate[1], recall_rate[2], recall_rate[3]))