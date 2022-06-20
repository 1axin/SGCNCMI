# 作者:     wxf

# 开发时间: 2022/1/8 16:26
from __future__ import division
from __future__ import print_function

# from sys import flags
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from utilize import *
import time
import os
import csv
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
from optimizer import OptimizerGCAE
from model import SGCNCMI
from preprocessing import preprocess_graph, preprocess_normalized, construct_feed_dict, sparse_to_tuple, mask_test_edges, mask_test_edges_all_neg, mask_test_edges_local_5FCV
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
flags = tf.app.flags
FLAGS = flags.FLAGS
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        for i in range(len(row)):
            row[i] = float(row[i])
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def ReadMyCsv_1(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

# m1, tep_N1 = get_new_matrix_MD(M_M, th_r1)
def get_new_matrix_MD(D_D, th_d):
    N_d= D_D.shape[0]  #383
    d = np.zeros([N_d, N_d])  #383*383的0矩阵
    tep_d = 0  #定义一个临时为0
    for i in range(N_d):
        for j in range(i):
            if D_D[i][j]>th_d or D_D[j][i]>th_d:
               d[i][j] = 1
               d[j][i]=1
               tep_d = tep_d + 1
            else:
                d[i][j]=0   #d里面除了0就是1
    return d, tep_d



def MyEnlarge(x0, y0, width, height, x1, y1, times, mean_fpr, mean_tpr, thickness=1, color = 'blue'):
    def MyFrame(x0, y0, width, height):
        import matplotlib.pyplot as plt
        import numpy as np

        x1 = np.linspace(x0, x0, num=20)  # 生成列的横坐标，横坐标都是x0，纵坐标变化
        y1 = np.linspace(y0, y0, num=20)
        xk = np.linspace(x0, x0 + width, num=20)
        yk = np.linspace(y0, y0 + height, num=20)

        xkn = []
        ykn = []
        counter = 0
        while counter < 20:
            xkn.append(x1[counter] + width)
            ykn.append(y1[counter] + height)
            counter = counter + 1

        plt.plot(x1, yk, color='k', linestyle=':', lw=1, alpha=1)  # 左
        plt.plot(xk, y1, color='k', linestyle=':', lw=1, alpha=1)  # 下
        plt.plot(xkn, yk, color='k', linestyle=':', lw=1, alpha=1)  # 右
        plt.plot(xk, ykn, color='k', linestyle=':', lw=1, alpha=1)  # 上

        return
    # 画虚线框
    width2 = times * width
    height2 = times * height
    MyFrame(x0, y0, width, height)
    MyFrame(x1, y1, width2, height2)

    # 连接两个虚线框
    xp = np.linspace(x0 + width, x1, num=20)
    yp = np.linspace(y0, y1 + height2, num=20)
    plt.plot(xp, yp, color='k', linestyle=':', lw=1, alpha=1)

    # 小虚框内各点坐标
    XDottedLine = []
    YDottedLine = []
    counter = 0
    while counter < len(mean_fpr):
        if mean_fpr[counter] > x0 and mean_fpr[counter] < (x0 + width) and mean_tpr[counter] > y0 and mean_tpr[counter] < (y0 + height):
            XDottedLine.append(mean_fpr[counter])
            YDottedLine.append(mean_tpr[counter])
        counter = counter + 1

    # 画虚线框内的点
    # 把小虚框内的任一点减去小虚框左下角点生成相对坐标，再乘以倍数（4）加大虚框左下角点
    counter = 0
    while counter < len(XDottedLine):
        XDottedLine[counter] = (XDottedLine[counter] - x0) * times + x1
        YDottedLine[counter] = (YDottedLine[counter] - y0) * times + y1
        counter = counter + 1


    plt.plot(XDottedLine, YDottedLine, color=color, lw=thickness, alpha=1)
    return
#占位符 主要是在用命令行执行程序时，需要传些参数
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 64, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 128, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 128, 'Number of units in hidden layer 3.')
flags.DEFINE_float('weight_decay', 0.0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'miRNA-disease', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
model_str = FLAGS.model
dataset_str = FLAGS.dataset

M_M = S_miR
m_m = preprocess_normalized(M_M)

N_m = M_M.shape[0]
print('N_m',N_m)

D_D = S_circR
d_d = preprocess_normalized(D_D)

N_d = D_D.shape[0]
print('N_d',N_d)

M_D = Y
H =get_feature(M_M,M_D,D_D)
# X_m = H[0:N_m, :]
# X_d = H[N_m:N_m+N_d, :]
X_m = H[0:N_m, :]
X_d = H[N_m:, :]

d = np.zeros([N_d, N_d])
m = np.zeros([N_m, N_m])
adj111 = np.hstack((m,M_D))
adj222 = np.hstack((M_D.transpose(), d))
adj_MD = np.vstack((adj111,adj222))

label_matrix = adj_MD

adj_MD_1 = adj_MD
adj_MD = sp.coo_matrix(adj_MD)

#******************************************************************

edges_all, edges_pos, edges_false = mask_test_edges(adj_MD)
X_sample =np.vstack((edges_pos, edges_false))
Y_sample = np.hstack((np.ones(len(edges_pos)), np.zeros(len(edges_false))))

labes_all = []
preds_all = []
ite = 0
labes_all_MD = []
preds_all_MD = []

adj_orig =sp.coo_matrix(adj_MD)

adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()
adj_or = adj_MD

kf = KFold(n_splits=5, shuffle=True)
pre_matrix_M = np.zeros([N_d+N_m, N_d+N_m])
pre_matrix_all = np.zeros([N_d+N_m, N_d+N_m])
#########################################################################
ii = 1
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
cnt = 0

Y_test = []
Y_pre = []
Y_all = []
# colorlist = ['red', 'gold', 'purple', 'green', 'blue', 'black']
#########################################################################
for train_index, test_index in kf.split(X_sample):
    adj = adj_MD
    features = H
    ite = ite +1
    cnt += 1
    if ite<6:
        train_x_edge = X_sample[train_index, :]
        train_y_label = Y_sample[train_index]
        train_pos = []
        train_neg = []
        train_N = len(train_x_edge)
        for i in range(train_N):
            if train_y_label [i] == 1:
                train_pos.append(train_x_edge[i,:])  #
            else:
                train_neg.append(train_x_edge[i,:])
        test_pos = []
        test_neg = []
        test_x_edge = X_sample[test_index, :]
        test_y_label = Y_sample[test_index]
        test_N = len(test_x_edge)
        for i in range(test_N):
            if test_y_label[i] == 1:
                test_pos.append(test_x_edge[i,:])
            else:
                test_neg.append(test_x_edge[i,:])

        train_pos = np.array(train_pos)
        data = np.ones(train_pos.shape[0])

        adj_all =sp.coo_matrix(adj)
        adj_train = adj_orig - sp.dia_matrix((adj_all.diagonal()[np.newaxis, :], [0]), shape=adj_all.shape) #adj_orig 变成对角上全为0的矩阵
        adj = adj_train
        features = sp.coo_matrix(features)
        adj_norm = preprocess_graph(adj)
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }
        num_nodes = adj.shape[0]
        features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        model = None
        model = SGCNCMI(placeholders, num_features, num_nodes, features_nonzero)
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float(
            (adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        opt = OptimizerGCN(preds = model.reconstructions,
                            labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices = False), [-1]),
                            model = model, num_nodes = num_nodes,
                            pos_weight = pos_weight,
                            norm = norm)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Train model
        for epoch in range(FLAGS.epochs) :
            t = time.time()
            feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout'] : FLAGS.dropout})
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict = feed_dict)
            avg_cost = outs[1]
            avg_accuracy = outs[2]
        print("Optimization Finished!")
        labes_y, preds_y, pre_matrix_M, pre_matrix_all = get_roc_score(test_pos, test_neg, pre_matrix_M, pre_matrix_all)
        roc_score1 = roc_auc_score(labes_y, preds_y)
        print(roc_score1)
        ap_score1 = average_precision_score(labes_y, preds_y)
        print(ap_score1)
        print("the training of this time", ite, roc_score1, ap_score1)
        print(ite)
        labes_all.extend(labes_y)
        preds_all.extend(preds_y)
        ###########################################################################

        i += 1
        ###########################################################################
    roc_score = roc_auc_score(labes_all, preds_all)
    ap_score = average_precision_score(labes_all, preds_all)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


    fpr = []
    tpr = []
    fpr, tpr, _ = roc_curve(labes_y, preds_y)

    Y_test1 = []
    Y_pre1 = []

    np.save('Y_test' + str(i), Y_test1)
    np.save('Y_pre' + str(i), Y_pre1)

    roc_auc = auc(fpr, tpr)

    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0

    plt.plot(fpr, tpr, lw = 1.5, alpha = 0.5,
             label = 'ROC fold %d (AUC = %0.4f)'  % (ii , roc_auc))

    # MyEnlarge(0, 0.7, 0.25, 0.25, 0.5, 0, 2, mean_fpr, mean_tpr, 2, colorlist[ii-1])

    ii += 1

# plt.figure()
mean_tpr /= cnt  # 求数组的平均值
mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, 'k--', label = 'Mean ROC (AUC = {0:.4f})'.format(mean_auc), lw = 2)
# MyEnlarge(0, 0.7, 0.25, 0.25, 0.5, 0, 2, mean_fpr, mean_tpr, 2, colorlist[5] )

plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r'
         , alpha=0.5)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize = 13)
plt.ylabel('True Positive Rate', fontsize = 13)
plt.title('Receiver operating characteristic')
plt.legend(bbox_to_anchor = (0.45, 0.45))

# # 保存图片
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
# plt.savefig('ROC-5fold.svg')
plt.savefig('ROC-5fold.tif')
plt.show()