# -*- coding:UTF-8 -*-
import numpy as np
import h5py
import itertools
import tensorflow as tf
import time
import argparse
import pandas as pd
import os


def transform_one_hot_X_into_kmer_frequences(one_hot_X, kmer_len,slide_window):
    base_list = ['A', 'C', 'G', 'T']
    kmers_list = [''.join(item) for item in itertools.product(base_list, repeat=kmer_len)]

    kmer_id_dict = {}
    for i in range(len(kmers_list)):
        kmer_id_dict[kmers_list[i]] = i

    #     for key in final_kmer_to_new_id_dict.keys():
    #     index = final_kmer_to_new_id_dict[key]
    #     if (key != final_kmer_list[index]) and (key != revcomp(final_kmer_list[index])):
    #         print('Something Wrong')

    frenquces_array = np.zeros((one_hot_X.shape[0], len(kmers_list)))
    slide_frequence_array = np.zeros((one_hot_X.shape[0]*(one_hot_X.shape[1]-slide_window+1), len(kmers_list)))
    '''
    print(one_hot_X.shape[0])
    print(one_hot_X.shape[1])
    '''
    #total_sequence=''
    n=0
    for i in range(one_hot_X.shape[0]):
        x = one_hot_X[i, :, :]
        s = ''.join([base_list[np.argmax(x[j])] for j in range(x.shape[0])])
        #s1=s+s[0:(slide_window-(x.shape[1]-(x.shape[1]//slide_window)*slide_window))]
        #total_sequence=total_sequence+s
        for start_index in range(len(s) - kmer_len + 1):
            this_kmer = s[start_index:start_index + kmer_len]
            frenquces_array[i, kmer_id_dict[this_kmer]] += 1
        for j in range(len(s)-slide_window+1):
            s1=s[j:(j+slide_window)]
            for k in range(slide_window-kmer_len+1):
                that_kmer=s1[k:(kmer_len+k)]
                slide_frequence_array[n*(len(s)-slide_window+1)+j, kmer_id_dict[that_kmer]] += 1
        n += 1

    '''
    slide_frequence_array=np.zeros((len(total_sequence)//slide_window,len(kmers_list)))
    for i in range(slide_frequence_array.shape[0]):
        s2=total_sequence[(i*slide_window):(slide_window*(i+1))]
        for j in range(slide_window-kmer_len+1):
            that_kmer=s2[j:(j+kmer_len)]
            slide_frequence_array[i,kmer_id_dict[that_kmer]]+=1
    '''
    return frenquces_array, slide_frequence_array

def load_data(dataset):
    data = h5py.File(dataset, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    label=np.array(label).reshape(len(label),1)
    return sequence_code,label

def data_preprocessing(train_dataset,test_dataset,kmer_len,slide_window):
    t = time.time()
    train_data,train_label = load_data(train_dataset)
    test_data,test_label = load_data(test_dataset)
    train_frequence_array,train_slide_frequence_array = transform_one_hot_X_into_kmer_frequences(train_data, kmer_len,slide_window)
    test_frequence_array,test_slide_frequence_array = transform_one_hot_X_into_kmer_frequences(test_data,kmer_len,slide_window)
    frequence_array = np.vstack((train_frequence_array,test_frequence_array))
    total_slide_frequence_array = np.vstack((train_slide_frequence_array,test_slide_frequence_array))
    '''
    print("已求解total_slide_fraquence_array")
    print(time.time()-t)
    '''
    total_raw_label = np.vstack((train_label,test_label))
    count = (frequence_array!=0).sum(0)
    slide_count = (total_slide_frequence_array!=0).sum(0)
    a, b = total_slide_frequence_array.shape[0], total_slide_frequence_array.shape[1]
    w = np.zeros((b, b))
    for i in range(b):
        for j in range(i + 1, b):
            #n = 0
            #for k in range(a):
                #if total_slide_frequence_array[k, i] > 0 and total_slide_frequence_array[k, j] > 0:
                    #n += 1
            #w[i, j] = n / a
            w[i ,j] = np.sum((total_slide_frequence_array[:,i]>0)&(total_slide_frequence_array[:,j]>0))
            w[j, i] = w[i, j]

    #print("已结束求解w[i,j]")
    p = slide_count / a  # p为一维数组
    m = frequence_array.shape[0]+frequence_array.shape[1]
    a1 = frequence_array.shape[0]
    b1 = frequence_array.shape[1]
    #A1 = cosine_similarity(frequence_array)
    A1 = np.identity(a1)
    #A2 = np.zeros((a1, b1))
    #A3 = np.zeros((b1, a1))
    A4 = np.identity(b1)
    '''
    for i in range(a1):
        for j in range(b1):
            A2[i, j] = frequence_array[i, j]* math.log(a1 / (1 + count[j]))
    '''
    A2 = frequence_array * (np.ones(a1).reshape(a1,1) * np.log(a1/(1 + count)))
    A3 = np.transpose(A2)
    p_ = p.reshape(len(p),1)*p
    A4[w>0] = np.log(w[w>0] / p_[w>0])
    A4[A4<0] = 0
    A4 = A4 - np.diag(np.diag(A4)) + np.identity(A4.shape[0])
    '''
    PMI_list=[]
    for i in range(b1):
        for j in range(i+1,b1):
            if w[i,j]>0:
                PMI=math.log(w[i,j]/(p[i]*p[j]))
                PMI_list.append(PMI)
                if PMI>0:
                    A4[i,j]=PMI
                    A4[j,i]=A4[i,j]
    '''
    '''
    print("the five percent of PMI is %g" % (np.percentile(PMI_list, 5)))
    print("the ten percent of PMI is %g" % np.percentile(PMI_list, 10))
    print("the fifteen percent of PMI is %g" % np.percentile(PMI_list, 15))
    print("the twenty percent of PMI is %g" % np.percentile(PMI_list, 20))
    print("the twenty five percent of PMI is %g" % np.percentile(PMI_list, 25))
    print("the fifty percent of PMI is %g" % np.percentile(PMI_list, 50))
    print("the seventy five percent of PMI is %g" % np.percentile(PMI_list, 75))
    
    for i in range(b):
        for j in range(b):
            if w1[i, j] + w[i, j] > 0:
                PMI = math.log((w1[i, j] + w[i, j]) / (p[i] * p[j]))
                if PMI > 0:
                    A4[i, j] = PMI
        A4[i, i] = 1
    '''
    A = np.vstack((np.hstack((A1, A2)), np.hstack((A3, A4))))
    '''
    print("已结束求解A")
    print(train_frequence_array.shape[0])
    print(test_frequence_array.shape[0])
    print(A)
    print(A.shape)
    '''
    D = np.zeros((m, m))
    #for i in range(m):
        #D[i, i] = math.sqrt(np.sum(A[i]))
    #D = np.diag(np.sqrt(np.sum(A, axis=1)))
    D = np.linalg.inv(np.diag(np.sqrt(np.sum(A, axis=1))))
    A_ = np.dot(np.dot(D, A), D)
    label = np.vstack((total_raw_label, np.array([0] * (frequence_array.shape[1])).reshape(frequence_array.shape[1], 1)))
    print(time.time() - t)
    return A_, label, train_frequence_array, test_frequence_array, total_slide_frequence_array


def mask_metric(train_data, test_data, kmer_len):
    mask_train = [1] * (np.shape(train_data)[0]) + [0] * (np.shape(test_data)[0]) + [0] * (4 ** kmer_len)
    mask_train = np.array(mask_train).reshape(len(mask_train), 1)
    mask_test = [0] * (np.shape(train_data)[0]) + [1] * (np.shape(test_data)[0]) + [0] * (4 ** kmer_len)
    mask_test = np.array(mask_test).reshape(len(mask_test), 1)
    return mask_train, mask_test


def mask_sigmoid_cross_entropy(preds, labels, mask):
    print(preds)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    # mask=tf.cast(mask,dtype=tf.float32)
    # mask/=tf.reduce_mean(mask)
    loss *= mask
    return loss


def masked_accuracy(preds, labels, mask):
    zz = tf.greater(preds, 0.5)
    preds = tf.cast(zz, dtype=tf.float32)
    correct_prediction = tf.equal(preds - labels, 0)
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    print(accuracy_all)
    # mask=tf.cast(mask,dtype=tf.floa32)
    mask /= tf.reduce_mean(mask)
    print(mask)
    accuracy_all *= mask
    print(accuracy_all)
    return accuracy_all


def masked_AUC(preds, labels, mask, a, b):
    preds = preds[a:b, :] * mask[a:b, :]
    labels = labels[a:b, :] * mask[a:b, :]
    auc = tf.metrics.auc(labels, preds, num_thresholds=1000)[1]
    return auc

'''
def Set_GPU(list_gpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = list_gpu
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('result_path')
    parser.add_argument('data_info')
    parser.add_argument('kmer_length',type = int)
    parser.add_argument('slidewindow_width', type = int)
    parser.add_argument('random_seed', type = int)
    args = parser.parse_args()
    train_path = args.data_path + 'train.hdf5'
    test_path = args.data_path + 'test.hdf5'
    X, Y, train_data, test_data, total_data = data_preprocessing(train_path, test_path, args.kmer_length, args.slidewindow_width)
    dir_path = os.path.join(args.result_path, args.data_info, 'preprocessing')
    #dir = os.mkdir(os.path.join(args.result_path, args.data_info, 'preprocessing'))
    result_dir_path = os.path.join(args.result_path, args.data_info, 'result')
    #result_dir = os.mkdir(os.path.join(args.result_path, args.data_info, 'result'))
    pd.DataFrame(X).to_csv(dir_path + '/graph_matric.csv')
    pd.DataFrame(Y).to_csv(dir_path + '/label.csv')
    pd.DataFrame(train_data).to_csv(dir_path + '/train_frequence_array.csv')
    pd.DataFrame(test_data).to_csv(dir_path + '/test_frequence_array.csv')
    pd.DataFrame(total_data).to_csv(dir_path + '/total_slide_frequence_array.csv')
    train_len = np.shape(train_data)[0]
    test_len = np.shape(test_data)[0]
    Y = Y.astype(np.float32)
    Z = np.identity(X.shape[0]).astype(np.float32)
    mask_train, mask_test = mask_metric(train_data, test_data, args.kmer_length)
    mask_train = mask_train.astype(np.float32)
    mask_test = mask_test.astype(np.float32)
    # 开始tensorflow阶段
    w1 = tf.Variable(tf.random_normal([X.shape[0], 200], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([200, 1], stddev=1, seed=1))
    # w3 = tf.Variable(tf.random_normal([40, 1], stddev=1, seed=1))
    x = tf.placeholder(tf.float32, shape=(None, None), name="x-input")
    z = tf.placeholder(tf.float32, shape=(None, None), name="z-feature")
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
    a = tf.nn.relu(tf.matmul(tf.matmul(x, z), w1))
    # a = tf.nn.dropout(a,keep_prob=0.5)
    # b = tf.nn.relu(tf.matmul(tf.matmul(x, a), w2))
    y = tf.matmul(tf.matmul(x, a), w2)
    y1 = tf.sigmoid(y)
    cross_entropy = tf.reduce_mean(mask_sigmoid_cross_entropy(y, Y, mask_train))
    # accuracy=tf.divide(tf.reduce_sum(masked_accuracy(y,Y,mask)),tf.reduce_sum(mask))
    train_accuracy = tf.reduce_mean(masked_accuracy(y, Y, mask_train))
    test_accuracy = tf.reduce_mean(masked_accuracy(y, Y, mask_test))
    train_auc = masked_AUC(y1, Y, mask_train, 0, train_len)
    test_auc = masked_AUC(y1, Y, mask_test, train_len, train_len + test_len)
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    tf.summary.scalar("train_loss", cross_entropy)
    tf.summary.scalar("train_accuracy", train_accuracy)
    tf.summary.scalar("test_accuracy", test_accuracy)
    tf.summary.scalar("train_auc",train_auc)
    tf.summary.scalar("test_auc", test_auc)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join(args.result_path, args.data_info, 'tf_log'))
    tf.set_random_seed(args.random_seed)
    loss_list = []
    train_accuracy_list = []
    train_auc_list = []
    test_accuracy_list = []
    test_auc_list = []
    '''
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        steps = 5001
        for i in range(steps):
            _, total_cross_entropy, total_train_accuracy, total_train_auc = sess.run(
                [train_step, cross_entropy, train_accuracy, train_auc], feed_dict={x: X, z: Z, y_: Y})
            if i % 50 == 0:
                loss_list.append(total_cross_entropy)
                train_accuracy_list.append(total_train_accuracy)
                train_auc_list.append(total_train_auc)
                total_test_accuracy, total_test_auc, summary = sess.run([test_accuracy, test_auc, merged],
                                                                        feed_dict={x: X, z: Z, y_: Y})
                test_accuracy_list.append(total_test_accuracy)
                test_auc_list.append(total_test_auc)
                writer.add_summary(summary, i)
        pd.DataFrame(loss_list).to_csv(result_dir_path + '/loss.csv')
        pd.DataFrame(train_accuracy_list).to_csv(result_dir_path + '/train_accuracy.csv')
        pd.DataFrame(train_auc_list).to_csv(result_dir_path + '/train_auc.csv')
        pd.DataFrame(test_accuracy_list).to_csv(result_dir_path + '/test_accuracy.csv')
        pd.DataFrame(test_auc_list).to_csv(result_dir_path + '/test_auc.csv')
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    steps = 5001
    for i in range(steps):
        _, total_cross_entropy, total_train_accuracy, total_train_auc = sess.run(
            [train_step, cross_entropy, train_accuracy, train_auc], feed_dict={x: X, z: Z, y_: Y})
        if i % 50 == 0:
            loss_list.append(total_cross_entropy)
            train_accuracy_list.append(total_train_accuracy)
            train_auc_list.append(total_train_auc)
            total_test_accuracy, total_test_auc, summary = sess.run([test_accuracy, test_auc, merged],
                                                                    feed_dict={x: X, z: Z, y_: Y})
            test_accuracy_list.append(total_test_accuracy)
            test_auc_list.append(total_test_auc)
            writer.add_summary(summary, i)
    pd.DataFrame(loss_list).to_csv(args.result_path + args.data_info + '_loss.csv')
    pd.DataFrame(train_accuracy_list).to_csv(args.result_path + args.data_info + '_train_accuracy.csv')
    pd.DataFrame(train_auc_list).to_csv(args.result_path + args.data_info + '_train_auc.csv')
    pd.DataFrame(test_accuracy_list).to_csv(args.result_path + args.data_info + '_test_accuracy.csv')
    pd.DataFrame(test_auc_list).to_csv(args.result_path + args.data_info + '_test_auc.csv')
    sess.close()
    '''

if __name__ == "__main__":
    main()

'''
X, Y, M, N,P = data_preprocessing("C:/Users/dell/Desktop/haib_train.hdf5", "C:/Users/dell/Desktop/haib_test.hdf5", 4,16)
pd.DataFrame(X).to_excel("C:/users/dell/desktop/graph matric.xlsx")
np.save("A_haib.npy", X)
np.save("label_haib.npy", Y)
np.save("train_frequence_array_haib.npy", M)
np.save("test_frequence_array_haib.npy", N)
np.save("total_slide_frequence_array_haib.npy",P)
# t=time.time()
X = np.load("A_haib.npy")
Y = np.load("label_haib.npy")
train_data = np.load("train_frequence_array_haib.npy")
test_data = np.load("test_frequence_array_haib.npy")
train_len = np.shape(train_data)[0]
test_len=np.shape(test_data)[0]
Y = Y.astype(np.float32)
m = np.shape(Y)[0]
Z = np.identity(X.shape[0]).astype(np.float32)
mask_train, mask_test = mask_metric(train_data, test_data, 4)
mask_train = mask_train.astype(np.float32)
mask_test = mask_test.astype(np.float32)
# 开始tensorflow阶段
w1 = tf.Variable(tf.random_normal([X.shape[0], 200], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([200, 1], stddev=1, seed=1))
#w3 = tf.Variable(tf.random_normal([40, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape=(None, None), name="x-input")
z = tf.placeholder(tf.float32, shape=(None, None), name="z-feature")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
a = tf.nn.relu(tf.matmul(tf.matmul(x, z), w1))
#a = tf.nn.dropout(a,keep_prob=0.5)
#b = tf.nn.relu(tf.matmul(tf.matmul(x, a), w2))
y = tf.matmul(tf.matmul(x, a), w2)
y1 = tf.sigmoid(y)

y=tf.sigmoid(tf.matmul(tf.matmul(x,a),w2))
zz=tf.greater(y,0.5)
y=tf.cast(zz,dtype=tf.float32)


cross_entropy = tf.reduce_mean(mask_sigmoid_cross_entropy(y, Y, mask_train))
# accuracy=tf.divide(tf.reduce_sum(masked_accuracy(y,Y,mask)),tf.reduce_sum(mask))
train_accuracy = tf.reduce_mean(masked_accuracy(y, Y, mask_train))
test_accuracy = tf.reduce_mean(masked_accuracy(y, Y, mask_test))
train_auc = masked_AUC(y1, Y, mask_train, 0, train_len)
test_auc = masked_AUC(y1, Y, mask_test, train_len, train_len + test_len)
train_step = tf.train.AdamOptimizer(0.02).minimize(cross_entropy)

with tf.Session() as sess:
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    #print(sess.run(w1))
    #print(sess.run(w2))
    steps = 5000
    for i in range(steps):
        sess.run(train_step, feed_dict={x: X, z: Z, y_: Y})
        if i % 50 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, z: Z, y_: Y})
            total_train_accuracy = sess.run(train_accuracy, feed_dict={x: X, z: Z, y_: Y})
            # train_auc_op = sess.run(train_auc_op, feed_dict={x: X, z: Z, y_: Y})
            total_train_auc = sess.run(train_auc, feed_dict={x: X, z: Z, y_: Y})
            print("after %d training step(s),cross entropy on all data is %g" % (i, total_cross_entropy))
            print("after %d training step(s),train accuracy on all data is %g" % (i, total_train_accuracy))
            # print(total_train_auc)
            print("after %d training step(s),train auc on all data is %g" % (i, total_train_auc))
            total_test_accuracy = sess.run(test_accuracy, feed_dict={x: X, z: Z, y_: Y})
            # test_auc_op=sess.run(test_auc_op,feed_dict={x:X,z:Z,y_:Y})
            total_test_auc = sess.run(test_auc, feed_dict={x: X, z: Z, y_: Y})
            print("after %d training step(s),test accuracy on all data is %g" % (i, total_test_accuracy))
            print("after %d training step(s),test acu on all data is %g" % (i, total_test_auc))
            # print(total_test_auc)

print(time.time() - t)
'''