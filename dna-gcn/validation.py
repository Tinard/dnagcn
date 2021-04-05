import numpy as np
import tensorflow as tf
import pandas as pd
import os

def shuffle(X, Y, M):
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    X = X[index]
    Y = Y[index]
    M = M[index]
    return X, Y, M

def markov(X1):
    X2 = np.dot(X1, X1)
    X2 = np.power(X2, 2)
    D = np.linalg.inv(np.diag(np.sqrt(np.sum(X2, axis=1))))
    X2 = np.dot(np.dot(D, X2), D)
    return X2

def mask_metric(train_len, validation_len, kmer_len):
    mask_train = [1] * train_len + [0] * validation_len + [0] * (4 ** kmer_len)
    mask_train = np.array(mask_train).reshape(len(mask_train), 1)
    mask_test = [0] * train_len + [1] * validation_len + [0] * (4 ** kmer_len)
    mask_test = np.array(mask_test).reshape(len(mask_test), 1)
    return mask_train, mask_test


def mask_sigmoid_cross_entropy(preds, labels, mask):
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
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return accuracy_all

def masked_AUC(preds, labels, mask, a, b):
    preds = preds[a:b, :] * mask[a:b, :]
    labels = labels[a:b, :] * mask[a:b, :]
    auc = tf.metrics.auc(labels, preds, num_thresholds=1000)[1]
    return auc

def validation(frequence_array, slide_frequence_array, label, kmer_length, random_seed, GPU_option):
    frequence_array, label, slide_frequence_array = shuffle(frequence_array, label, slide_frequence_array)
    train_num = int(0.8 * frequence_array.shape[0])
    validation_num = frequence_array.shape[0] - train_num
    count = (frequence_array != 0).sum(0)
    slide_count = (slide_frequence_array != 0).sum(0)
    a, b = slide_frequence_array.shape[0], slide_frequence_array.shape[1]
    w = np.zeros((b, b))
    for i in range(b):
        for j in range(i + 1, b):
            w[i, j] = np.sum((slide_frequence_array[:, i] > 0) & (slide_frequence_array[:, j] > 0))
            w[j, i] = w[i, j]
    p = slide_count / a  # p为一维数组
    a1 = frequence_array.shape[0]
    b1 = frequence_array.shape[1]
    A1 = np.identity(a1)
    A4 = np.identity(b1)
    A2 = frequence_array * (np.ones(a1).reshape(a1, 1) * np.log(a1 / (1 + count)))
    A3 = np.transpose(A2)
    p_ = p.reshape(len(p), 1) * p
    A4[w > 0] = np.log(w[w > 0] / p_[w > 0])
    A4[A4 < 0] = 0
    A4 = A4 - np.diag(np.diag(A4)) + np.identity(A4.shape[0])
    A = np.vstack((np.hstack((A1, A2)), np.hstack((A3, A4))))
    D = np.linalg.inv(np.diag(np.sqrt(np.sum(A, axis=1))))
    nor_A = np.dot(np.dot(D, A), D)

    mask_train, mask_test = mask_metric(train_num, validation_num, kmer_length)
    mask_train = mask_train.astype(np.float32)
    mask_test = mask_test.astype(np.float32)
    train_len = train_num
    test_len = validation_num
    feature_matrix = np.identity(nor_A.shape[0]).astype(np.float32)
    graph_matrix = nor_A.astype(np.float32)
    true_label = label.astype(np.float32)

    # 开始tensorflow阶段
    w1 = tf.Variable(tf.random_normal([graph_matrix.shape[0], 200], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([200, 1], stddev=1, seed=1))

    x = tf.placeholder(tf.float32, shape=(None, None), name="x-input")
    z = tf.placeholder(tf.float32, shape=(None, None), name="z-feature")
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
    a = tf.nn.relu(tf.matmul(tf.matmul(x, z), w1))
    y = tf.matmul(tf.matmul(x, a), w2)
    y1 = tf.sigmoid(y)

    cross_entropy = tf.reduce_mean(mask_sigmoid_cross_entropy(y, true_label, mask_train))
    train_accuracy = tf.reduce_mean(masked_accuracy(y, true_label, mask_train))
    test_accuracy = tf.reduce_mean(masked_accuracy(y, true_label, mask_test))
    train_auc = masked_AUC(y1, true_label, mask_train, 0, train_len)
    test_auc = masked_AUC(y1, true_label, mask_test, train_len, train_len + test_len)
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    tf.set_random_seed(random_seed)
    loss_list = []
    train_accuracy_list = []
    train_auc_list = []
    test_accuracy_list = []
    test_auc_list = []

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_option
    gpu_options = tf.GPUOptions(allow_growth = True)
    config = tf.ConfigProto(gpu_options = gpu_options)
    print("begin one dataset training")
    with tf.Session(config = config) as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        steps = 5001
        for i in range(steps):
            _, total_cross_entropy, total_train_accuracy, total_train_auc = sess.run(
                [train_step, cross_entropy, train_accuracy, train_auc], feed_dict={x: graph_matrix, z: feature_matrix, y_: true_label})
            if i % 50 == 0:
                loss_list.append(total_cross_entropy)
                train_accuracy_list.append(total_train_accuracy)
                train_auc_list.append(total_train_auc)
                total_test_accuracy, total_test_auc = sess.run([test_accuracy, test_auc],
                                                                        feed_dict={x: graph_matrix, z: feature_matrix, y_: true_label})
                test_accuracy_list.append(total_test_accuracy)
                test_auc_list.append(total_test_auc)

    print("end one dataset training")
    return np.average(np.array(test_auc_list)[-5:]), np.average(np.array(test_accuracy_list)[-5:]), np.average(np.array(loss_list)[-5:])



