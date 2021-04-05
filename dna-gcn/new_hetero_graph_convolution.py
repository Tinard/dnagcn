import numpy as np
import tensorflow as tf
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

def markov(X1):
    X2 = np.dot(X1, X1)
    X2 = np.power(X2, 2)
    D = np.linalg.inv(np.diag(np.sqrt(np.sum(X2, axis=1))))
    X2 = np.dot(np.dot(D, X2), D)
    '''
    X2 = np.exp(X1)
    D = np.linalg.inv(np.diag(np.sqrt(np.sum(X2, axis=1))))
    X2 = np.dot(np.dot(D, X2), D)
    '''
    return X2

def mask_metric(train_data, test_data, kmer_len):
    num = int(0.8 * train_data.shape[0])
    mask_train = [1] * num + [0] * (train_data.shape[0] - num) + [0] * (np.shape(test_data)[0]) + [0] * (4 ** kmer_len)
    mask_validation = [0] * num + [1] * (train_data.shape[0] - num) + [0] * (np.shape(test_data)[0]) + [0] * (4 ** kmer_len)
    mask_train = np.array(mask_train).reshape(len(mask_train), 1)
    mask_validation = np.array(mask_validation).reshape(len(mask_validation), 1)
    mask_test = [0] * (np.shape(train_data)[0]) + [1] * (np.shape(test_data)[0]) + [0] * (4 ** kmer_len)
    mask_test = np.array(mask_test).reshape(len(mask_test), 1)
    return mask_train, mask_validation, mask_test



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
    #auc = tf.py_func(roc_auc_score, (labels, preds), tf.float32)
    #auc = roc_auc_score(labels, preds)
    return auc

def train(graph_matrix, true_label, train_data, test_data, kmer_length, result_path, data_info, random_seed, result_dir_path, GPU_option):

    mask_train, mask_validation, mask_test = mask_metric(train_data, test_data, kmer_length)
    mask_train = mask_train.astype(np.float32)
    mask_validation = mask_validation.astype(np.float32)
    mask_test = mask_test.astype(np.float32)
    train_len = np.shape(train_data)[0]
    train_len1 = int(0.8 * train_data.shape[0])
    validation_len = train_data.shape[0] - train_len1
    test_len = np.shape(test_data)[0]
    total_len = train_len + test_len
    graph_matrix = graph_matrix.astype(np.float32)
    markov_graph_matrix = markov(graph_matrix)
    graph_size = graph_matrix.shape[0]
    true_label = true_label.astype(np.float32)

    A11 = graph_matrix[0:total_len, 0:total_len]
    A12 = graph_matrix[0:total_len, total_len:graph_size]
    A21 = graph_matrix[total_len:graph_size, 0:total_len]
    A22 = graph_matrix[total_len:graph_size, total_len:graph_size]

    B11 = markov_graph_matrix[0:total_len, 0:total_len]
    B12 = markov_graph_matrix[0:total_len, total_len:graph_size]
    B21 = markov_graph_matrix[total_len:graph_size, 0:total_len]
    B22 = markov_graph_matrix[total_len:graph_size, total_len:graph_size]

    feature11 = np.identity(total_len).astype(np.float32)
    feature12 = np.zeros((total_len, graph_size - total_len)).astype(np.float32)
    feature21 = feature12.T
    feature22 = np.identity(graph_size - total_len).astype(np.float32)

    # 开始tensorflow阶段
    #W1 = tf.Variable(tf.random_normal([total_len, 200], stddev = 1, seed = 1))
    #W2 = tf.Variable(tf.random_normal([graph_size - total_len, 200], stddev = 1, seed = 1))
    #W3 = tf.Variable(tf.random_normal([200, 1], stddev = 1, seed = 1))
    #W4 = tf.Variable(tf.random_normal([200, 1], stddev = 1, seed = 1))
    W1 = tf.get_variable("W1", shape=[total_len, 200], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", shape=[graph_size - total_len, 200], initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", shape=[200, 1], initializer=tf.contrib.layers.xavier_initializer())
    W4 = tf.get_variable("W4", shape=[200, 1], initializer=tf.contrib.layers.xavier_initializer())

    x11 = tf.placeholder(tf.float32, shape=(total_len, total_len), name="x11-input")
    x12 = tf.placeholder(tf.float32, shape=(total_len, graph_size - total_len), name="x12-input")
    x21 = tf.placeholder(tf.float32, shape=(graph_size - total_len, total_len), name="x21-input")
    x22 = tf.placeholder(tf.float32, shape=(graph_size - total_len, graph_size - total_len), name="x22-input")

    v11 = tf.placeholder(tf.float32, shape=(total_len, total_len), name="v11-input")
    v12 = tf.placeholder(tf.float32, shape=(total_len, graph_size - total_len), name="v12-input")
    v21 = tf.placeholder(tf.float32, shape=(graph_size - total_len, total_len), name="v21-input")
    v22 = tf.placeholder(tf.float32, shape=(graph_size - total_len, graph_size - total_len), name="v22-input")

    z11 = tf.placeholder(tf.float32, shape=(total_len, total_len), name="z11-feature")
    z12 = tf.placeholder(tf.float32, shape=(total_len, graph_size - total_len), name="z12-feature")
    z21 = tf.placeholder(tf.float32, shape=(graph_size - total_len, total_len), name="z21-feature")
    z22 = tf.placeholder(tf.float32, shape=(graph_size - total_len, graph_size - total_len), name="z22-feature")

    y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

    h1 = tf.nn.relu(tf.matmul(tf.matmul(x11, z11) + tf.matmul(x12, z21), W1) + tf.matmul(tf.matmul(x11, z12) + tf.matmul(x12, z22) , W2))
    h2 = tf.nn.relu(tf.matmul(tf.matmul(x21, z11) + tf.matmul(x22, z21), W1) + tf.matmul(tf.matmul(x21, z12) + tf.matmul(x22, z22) , W2))

    h3 = tf.matmul(tf.matmul(x11, h1), W3) + tf.matmul(tf.matmul(x12, h2), W4)
    h4 = tf.matmul(tf.matmul(x21, h1), W3) + tf.matmul(tf.matmul(x22, h2), W4)
    y = tf.concat([h3, h4], 0)
    y1 = tf.sigmoid(y)

    g1 = tf.nn.relu(tf.matmul(tf.matmul(v11, z11) + tf.matmul(v12, z21), W1) + tf.matmul(tf.matmul(v11, z12) + tf.matmul(v12, z22), W2))
    g2 = tf.nn.relu(tf.matmul(tf.matmul(v21, z11) + tf.matmul(v22, z21), W1) + tf.matmul(tf.matmul(v21, z12) + tf.matmul(v22, z22), W2))
    g3 = tf.matmul(tf.matmul(v11, g1), W3) + tf.matmul(tf.matmul(v12, g2), W4)
    g4 = tf.matmul(tf.matmul(v21, g1), W3) + tf.matmul(tf.matmul(v22, g2), W4)
    t = tf.concat([g3, g4], 0)
    t1 = tf.sigmoid(t)

    cross_entropy = tf.reduce_mean(mask_sigmoid_cross_entropy(y, true_label, mask_train))
    regularizer = tf.reduce_mean(tf.square(y1 - t1))
    total_loss = cross_entropy + regularizer
    train_accuracy = tf.reduce_mean(masked_accuracy(y, true_label, mask_train))
    validation_accuracy = tf.reduce_mean(masked_accuracy(y, true_label, mask_validation))
    test_accuracy = tf.reduce_mean(masked_accuracy(y, true_label, mask_test))
    train_auc = masked_AUC(y1, true_label, mask_train, 0, train_len1)
    validation_auc = masked_AUC(y1, true_label, mask_validation, train_len1, train_len1 + validation_len)
    test_auc = masked_AUC(y1, true_label, mask_test, train_len1 + validation_len, train_len1 + validation_len + test_len)
    #train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(0.01).minimize(total_loss)

    tf.summary.scalar("train_loss", cross_entropy)
    tf.summary.scalar("train_accuracy", train_accuracy)
    tf.summary.scalar("validation_accuracy", validation_accuracy)
    tf.summary.scalar("test_accuracy", test_accuracy)
    tf.summary.scalar("train_auc", train_auc)
    tf.summary.scalar("validation_auc", validation_auc)
    tf.summary.scalar("test_auc", test_auc)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join(result_path, data_info, 'tf_log'))
    tf.set_random_seed(random_seed)
    loss_list = []
    train_accuracy_list = []
    train_auc_list = []
    validation_accuracy_list = []
    validation_auc_list = []
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
        steps = 10001
        for i in range(steps):
            _, total_cross_entropy, total_train_accuracy, total_validation_accuracy, total_train_auc, total_validation_auc = sess.run([train_step, cross_entropy, train_accuracy, validation_accuracy, train_auc, validation_auc], feed_dict={x11: A11, x12: A12, x21: A21, x22: A22, v11: B11, v12: B12, v21: B21, v22: B22, z11: feature11,
                                                                                                                                                        z12: feature12, z21: feature21, z22: feature22, y_: true_label})

            if i % 10 == 0:
                loss_list.append(total_cross_entropy)
                train_accuracy_list.append(total_train_accuracy)
                validation_accuracy_list.append(total_validation_accuracy)
                train_auc_list.append(total_train_auc)
                validation_auc_list.append(total_validation_auc)
                total_test_accuracy, total_test_auc, summary = sess.run([test_accuracy, test_auc, merged],
                                                                        feed_dict={x11: A11, x12: A12, x21: A21,
                                                                                   x22: A22, v11: B11, v12: B12, v21: B21, v22: B22, z11: feature11,
                                                                                   z12: feature12, z21: feature21,
                                                                                   z22: feature22, y_: true_label})
                test_accuracy_list.append(total_test_accuracy)
                test_auc_list.append(total_test_auc)
                writer.add_summary(summary, i)
        pd.DataFrame(loss_list).to_csv(result_dir_path + '/loss.csv')
        pd.DataFrame(train_accuracy_list).to_csv(result_dir_path + '/train_accuracy.csv')
        pd.DataFrame(train_auc_list).to_csv(result_dir_path + '/train_auc.csv')
        pd.DataFrame(validation_accuracy_list).to_csv(result_dir_path + '/validation_accuracy.csv')
        pd.DataFrame(validation_auc_list).to_csv(result_dir_path + '/validation_auc.csv')
        pd.DataFrame(test_accuracy_list).to_csv(result_dir_path + '/test_accuracy.csv')
        pd.DataFrame(test_auc_list).to_csv(result_dir_path + '/test_auc.csv')
    print("end one dataset training")
    return np.array(test_auc_list)[np.argmax(np.array(validation_auc_list))], np.max(np.array(validation_auc_list)), np.array(loss_list)[np.argmax(np.array(validation_auc_list))]



