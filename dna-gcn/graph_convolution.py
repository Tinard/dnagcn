import numpy as np
import tensorflow as tf
import pandas as pd
import os

def markov(X1):
    X2 = np.dot(X1, X1)
    X2 = np.power(X2, 2)
    D = np.linalg.inv(np.diag(np.sqrt(np.sum(X2, axis=1))))
    X2 = np.dot(np.dot(D, X2), D)
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
    return auc

def train(graph_matrix, true_label, train_data, test_data, kmer_length, result_path, data_info, random_seed, result_dir_path, GPU_option):
    mask_train, mask_validation, mask_test = mask_metric(train_data, test_data, kmer_length)
    mask_train = mask_train.astype(np.float32)
    mask_validation = mask_validation.astype(np.float32)
    mask_test = mask_test.astype(np.float32)
    train_len = int(0.8 * train_data.shape[0])
    validation_len = train_data.shape[0] - train_len
    test_len = np.shape(test_data)[0]
    feature_matrix = np.identity(graph_matrix.shape[0]).astype(np.float32)
    graph_matrix = graph_matrix.astype(np.float32)
    true_label = true_label.astype(np.float32)

    # 开始tensorflow阶段
    w1 = tf.Variable(tf.random_normal([graph_matrix.shape[0], 200], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([200, 1], stddev=1, seed=1))
    #w1 = tf.Variable(tf.contrib.layers.xavier_initializer()((graph_matrix.shape[0], 200)))
    #w2 = tf.Variable(tf.contrib.layers.xavier_initializer()((200, 1)))
    #w1 = tf.get_variable("w1", shape=[graph_matrix.shape[0], 200], initializer=tf.contrib.layers.xavier_initializer())
    #w2 = tf.get_variable("w2", shape=[200, 1], initializer=tf.contrib.layers.xavier_initializer())


    x = tf.placeholder(tf.float32, shape=(None, None), name="x-input")
    z = tf.placeholder(tf.float32, shape=(None, None), name="z-feature")
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
    a = tf.nn.relu(tf.matmul(tf.matmul(x, z), w1))
    y = tf.matmul(tf.matmul(x, a), w2)
    y1 = tf.sigmoid(y)

    cross_entropy = tf.reduce_mean(mask_sigmoid_cross_entropy(y, true_label, mask_train))
    train_accuracy = tf.reduce_mean(masked_accuracy(y, true_label, mask_train))
    validation_accuracy = tf.reduce_mean(masked_accuracy(y, true_label, mask_validation))
    test_accuracy = tf.reduce_mean(masked_accuracy(y, true_label, mask_test))
    train_auc = masked_AUC(y1, true_label, mask_train, 0, train_len)
    validation_auc = masked_AUC(y1, true_label, mask_validation, train_len, train_len + validation_len)
    test_auc = masked_AUC(y1, true_label, mask_test, train_len + validation_len, train_len + validation_len + test_len)
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

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
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
    config = tf.ConfigProto(gpu_options = gpu_options)
    print("begin one dataset training")
    with tf.Session(config = config) as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        steps = 10001
        for i in range(steps):
            _, total_cross_entropy, total_train_accuracy, total_validation_accuracy, total_train_auc, total_validation_auc = sess.run(
                [train_step, cross_entropy, train_accuracy, validation_accuracy, train_auc, validation_auc], feed_dict={x: graph_matrix, z: feature_matrix, y_: true_label})
            if i % 10 == 0:
                loss_list.append(total_cross_entropy)
                train_accuracy_list.append(total_train_accuracy)
                validation_accuracy_list.append(total_validation_accuracy)
                train_auc_list.append(total_train_auc)
                validation_auc_list.append(total_validation_auc)
                total_test_accuracy, total_test_auc, summary = sess.run([test_accuracy, test_auc, merged],
                                                                        feed_dict={x: graph_matrix, z: feature_matrix, y_: true_label})
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



