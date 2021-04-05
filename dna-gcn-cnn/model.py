import tensorflow as tf
import keras
from keras import backend as K
import keras.callbacks
from keras.layers import Conv1D, Activation
from preprocessing import *
import os
from sklearn.metrics import roc_auc_score

batch_size = 128
class Databatch:
    def __init__(self, data, label, batch_size = batch_size, shuffle = False):
        self.data = data
        self.label = label
        self.index = list(range(self.data.shape[0]))

        self.cur_pos = 0
        self.batch_counter = 0

        self.batch_size = batch_size
        self.shuffle = shuffle
        if shuffle:
            self.shuffle_index()

    def next(self):
        batch_size = self.batch_size
        if self.cur_pos >= self.data.shape[0]:
            return None, None, None
        be, en = self.cur_pos, min(self.data.shape[0], self.cur_pos + batch_size)
        if en != self.data.shape[0]:
            batch_index = self.index[be:en]
            batch_data = self.data[batch_index]
            batch_label = self.label[batch_index]
        else:
            batch_index = self.index[be:en]
            batch_index.extend(self.index[0:(batch_size - en + be)])
            batch_data = self.data[batch_index]
            batch_label = self.label[batch_index]
        self.cur_pos = en
        self.batch_counter += 1
        return batch_index, batch_data, batch_label

    def shuffle_index(self):
        self.index = np.random.permutation(list(range(self.data.shape[0])))

    def reset(self):
        self.cur_pos = 0
        self.batch_counter = 0
        self.index = list(range(self.data.shape[0]))
        '''
        if self.shuffle:
            self.shuffle_index()
        '''


def build_CNN(model_template, number_of_kernel, kernel_length, input_shape, pooling = "GlobalMax", mode = 0, m = 1, m_trainable = False):
    def relu_advance(x):
        return K.relu(x, alpha=0.5, max_value=10)
    model_template.add(Conv1D(input_shape = input_shape, kernel_size = kernel_length, filters = number_of_kernel, padding = "same", strides = 1))
    model_template.add(Activation(relu_advance))
    model_template.add(keras.layers.GlobalMaxPooling1D())
    model_template.add(keras.layers.core.Dense(output_dim = 1, name = "Dense_l1"))
    model_template.add(keras.layers.Activation("sigmoid"))
    optimizer = keras.optimizers.Adam(lr = 0.01, beta_1=0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
    return model_template, optimizer

def relu_advance(x):
    return K.relu(x, alpha=0.5, max_value=10)

def gcn_cnn(kmer_len, kmer_num, embeded_size, kernel_len, kernel_num, local_window_size):
    m = np.power(kmer_len, 4)
    w1 = tf.Variable(tf.random_normal([m, 64], stddev = 1, seed = 1))
    w2 = tf.Variable(tf.random_normal([64, embeded_size], stddev = 1, seed = 1))
    weight_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=False)
    bias_init = tf.zeros_initializer()
    reg = tf.contrib.layers.l2_regularizer(1e-10)
    conv_filter = tf.get_variable(name = "filter", shape = [kernel_len, embeded_size, kernel_num], initializer = weight_init, regularizer = reg)
    conv_bias = tf.get_variable(name = "bias", shape = [kernel_num], initializer = bias_init)
    x = tf.placeholder(tf.float32, shape = (m, m), name = "kmer-graph")
    z = tf.placeholder(tf.float32, shape = (None, kmer_num, m), name = "data")
    y_ = tf.placeholder(tf.float32, shape = (None, 1), name = "label")
    a = tf.nn.relu(tf.matmul(x, w1))
    b = tf.nn.relu(tf.matmul(tf.matmul(x, a), w2))
    b = tf.tile(tf.expand_dims(b, 0), [batch_size, 1, 1])
    data = tf.matmul(z, b)
    conv1 = tf.nn.conv1d(data, conv_filter, stride = 1, padding = "SAME")
    conv1 = tf.nn.bias_add(conv1, conv_bias)
    conv1 = tf.nn.relu(conv1)
    #conv1 = keras.layers.Conv1D(input_shape=[tf.shape(data)[1], embeded_size], kernel_size=kernel_len, filters=kernel_num, padding="same", strides=1)(data)
    #conv1 = keras.layers.Activation(relu_advance)(conv1)
    pool1 = tf.layers.max_pooling1d(conv1, pool_size = local_window_size, strides = 1, padding = "valid")
    #pool1 = keras.layers.pooling.MaxPool1D(pool_size = local_window_size, stride=None, border_mode = "valid")(conv1)
    #pool1 = tf.reshape(pool1, [batch_size, -1])
    #pool1 = tf.layers.flatten(pool1)
    #output = keras.layers.core.Dense(output_dim = 1)(pool1)
    with tf.variable_scope("dense") as scope:
        shape = pool1.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim = tf.multiply(dim, d)
        output = tf.reshape(pool1, [-1, dim])
        output = tf.layers.dense(output, 1, kernel_initializer = weight_init, bias_initializer = bias_init, kernel_regularizer = reg)
    output1 = keras.layers.Activation("sigmoid")(output)
    print(output1.get_shape())
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = output, labels = y_) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y_)
    placeholders = {"graph_matrix": x, "data_matrix": z, "label": y_}
    return placeholders, output, output1, loss

def main(train_set, test_set, kmer_len, slide_window, gap, embeded_size, kernel_len, kernel_num, local_window_size, random_seed, GPU_option):
    train_data, train_label, test_data, test_label, graph_matrix, kmer_num = data_preprocessing(train_set, test_set, kmer_len, slide_window, gap)
    train_dataset = Databatch(train_data, train_label)
    test_dataset = Databatch(test_data, test_label)
    print(test_label)
    print("end the preprocessing")
    train_data = train_data.astype(np.float32)
    train_label = train_label.astype(np.float32)
    test_data = test_data.astype(np.float32)
    test_label = test_label.astype(np.float32)
    graph_matrix = graph_matrix.astype(np.float32)
    placeholders, output, output1, loss = gcn_cnn(kmer_len, kmer_num, embeded_size, kernel_len, kernel_num, local_window_size)
    opt = tf.train.AdamOptimizer(0.01).minimize(loss)
    tf.set_random_seed(random_seed)
    total_train_loss_list = []
    total_test_loss_list = []
    total_train_auc_list = []
    total_test_auc_list = []
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_DEVICE_ORDER"] = "1"
    gpu_options = tf.GPUOptions(allow_growth = True)
    config = tf.ConfigProto(gpu_options = gpu_options)
    with tf.Session(config = config) as sess:
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        for epoch in range(5000):
            total_train_loss = 0
            train_pred = np.zeros_like(train_label)
            while True:
                train_batch_index, train_batch_data, train_batch_label = train_dataset.next()
                if train_batch_data is None:
                    train_dataset.reset()
                    break
                print("the input label shape is {}".format(train_batch_label.shape))
                _, pred, train_loss = sess.run([opt, output1, loss], feed_dict={
                    placeholders["graph_matrix"]: graph_matrix,
                    placeholders["data_matrix"]: train_batch_data,
                    placeholders["label"]: train_batch_label
                })
                total_train_loss += train_loss
                train_pred[train_batch_index] = pred
            print(train_pred)
            train_auc = roc_auc_score(train_label, train_pred)
            total_train_loss_list.append(total_train_loss)
            total_train_auc_list.append(train_auc)
            print("after {} iteration the training AUC is {}".format(epoch, train_auc))
            if epoch % 10 ==0:
                total_test_loss = 0
                test_pred = np.zeros_like(test_label)
                while True:
                    test_batch_index, test_batch_data, test_batch_label = test_dataset.next()
                    if test_batch_data is None:
                        test_dataset.reset()
                        break
                    pred, test_loss = sess.run([output1, loss], feed_dict={
                        placeholders["graph_matrix"]: graph_matrix,
                        placeholders["data_matrix"]: test_batch_data,
                        placeholders["label"]: test_batch_label
                    })
                    total_test_loss += test_loss
                    test_pred[test_batch_index] = pred
                test_auc = roc_auc_score(test_label, test_pred)
                total_test_loss_list.append(total_test_loss)
                total_test_auc_list.append(test_auc)
                print("after {} iteration the testing AUC is {}".format(epoch, test_auc))
    return np.array(total_test_loss_list)[np.argmin(np.array(total_test_loss_list))], np.array(total_test_auc_list)[np.argmin(np.array(total_test_loss_list))], np.array(total_test_auc_list)[np.argmax(np.array(total_test_auc_list))]













































