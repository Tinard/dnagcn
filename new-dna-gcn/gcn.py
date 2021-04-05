import argparse
from preprocessing import *
from graph_convolution import *
#from hetero_graph_convolution import *
import pandas as pd

def run(data_path, result_path, data_info, kmer_length, slidewindow_width, random_seed, GPU_option):
    '''
    train_path = data_path + 'train.hdf5'
    test_path = data_path + 'test.hdf5'
    Z, X, Y, train_data, test_data, weight_train_data, weight_test_data = data_preprocessing(train_path, test_path, kmer_length, slidewindow_width)
    print("end the preprocessing of dataset {}".format(data_info))
    dir_path = os.path.join(result_path, data_info, 'preprocessing')

    pd.DataFrame(X).to_csv(dir_path + '/graph_matric.csv')
    pd.DataFrame(X).to_csv(dir_path + '/unnormalize_graph_matric.csv')
    pd.DataFrame(Y).to_csv(dir_path + '/label.csv')
    pd.DataFrame(train_data).to_csv(dir_path + '/train_frequence_array.csv')
    pd.DataFrame(test_data).to_csv(dir_path + '/test_frequence_array.csv')
    '''

    dir_path = os.path.join(result_path, data_info, 'preprocessing')
    result_dir_path = os.path.join(result_path, data_info, 'result')
    X = np.array(pd.read_csv(dir_path + '/graph_matric.csv'))[:, 1:]
    #X = np.array(pd.read_csv(dir_path + '/unnormalize_graph_matric.csv'))[:, 1:]
    Y = np.array(pd.read_csv(dir_path + '/label.csv'))[:, 1].reshape(-1, 1)
    train_data = np.array(pd.read_csv(dir_path + '/train_frequence_array.csv'))[:, 1:]
    test_data = np.array(pd.read_csv(dir_path + '/test_frequence_array.csv'))[:, 1:]
    test_auc, test_accuracy, test_loss = train(X, Y, train_data, test_data, kmer_length, result_path, data_info, random_seed, result_dir_path, GPU_option)
    print("the test loss of dataset {} is {}".format(data_info, test_loss))
    print("the test accuracy of dataset {} is {}".format(data_info, test_accuracy))
    print("the test AUC of dataset {} is {}".format(data_info, test_auc))
    print("end the training of dataset {}".format(data_info))
    return test_loss, test_accuracy, test_auc


'''
parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('result_path')
parser.add_argument('data_info')
parser.add_argument('kmer_length', type=int)
parser.add_argument('slidewindow_width', type=int)
parser.add_argument('random_seed', type=int)
args = parser.parse_args()
train_path = args.data_path + 'train.hdf5'
test_path = args.data_path + 'test.hdf5'

X, Y, train_data, test_data, total_data = data_preprocessing(train_path, test_path, args.kmer_length,
                                                             args.slidewindow_width)
print("end the preprocessing of dataset {}".format(args.data_info))
dir_path = os.path.join(args.result_path, args.data_info, 'preprocessing')
# dir = os.mkdir(os.path.join(args.result_path, args.data_info, 'preprocessing'))
result_dir_path = os.path.join(args.result_path, args.data_info, 'result')
# result_dir = os.mkdir(os.path.join(args.result_path, args.data_info, 'result'))
pd.DataFrame(X).to_csv(dir_path + '/graph_matric.csv')
pd.DataFrame(Y).to_csv(dir_path + '/label.csv')
pd.DataFrame(train_data).to_csv(dir_path + '/train_frequence_array.csv')
pd.DataFrame(test_data).to_csv(dir_path + '/test_frequence_array.csv')
pd.DataFrame(total_data).to_csv(dir_path + '/total_slide_frequence_array.csv')

test_auc, test_accuracy = train(X, Y, train_data, test_data, args.kmer_length, args.result_path, args.data_info, args.random_seed, result_dir_path)
print("the test accuracy of dataset {} is {}".format(args.data_info, test_accuracy))
print("the test AUC of dataset {} is {}".format(args.data_info, test_auc))
print("end the training of dataset {}".format(args.data_info))
'''



