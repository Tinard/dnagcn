import argparse
from preprocessing import *
#from graph_convolution import *
#from hetero_graph_convolution import *
#from new_hetero_graph_convolution import *
from simple_graph_convolution import *
#from markov_graph_convolution import *
import gc
import pandas as pd

def run(data_path, result_path, data_info, kmer_length, slidewindow_width, random_seed, GPU_option):
    '''
    train_path = data_path + 'train.hdf5'
    test_path = data_path + 'test.hdf5'
    Z, X, train_Y, Y, train_data, test_data, train_slide_data, total_slide_data = data_preprocessing(train_path, test_path, kmer_length, slidewindow_width)
    print("end the preprocessing of dataset {}".format(data_info))
    dir_path = os.path.join(result_path, data_info, 'preprocessing')
    pd.DataFrame(Z).to_csv(dir_path + '/unnormalize_graph_matric.csv')
    pd.DataFrame(X).to_csv(dir_path + '/graph_matric.csv')
    pd.DataFrame(Y).to_csv(dir_path + '/label.csv')
    pd.DataFrame(train_Y).to_csv(dir_path + '/train_label.csv')
    pd.DataFrame(train_data).to_csv(dir_path + '/train_frequence_array.csv')
    pd.DataFrame(test_data).to_csv(dir_path + '/test_frequence_array.csv')
    pd.DataFrame(train_slide_data).to_csv(dir_path + '/train_slide_frequence_array.csv')
    pd.DataFrame(total_slide_data).to_csv(dir_path + '/total_slide_frequence_array.csv')
    '''

    '''
    for x in locals().keys():
        del locals()[x]
    gc.collect()
    end = "end the dataset {} preprocessing".format(data_info)
    return end
    '''
    #for graph convolution
    '''
    dir_path = os.path.join(result_path, data_info, 'preprocessing')
    result_dir_path = os.path.join(result_path, data_info, 'result')
    X = np.array(pd.read_csv(dir_path + '/graph_matric.csv'))[:, 1:]
    Y = np.array(pd.read_csv(dir_path + '/label.csv'))[:, 1].reshape(-1, 1)
    train_data = np.array(pd.read_csv(dir_path + '/train_frequence_array.csv'))[:, 1:]
    test_data = np.array(pd.read_csv(dir_path + '/test_frequence_array.csv'))[:, 1:]
    test_auc, validation_auc, train_loss = train(X, Y, train_data, test_data, kmer_length, result_path, data_info, random_seed, result_dir_path, GPU_option)
    print("the train loss of dataset {} is {}".format(data_info, train_loss))
    print("the validation AUC of dataset {} is {}".format(data_info, validation_auc))
    print("the test AUC of dataset {} is {}".format(data_info, test_auc))
    print("end the training of dataset {}".format(data_info))
    return train_loss, validation_auc, test_auc
    '''
    #for hetero graph
    '''

    dir_path = os.path.join(data_path, data_info, 'preprocessing')
    result_dir_path = os.path.join(result_path, data_info, 'result')
    X = np.array(pd.read_csv(dir_path + '/graph_matric.csv'))[:, 1:]
    #X = np.array(pd.read_csv(dir_path + '/unnormalize_graph_matric.csv'))[:, 1:]
    Y = np.array(pd.read_csv(dir_path + '/label.csv'))[:, 1].reshape(-1, 1)
    train_data = np.array(pd.read_csv(dir_path + '/train_frequence_array.csv'))[:, 1:]
    test_data = np.array(pd.read_csv(dir_path + '/test_frequence_array.csv'))[:, 1:]
    test_auc, validation_auc, train_loss = train(X, Y, train_data, test_data, kmer_length, result_path, data_info,
                                                 random_seed, result_dir_path, GPU_option)
    print("the train loss of dataset {} is {}".format(data_info, train_loss))
    print("the validation AUC of dataset {} is {}".format(data_info, validation_auc))
    print("the test AUC of dataset {} is {}".format(data_info, test_auc))
    print("end the training of dataset {}".format(data_info))
    return train_loss, validation_auc, test_auc
    '''
    # for simple
    dir_path = os.path.join(data_path, data_info, 'preprocessing')
    result_dir_path = os.path.join(result_path, data_info, 'result')
    X = np.array(pd.read_csv(dir_path + '/graph_matric.csv'))[:, 1:]
    # X = np.array(pd.read_csv(dir_path + '/unnormalize_graph_matric.csv'))[:, 1:]
    Y = np.array(pd.read_csv(dir_path + '/label.csv'))[:, 1].reshape(-1, 1)
    train_data = np.array(pd.read_csv(dir_path + '/train_frequence_array.csv'))[:, 1:]
    test_data = np.array(pd.read_csv(dir_path + '/test_frequence_array.csv'))[:, 1:]
    test_auc, validation_auc, train_loss = train(X, Y, train_data, test_data, kmer_length, result_path, data_info,
                                                 random_seed, result_dir_path, GPU_option)
    print("the train loss of dataset {} is {}".format(data_info, train_loss))
    print("the validation AUC of dataset {} is {}".format(data_info, validation_auc))
    print("the test AUC of dataset {} is {}".format(data_info, test_auc))
    print("end the training of dataset {}".format(data_info))
    return train_loss, validation_auc, test_auc



'''
#for simple
dir_path = os.path.join(data_path, data_info, 'preprocessing')
    result_dir_path = os.path.join(result_path, data_info, 'result')
    X = np.array(pd.read_csv(dir_path + '/graph_matric.csv'))[:, 1:]
    #X = np.array(pd.read_csv(dir_path + '/unnormalize_graph_matric.csv'))[:, 1:]
    Y = np.array(pd.read_csv(dir_path + '/label.csv'))[:, 1].reshape(-1, 1)
    train_data = np.array(pd.read_csv(dir_path + '/train_frequence_array.csv'))[:, 1:]
    test_data = np.array(pd.read_csv(dir_path + '/test_frequence_array.csv'))[:, 1:]
    test_auc, validation_auc, train_loss = train(X, Y, train_data, test_data, kmer_length, result_path, data_info,
                                                 random_seed, result_dir_path, GPU_option)
    print("the train loss of dataset {} is {}".format(data_info, train_loss))
    print("the validation AUC of dataset {} is {}".format(data_info, validation_auc))
    print("the test AUC of dataset {} is {}".format(data_info, test_auc))
    print("end the training of dataset {}".format(data_info))
    return train_loss, validation_auc, test_auc


'''
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



