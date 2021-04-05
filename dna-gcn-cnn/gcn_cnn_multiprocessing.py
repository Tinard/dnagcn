# -*- coding:UTF-8 -*-
import os
import numpy as np
from multiprocessing import Pool
import glob
import time
from model import *
import gc
import pandas as pd


def run_data(train_set, test_set, kmer_len, slide_window, gap, embeded_size, kernel_len, kernel_num, local_window_size):
    random_seed = 123
    GPU_option = "0"
    test_loss, test_auc1, test_auc2 = main(train_set, test_set, kmer_len, slide_window, gap, embeded_size, kernel_len, kernel_num, local_window_size, random_seed, GPU_option)
    #end = run(data_path, result_path, data_info, kmer_length, slidewindow_width, random_seed, GPU_option)
    '''
    print("the test accuracy of dataset {} is {}".format(data_info, test_accuracy))
    print("the test AUC of dataset {} is {}".format(data_info, test_auc))
    print("end the training of dataset {}".format(data_info))
    '''
    #return end
    return [test_loss, test_auc1, test_auc2]


'''
def get_data_info(path):
    path_list = glob.glob(path + '*/')
    data_list = []
    for rec in path_list:
        data_info = rec.split("/")[-2]
        data_list.extend([data_info])
    return data_list
'''

'''
def get_data_info(path):
    data = pd.read_csv(path)
    low_precision_dataname = data["data_set"][data["1layer_128motif"] < 0.8]
    low_precision_dataname_list = list(low_precision_dataname)
    return low_precision_dataname_list
'''

def get_data_info(path, m, n):
    data = np.array(pd.read_csv(path))
    low_length_dataname = data[m:n, 0]
    low_length_dataname_list = list(low_length_dataname)
    return low_length_dataname_list

if __name__ == '__main__':
    #start = 16
    #end = 690

    start = 1  #127
    end = 2  #137

    #path = "/rd2/lijy/KDD/vCNNFinal/Data/ChIPSeqData/HDF5/"
    #data_prefix = "/rd2/lijy/KDD/vCNNFinal/Data/ChIPSeqData/HDF5/"
    #result_path = "/rd3/luox/dna-gcn/result_new"
    #data_list = get_data_info("/home/luox/pycharm/dna-gcn/9_model_result.csv")
    #data_list = ["wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk"]
    path = "/data/public/dna-gcn/data_size1.csv"
    #path = "/data/public/dna-gcn/low_auc_dataset.csv"
    data_prefix = "/data/public/CHIPSeqData/HDF5/"
    #result_path = "/data/public/dna-gcn/low_len_result"
    result_path = "/data/public/dna-gcn-cnn/16_4_low_len_result"
    #result_path = "/data/public/dna-gcn/new_initialization_final_16_4_low_len_result"
    data_list = get_data_info(path, start, end)
    #data_list = ["wgEncodeAwgTfbsSydhK562Elk112771IggrabUniPk", "wgEncodeAwgTfbsSydhK562Znf263UcdUniPk", "wgEncodeAwgTfbsBroadHelas3Pol2bUniPk"]

    start_time =  time.time()

    #pool = Pool(processes  = 1)
    metric_result = []
    kmer_length = 4
    random_seed = 13
    slidewindow_width = 16
    print("start run all models")

    for kmer_len in list([4]):
        for slide_window in list([16]):
            for gap in list([2]):
                for embeded_size in list([8]):
                    for kernel_len in list([32]):
                        for kernel_num in list([128]):
                            for local_window_size in list([10]):
                                for data_info in data_list[0:1]:
                                    if os.path.exists(os.path.join(result_path, data_info, 'preprocessing')) == False:
                                        os.makedirs(os.path.join(result_path, data_info, 'preprocessing'))
                                    if os.path.exists(os.path.join(result_path, data_info, 'result')) == False:
                                        os.makedirs(os.path.join(result_path, data_info, 'result'))
                                    # time.sleep(2)
                                    # pool.apply_async(run_data, (data_prefix, result_path, data_info, kmer_length, slidewindow_width, random_seed))
                                    train_dataset = data_prefix + data_info + "/train.hdf5"
                                    test_dataset = data_prefix + data_info + "/test.hdf5"
                                    #metric_result.append(pool.apply_async(run_data, (train_dataset, test_dataset, kmer_len, slide_window, gap, embeded_size, kernel_len,
                                    #kernel_num, local_window_size)))
                                    result = run_data(train_dataset, test_dataset, kmer_len, slide_window, gap, embeded_size, kernel_len,
                                    kernel_num, local_window_size)
                                    print(result)
    '''                                
    pool.close()
    pool.join()
    for res in metric_result:
        print(res.get())
    print("all model cost ", time.time() - start_time)
    '''



















