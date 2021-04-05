# -*- coding:UTF-8 -*-
import os
from multiprocessing import Pool
import glob
import time
from gcn import *
import gc
import pandas as pd

'''
def run_data(data_prefix, result_path, data_info, kmer_length, slidewindow_width, random_seed):
    cmd = "python /home/luox/pycharm/dna-gcn/gcn1.py"
    data_path = data_prefix + data_info + "/"
    set = cmd + " " + data_path + " " + result_path + "  " + data_info + " "+ str(kmer_length) + " " + str(slidewindow_width) + " " + str(random_seed)
    os.system(set)
'''

def run_data(data_prefix, result_path, data_info, kmer_length, slidewindow_width, random_seed):
    data_path = data_prefix + data_info + "/"
    GPU_option = "0, 1, 2, 3"
    test_loss, test_accuracy, test_auc = run(data_path, result_path, data_info, kmer_length, slidewindow_width, random_seed, GPU_option)
    #end = run(data_path, result_path, data_info, kmer_length, slidewindow_width, random_seed)
    '''
    print("the test accuracy of dataset {} is {}".format(data_info, test_accuracy))
    print("the test AUC of dataset {} is {}".format(data_info, test_auc))
    print("end the training of dataset {}".format(data_info))
    '''
    #return end
    return [data_info, test_loss, test_accuracy, test_auc]


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

    start = 49  #25
    end = 50  #50

    #path = "/data/public/dna-gcn/data_size.csv"
    path = "/data/public/new-dna-gcn/low_auc_dataset.csv"
    data_prefix = "/data/public/CHIPSeqData/HDF5/"
    #result_path = "/data/public/dna-gcn/low_len_result"
    result_path = "/data/public/new-dna-gcn/low_auc_result"
    #data_list = get_data_info(path, start, end)
    data_list = ["wgEncodeAwgTfbsSydhK562Setdb1UcdUniPk"]

    start_time =  time.time()

    pool = Pool(processes  = 1)
    metric_result = []
    kmer_length = 4
    random_seed = 13
    slidewindow_width = 16
    print("start run all models")

    for data_info in data_list[0:1]:
        if os.path.exists(os.path.join(result_path, data_info, 'preprocessing')) == False:
            os.makedirs(os.path.join(result_path, data_info, 'preprocessing'))
        if os.path.exists(os.path.join(result_path, data_info, 'result')) == False:
            os.makedirs(os.path.join(result_path, data_info, 'result'))
        time.sleep(2)
        #pool.apply_async(run_data, (data_prefix, result_path, data_info, kmer_length, slidewindow_width, random_seed))
        metric_result.append(pool.apply_async(run_data, (data_prefix, result_path, data_info, kmer_length, slidewindow_width, random_seed)))
    pool.close()
    pool.join()
    for res in metric_result:
        print(res.get())
    print("all model cost ", time.time() - start_time)









