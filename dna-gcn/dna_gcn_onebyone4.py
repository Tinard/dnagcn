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
    #data_path = data_prefix + data_info + "/"
    data_path = data_prefix + "/"
    GPU_option = "0"
    train_loss, validation_auc, test_auc = run(data_path, result_path, data_info, kmer_length, slidewindow_width, random_seed, GPU_option)
    #end = run(data_path, result_path, data_info, kmer_length, slidewindow_width, random_seed, GPU_option)
    '''
    print("the test accuracy of dataset {} is {}".format(data_info, test_accuracy))
    print("the test AUC of dataset {} is {}".format(data_info, test_auc))
    print("end the training of dataset {}".format(data_info))
    '''
    #return end
    return [data_info, train_loss, validation_auc, test_auc]


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

    start = 41  #127
    end = 47  #137

    #path = "/rd2/lijy/KDD/vCNNFinal/Data/ChIPSeqData/HDF5/"
    #data_prefix = "/rd2/lijy/KDD/vCNNFinal/Data/ChIPSeqData/HDF5/"
    #result_path = "/rd3/luox/dna-gcn/result_new"
    #data_list = get_data_info("/home/luox/pycharm/dna-gcn/9_model_result.csv")
    #data_list = ["wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk"]
    #path = "/data/public/dna-gcn/data_size.csv"
    path = "/data/public/dna-gcn/final_16_4_low_len_result_compare_integrate_size_50.csv"
    #path = "/data/public/dna-gcn/low_auc_dataset.csv"
    #data_prefix = "/data/public/CHIPSeqData/HDF5/"
    data_prefix = "/data/public/dna-gcn/final_16_4_low_len_result"
    #result_path = "/data/public/dna-gcn/low_len_result"
    #result_path = "/data/public/dna-gcn/final_16_4_low_len_result"
    result_path = "/data/public/dna-gcn/final_simple_16_4_low_len_result"
    #result_path = "/data/public/dna-gcn/new_initialization_final_16_4_low_len_result"
    data_list = get_data_info(path, start, end)
    #data_list = ["wgEncodeAwgTfbsSydhHelas3Znf274UcdUniPk"]
    '''
    data_list = ["wgEncodeAwgTfbsBroadH1hescChd1a301218aUniPk", "wgEncodeAwgTfbsBroadH1hescRbbp5a300109aUniPk",
                 "wgEncodeAwgTfbsBroadHelas3Ezh239875UniPk", "wgEncodeAwgTfbsBroadK562P300UniPk", "wgEncodeAwgTfbsBroadK562Pol2bUniPk",
                 "wgEncodeAwgTfbsBroadNhdfadEzh239875UniPk", "wgEncodeAwgTfbsBroadNhekPol2bUniPk", "wgEncodeAwgTfbsHaibA549Ets1V0422111Etoh02UniPk",
                 "wgEncodeAwgTfbsHaibA549GrPcr1xDex500pmUniPk", "wgEncodeAwgTfbsHaibGm12878Pmlsc71910V0422111UniPk", "wgEncodeAwgTfbsHaibGm12878Pol2Pcr2xUniPk",
                 "wgEncodeAwgTfbsHaibGm12878Pol24h8Pcr1xUniPk", "wgEncodeAwgTfbsHaibGm12891Pol2Pcr1xUniPk", "wgEncodeAwgTfbsHaibGm12891Pol24h8Pcr1xUniPk",
                 "wgEncodeAwgTfbsHaibGm12892Pol2V0416102UniPk", "wgEncodeAwgTfbsHaibH1hescFosl1sc183V0416102UniPk", "wgEncodeAwgTfbsHaibH1hescSp4v20V0422111UniPk",
                 "wgEncodeAwgTfbsHaibPanc1Pol24h8V0416101UniPk", "wgEncodeAwgTfbsHaibPanc1Sin3ak20V0416101UniPk", "wgEncodeAwgTfbsSydhA549Pol2s2IggrabUniPk",
                 "wgEncodeAwgTfbsSydhGm12878Tr4UniPk", "wgEncodeAwgTfbsSydhGm12891Pol2IggmusUniPk", "wgEncodeAwgTfbsSydhH1hescChd1a301218aIggrabUniPk",
                 "wgEncodeAwgTfbsSydhH1hescCtbp2UcdUniPk", "wgEncodeAwgTfbsSydhHek293Elk4UcdUniPk", "wgEncodeAwgTfbsSydhHelas3Brg1IggmusUniPk",
                 "wgEncodeAwgTfbsSydhHelas3E2f6UniPk", "wgEncodeAwgTfbsSydhHelas3Ini1IggmusUniPk", "wgEncodeAwgTfbsSydhHepg2Pol2IggrabUniPk",
                 "wgEncodeAwgTfbsSydhHepg2Pol2s2IggrabUniPk", "wgEncodeAwgTfbsSydhK562Brf1UniPk", "wgEncodeAwgTfbsSydhK562Brf2UniPk",
                 "wgEncodeAwgTfbsSydhK562Ini1IggmusUniPk", "wgEncodeAwgTfbsSydhK562Pol2Ifna30UniPk", "wgEncodeAwgTfbsSydhK562Pol2Ifng6hUniPk",
                 "wgEncodeAwgTfbsSydhK562Pol2Ifng30UniPk", "wgEncodeAwgTfbsSydhK562Pol2s2UniPk", "wgEncodeAwgTfbsSydhK562Tblr1ab24550IggrabUniPk",
                 "wgEncodeAwgTfbsSydhK562Tr4UcdUniPk", "wgEncodeAwgTfbsSydhK562Znf274m01UcdUniPk", "wgEncodeAwgTfbsSydhPbdePol2UcdUniPk",
                 "wgEncodeAwgTfbsSydhU2osKap1UcdUniPk", "wgEncodeAwgTfbsUchicagoK562Ehdac8UniPk", "wgEncodeAwgTfbsUtaGm12878CmycUniPk", "wgEncodeAwgTfbsUtaH1hescCmycUniPk"]
    '''

    start_time =  time.time()

    pool = Pool(processes  = 6)
    metric_result = []
    kmer_length = 4
    random_seed = 13
    slidewindow_width = 16
    print("start run all models")

    '''
    for kmer_number in kmer_length_list:
        for random_seed in random_seed_list:
            for local_window_size in slidewindow_width_list:
                for data_info in data_list[start:end]:
                    if os.path.exists(os.path.join(result_path, data_info, 'preprocessing')) == False:
                        os.makedirs(os.path.join(result_path, data_info, 'preprocessing'))
                    if os.path.exists(os.path.join(result_path, data_info, 'result')) == False:
                        os.makedirs(os.path.join(result_path, data_info, 'result'))
                    time.sleep(2)
                    #pool.apply_async(run_data, (data_prefix, result_path, data_info, kmer_number, local_window_size, random_seed))
                    metric_result.append(pool.apply_async(run_data,(data_prefix, result_path, data_info, kmer_number, local_window_size, random_seed)))
                    #result = pool.apply_async(run_data,(data_prefix, result_path, data_info, kmer_number, local_window_size, random_seed)).get()
                    #print(result)
                    #print(data_prefix)
                    #print(data_info)
                    #keyi = run_data(data_prefix, result_path, data_info, kmer_number, local_window_size, random_seed)
                    #print(keyi)
                    #print(end)
    '''

    for data_info in data_list[0:6]:
        #if os.path.exists(os.path.join(result_path, data_info, 'preprocessing')) == False:
            #os.makedirs(os.path.join(result_path, data_info, 'preprocessing'))
        if os.path.exists(os.path.join(result_path, data_info, 'result')) == False:
            os.makedirs(os.path.join(result_path, data_info, 'result'))
        #time.sleep(2)
        #pool.apply_async(run_data, (data_prefix, result_path, data_info, kmer_length, slidewindow_width, random_seed))
        metric_result.append(pool.apply_async(run_data, (data_prefix, result_path, data_info, kmer_length, slidewindow_width, random_seed)))
    pool.close()
    pool.join()
    for res in metric_result:
        print(res.get())
    print("all model cost ", time.time() - start_time)



















