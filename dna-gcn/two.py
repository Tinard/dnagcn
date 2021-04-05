# -*- coding:UTF-8 -*-
from multiprocessing import Pool
from gcn import *
import gc
import pandas as pd

def run_data(data_prefix, result_path, data_info, kmer_length, slidewindow_width, random_seed):
    data_path = data_prefix + data_info + "/"
    GPU_option = "1"
    test_loss, test_accuracy, test_auc = run(data_path, result_path, data_info, kmer_length, slidewindow_width, random_seed, GPU_option)
    return [data_info, test_loss, test_accuracy, test_auc]

def get_data_info(path, n):
    data = np.array(pd.read_csv(path))
    low_length_dataname = data[0:n, 0]
    low_length_dataname_list = list(low_length_dataname)
    return low_length_dataname_list

if __name__ == '__main__':
    start = 10
    end = 20

    path = "/data/public/dna-gcn/data_size.csv"
    data_prefix = "/data/public/CHIPSeqData/HDF5/"
    result_path = "/data/public/dna-gcn/low_len_result"
    data_list = get_data_info(path, 50)

    start_time =  time.time()
    metric_result = []
    kmer_length = 4
    random_seed = 0
    slidewindow_width = 16
    print("start run all models")

    for data_info in data_list[start:end]:
        if os.path.exists(os.path.join(result_path, data_info, 'preprocessing')) == False:
            os.makedirs(os.path.join(result_path, data_info, 'preprocessing'))
        if os.path.exists(os.path.join(result_path, data_info, 'result')) == False:
            os.makedirs(os.path.join(result_path, data_info, 'result'))
        metric_result.append(run_data(data_prefix, result_path, data_info, kmer_length, slidewindow_width, random_seed))
    print(metric_result)
    metric_result = pd.DataFrame(np.array(metric_result))
    metric_result.to_csv("/data/public/dna-gcn/two.csv")
    print("all model cost ",  time.time() - start_time)








