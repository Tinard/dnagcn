import os
from multiprocessing import Pool
import sys
import glob
import time

def run_model_test(data_info, GPU_SET):
    cmd = "python train_model.py"
    data_path = "/rd2/lijy/KDD/vCNNFinal/Data/ChIPSeqData/HDF5/" + data_info + "/"
    result_root = "/home/luox/softpooling/result/local_window_for_real_data"
    set = cmd + " " + data_path + " " +result_root + "  " + data_info + " "+GPU_SET
    print(set)
    os.system(set)

def get_data_info():
    path = "/rd2/lijy/KDD/vCNNFinal/Data/ChIPSeqData/HDF5/"
    path_list = glob.glob(path + '*/')
    data_list = []
    for rec in path_list:
        data_info = rec.split("/")[-2]
        data_list.extend([data_info])
    return data_list

if __name__ == '__main__':

    GPU_SET = sys.argv[1]
    start_time =  time.time()
    data_list = get_data_info()
    # print(len(data_list))
    # print(data_list[0:10])
    pool = Pool(processes  = 5)
    #data_list = ["wgEncodeAwgTfbsHaibGm12878Tcf3Pcr1xUniPk", "wgEncodeAwgTfbsSydhHepg2Mazab85725IggrabUniPk", "wgEncodeAwgTfbsSydhK562E2f6UcdUniPk", "wgEncodeAwgTfbsSydhGm12878Pol2IggmusUniPk"]

    for data_info in data_list[95:300]:
        time.sleep(2)
        pool.apply_async(run_model_test,(data_info,GPU_SET))
    pool.close()
    pool.join()
    print("all model cost ",  time.time() - start_time)



