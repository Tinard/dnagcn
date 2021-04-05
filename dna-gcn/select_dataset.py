import numpy as np
import pandas as pd
import glob
import h5py


data1 = pd.read_csv("/data/public/dna-gcn/9_model_result.csv")
low_precision_dataname = list(data1["data_set"][data1["1layer_128motif"] < 0.8])
#data2 = data1[data1["1layer_128motif"] < 0.8]
data3 = pd.read_csv("/data/public/dna-gcn/data_size.csv")
data4 = pd.merge(data1, data3, on = "data_set", how = "left")
data5 = data4[data4["1layer_128motif"] < 0.8]
data5.to_csv("/data/public/dna-gcn/low_auc_dataset.csv")
#print(low_precision_dataname_list)
#print(len(low_precision_dataname_list))




'''
def get_data_info(path):
    path_list = glob.glob(path + '*/')
    data_list = []
    for rec in path_list:
        data_info = rec.split("/")[-2]
        data_list.extend([data_info])
    return data_list

def load_data(dataset):
    data = h5py.File(dataset, 'r')
    label = data['labs'].value
    sequence_code = data['sequences'].value
    return sequence_code.shape[1]
    #return len(label)

path = "/data/public/CHIPSeqData/HDF5/"
data_list = get_data_info(path)
data_size_matrix = np.zeros((len(data_list), 2))
for i in range(len(data_list)):
    train_path = path + data_list[i] + '/train.hdf5'

    test_path = path + data_list[i] + '/test.hdf5'
    print(load_data(test_path))
    train_len = load_data(train_path)
    test_len = load_data(test_path)
    data_size_matrix[i, 0] = train_len
    data_size_matrix[i, 1] = test_len
    size_frame = pd.DataFrame(data_size_matrix, index = data_list, columns = ["train_data_length", "test_data_length"])
    size_frame.to_csv("/data/public/dna-gcn/data_size.csv")
'''

