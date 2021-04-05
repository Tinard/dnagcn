import numpy as np
import pandas as pd

def get_data_info(path, m, n):
    data = np.array(pd.read_csv(path))
    low_length_dataname = data[m:n, 0]
    low_length_dataname_list = list(low_length_dataname)
    return low_length_dataname_list

#path = "/data/public/dna-gcn/data_size.csv"
path = "/data/public/dna-gcn/low_auc_dataset.csv"
result_path = "/data/public/new-dna-gcn/low_auc_result/"
data_list = get_data_info(path, 25, 50)
result = []

for i in range(len(data_list)):
    data_info = data_list[i]
    train_auc = np.average(np.array(pd.read_csv(result_path + data_info + "/result/train_auc.csv"))[:, 1][-5:])
    test_loss = np.average(np.array(pd.read_csv(result_path + data_info + "/result/loss.csv"))[:, 1][-5:])
    test_accuracy = np.average(np.array(pd.read_csv(result_path + data_info + "/result/test_accuracy.csv"))[:, 1][-5:])
    max_epoch = (np.argmax(np.array(pd.read_csv(result_path + data_info + "/result/test_auc.csv"))[:, 1]) + 1) * 10
    max_test_auc = np.max(np.array(pd.read_csv(result_path + data_info + "/result/test_auc.csv"))[:, 1])
    test_auc = np.average(np.array(pd.read_csv(result_path + data_info + "/result/test_auc.csv"))[:, 1][-5:])
    result.append([data_info, train_auc, test_loss, test_accuracy, max_epoch, max_test_auc, test_auc])

result = pd.DataFrame(np.array(result), columns = ["data_info", "train_auc", "test_loss", "test_accuracy", "max_epoch", "max_test_auc", "test_auc"])
result.to_csv(result_path + "low_auc_result.csv")

