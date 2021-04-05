import numpy as np
import pandas as pd

def get_data_info(path, m, n):
    data = np.array(pd.read_csv(path))
    low_length_dataname = data[m:n, 0]
    low_length_dataname_list = list(low_length_dataname)
    return low_length_dataname_list

#path = "/data/public/dna-gcn/data_size1.csv"
path = "/data/public/dna-gcn/final_16_4_low_len_result_compare_integrate_size_50.csv"
#path = "/data/public/dna-gcn/low_auc_dataset.csv"
#result_path = "/data/public/dna-gcn/hetero_low_len_result/"
#result_path = "/data/public/dna-gcn/final_16_4_low_len_result/"
result_path = "/data/public/dna-gcn/final_simple_16_4_low_len_result/"
data_list = get_data_info(path, 0, 50)
result = []

for i in range(len(data_list)):
    data_info = data_list[i]
    validation_auc = np.max(np.array(pd.read_csv(result_path + data_info + "/result/validation_auc.csv"))[:, 1])
    index = np.argmax(np.array(pd.read_csv(result_path + data_info + "/result/validation_auc.csv"))[:, 1])
    train_auc = np.array(pd.read_csv(result_path + data_info + "/result/train_auc.csv"))[index, 1]
    test_auc = np.array(pd.read_csv(result_path + data_info + "/result/test_auc.csv"))[index, 1]
    test_loss = np.array(pd.read_csv(result_path + data_info + "/result/loss.csv"))[index, 1]
    test_accuracy = np.array(pd.read_csv(result_path + data_info + "/result/test_accuracy.csv"))[index, 1]
    max_validation_epoch = (index + 1) * 10
    max_test_epoch = (np.argmax(np.array(pd.read_csv(result_path + data_info + "/result/test_auc.csv"))[:, 1]) + 1) * 10
    max_test_auc = np.max(np.array(pd.read_csv(result_path + data_info + "/result/test_auc.csv"))[:, 1])
    result.append([data_info, train_auc, validation_auc, test_auc, test_loss, test_accuracy, max_validation_epoch, max_test_epoch, max_test_auc])

result = pd.DataFrame(np.array(result), columns = ["data_set", "train_auc", "validation_auc", "test_auc", "test_loss", "test_accuracy", "max_validation_epoch", "max_test_epoch", "max_test_auc"])
result.to_csv(result_path + "final_simple_16_4_low_len_result.csv")

