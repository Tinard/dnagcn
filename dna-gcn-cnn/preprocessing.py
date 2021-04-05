import numpy as np
import h5py
import itertools

def transform_one_hot_X_into_kmer_frequences(one_hot_X, kmer_len, slide_window, gap):
    base_list = ['A', 'C', 'G', 'T']
    kmers_list = [''.join(item) for item in itertools.product(base_list, repeat=kmer_len)]

    kmer_id_dict = {}
    for i in range(len(kmers_list)):
        kmer_id_dict[kmers_list[i]] = i

    frenquces_one_hot = np.zeros((one_hot_X.shape[0], (one_hot_X.shape[1] - kmer_len) // gap + 1, len(kmers_list)))
    slide_frequence_array = np.zeros((one_hot_X.shape[0] * ((one_hot_X.shape[1] - slide_window) // gap + 1), len(kmers_list)))
    kmer_num = (one_hot_X.shape[1] - kmer_len) // gap + 1

    N = 0
    for i in range(one_hot_X.shape[0]):
        x = one_hot_X[i, :, :]
        l = x.shape[0]
        s = "".join([base_list[np.argmax(x[j])] for j in range(l)])
        m = 0
        n = 0
        while m < ((l - kmer_len) // gap + 1):
            this_kmer = s[(m * gap): (m * gap + kmer_len)]
            frenquces_one_hot[i, m, kmer_id_dict[this_kmer]] += 1
            m = m + 1
        while n < ((l - slide_window) // gap + 1):
            s1 = s[(n * gap):(n * gap + slide_window)]
            p = 0
            while p < ((slide_window - kmer_len) // gap + 1):
                that_kmer = s1[(gap * p):(gap * p + kmer_len)]
                slide_frequence_array[N * ((l - slide_window) // gap + 1) + n, kmer_id_dict[that_kmer]] += 1
                p = p + 1
            n = n + 1
        N = N + 1
        '''
        while m < ((l - kmer_len) // gap + 2):
            if m != ((l - kmer_len) // gap + 1):
                this_kmer = s[(m * gap) : (m * gap + kmer_len)]
                frenquces_one_hot[i, m, kmer_id_dict[this_kmer]] += 1
            else:
                if l - m*gap == kmer_len:
                    this_kmer = s[(m * gap):]
                    frenquces_one_hot[i, m, kmer_id_dict[this_kmer]] += 1
                else:
                    this_kmer = s[(m * gap):] + s[0:(kmer_len - l + m*gap)]
                    frenquces_one_hot[i, m, kmer_id_dict[this_kmer]] += 1
            this_kmer = s[(m * gap): (m * gap + kmer_len)]
            frenquces_one_hot[i, m, kmer_id_dict[this_kmer]] += 1
            m = m + 1
        while n < ((l - slide_window) // gap + 2):
            if n != ((l - slide_window) // gap + 1):
                s1 = s[(n * gap):(n * gap + slide_window)]
                p = 0
                while p < ((slide_window - kmer_len) // gap + 2):
                    if p != ((slide_window - kmer_len) // gap + 1):
                        that_kmer = s1[(gap * p):(gap * p + kmer_len)]
                        slide_frequence_array[N * ((l - slide_window) // gap + 1) + n, kmer_id_dict[that_kmer]] += 1
                    else:
                        if slide_window - gap * p == kmer_len:
                            that_kmer = s1[(gap * p):]
                            slide_frequence_array[N * ((l - slide_window) // gap + 1) + n, kmer_id_dict[that_kmer]] += 1
                        else:
                            that_kmer = s1[(gap * p):] + s[(n * gap + slide_window):(n * gap + slide_window + slide_window - gap * p)]
                            slide_frequence_array[N * ((l - slide_window) // gap + 1) + n, kmer_id_dict[that_kmer]] += 1
                    p = p + 1
            else:
                if l - n * gap == slide_window:
                    s1 = s[(n * gap):]
                    p = 0
                    while p < ((slide_window - kmer_len) // gap + 2):
                        if p != ((slide_window - kmer_len) // gap + 1):
                            that_kmer = s1[(gap * p):(gap * p + kmer_len)]
                            slide_frequence_array[N * ((l - slide_window) // gap + 1) + n, kmer_id_dict[that_kmer]] += 1
                        else:
                            if slide_window - gap * p == kmer_len:
                                that_kmer = s1[(gap * p):]
                                slide_frequence_array[N * ((l - slide_window) // gap + 1) + n, kmer_id_dict[that_kmer]] += 1
                            else:
                                that_kmer = s1[(gap * p):] + s[0:(kmer_len - slide_window + gap * p)]
                                slide_frequence_array[N * ((l - slide_window) // gap + 1) + n, kmer_id_dict[that_kmer]] += 1
                        p = p + 1
                else:
                    s1 = s[(n * gap):] + s[0:(slide_window - l + n * gap)]
                    p = 0
                    while p < ((slide_window - kmer_len) // gap + 2):
                        if p != ((slide_window - kmer_len) // gap + 1):
                            that_kmer = s1[(gap * p):(gap * p + kmer_len)]
                            slide_frequence_array[N * ((l - slide_window) // gap + 1) + n, kmer_id_dict[that_kmer]] += 1
                        else:
                            if slide_window - gap * p == kmer_len:
                                that_kmer = s1[(gap * p):]
                                slide_frequence_array[N * ((l - slide_window) // gap + 1) + n, kmer_id_dict[that_kmer]] += 1
                            else:
                                that_kmer = s1[(gap * p):] + s[0:(kmer_len - slide_window + gap * p)]
                                slide_frequence_array[N * ((l - slide_window) // gap + 1) + n, kmer_id_dict[that_kmer]] += 1
                        p = p + 1
            n = n + 1
        N = N + 1
        '''
    return frenquces_one_hot, slide_frequence_array, kmer_num


def load_data(dataset):
    data = h5py.File(dataset, 'r')
    #sequence_code = data['sequences'].value
    sequence_code = data['sequences'][()]
    #label = data['labs'].value
    label = data['labs'][()]
    label = np.array(label).reshape(len(label), 1)
    return sequence_code, label


def data_preprocessing(train_dataset, test_dataset, kmer_len, slide_window, gap):
    train_data, train_label = load_data(train_dataset)
    test_data, test_label = load_data(test_dataset)
    train_frequence_one_hot, train_slide_frequence_array, kmer_num = transform_one_hot_X_into_kmer_frequences(train_data, kmer_len,
                                                                                                  slide_window, gap)
    test_frequence_one_hot, test_slide_frequence_array, kmer_num = transform_one_hot_X_into_kmer_frequences(test_data, kmer_len,
                                                                                                slide_window, gap)
    total_slide_frequence_array = np.vstack((train_slide_frequence_array, test_slide_frequence_array))
    slide_count = (total_slide_frequence_array != 0).sum(0)

    a, b = total_slide_frequence_array.shape[0], total_slide_frequence_array.shape[1]
    w = np.zeros((b, b))
    for i in range(b):
        for j in range(i + 1, b):
            w[i, j] = np.sum((total_slide_frequence_array[:, i] > 0) & (total_slide_frequence_array[:, j] > 0))
            w[j, i] = w[i, j]
    '''
    for i in range(b):
        for j in range(i + 1, b):
            for k in range(frequence_array.shape[0]):
                ref_matrix = total_slide_frequence_array[(k * (101 - slide_window + 1)):((k+1) * (101 - slide_window + 1))]
                if np.sum((ref_matrix[:, i] > 0) & (ref_matrix[:, j] > 0)) > 0:
                    w[i, j] = w[i, j] + 1
            w[j, i] = w[i, j]
    w = w / (frequence_array.shape[0])
    '''
    p = slide_count / a
    b1 = train_frequence_one_hot.shape[2]
    p_ = p.reshape(len(p), 1) * p
    A = np.zeros((b1, b1))
    A[w > 0] = np.log(w[w > 0] / p_[w > 0])
    A[A < 0] = 0
    A = A - np.diag(np.diag(A)) + np.identity(A.shape[0])
    D = np.linalg.inv(np.diag(np.sqrt(np.sum(A, axis=1))))
    A_ = np.dot(np.dot(D, A), D)
    return train_frequence_one_hot, train_label, test_frequence_one_hot, test_label, A_, kmer_num



