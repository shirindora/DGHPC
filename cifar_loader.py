import numpy as np
import pickle


CIFAR_PATH = '/home/shirin/model/CIFAR-10/'


def load_cifar(n_batch, norm_image=True, one_hot_label=False, return_labels=False, return_test_data=False):
    # ipdb.set_trace()
    n_class = 10
    tr_data = np.zeros((0, 32, 32, 3)) # initialize empty data array
    # initialize empty label array
    if one_hot_label is False:
        tr_label = np.zeros((0))
    else:
        tr_label = np.zeros((0, n_class))

    for b in range(n_batch):
        fo = open(CIFAR_PATH + 'data_batch_' + str(b+1), 'rb')
        batch_data = pickle.load(fo, encoding='latin1')
        fo.close()

        img_data = batch_data['data'].reshape(10000, 3, 32, 32).transpose([0, 2, 3, 1])
        img_label = batch_data['labels']

        tr_data = np.concatenate((tr_data, img_data), axis=0)
        if one_hot_label is False:
            tr_label = np.concatenate((tr_label, np.array(img_label)), axis=0)
        else:
            tr_label = np.concatenate((tr_label, gen_one_hot_label(img_label, n_class)), axis=0)

    # load testing data
    fo = open(CIFAR_PATH + 'test_batch', 'rb')
    batch_data = pickle.load(fo, encoding='latin1')
    fo.close()

    ts_data = batch_data['data'].reshape(10000, 3, 32, 32).transpose([0, 2, 3, 1])
    if one_hot_label is False:
        ts_label = np.array(batch_data['labels'])
    else:
        ts_label = gen_one_hot_label(batch_data['labels'], n_class)

    if norm_image is True:
        tr_data = tr_data / 255
        ts_data = ts_data / 255

    if return_test_data is True:
        if return_labels is True:
            return tr_data, tr_label, ts_data, ts_label
        else:
            return tr_data, ts_data
    else:
        if return_labels is True:
            return tr_data, tr_label
        else:
            return tr_data


def gen_one_hot_label(batch_label, n_class):
    one_hot_label = np.zeros((len(batch_label), n_class))
    one_hot_label[range(len(batch_label)), batch_label] = 1

    return one_hot_label