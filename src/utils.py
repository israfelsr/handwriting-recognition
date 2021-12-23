import numpy as np

def split_train_test(data, test_size):
    msk = np.random.rand(len(data)) < test_size
    train = data[msk]
    test = data[~msk]
    return train, test

def split_label(number):
    label = [int(i) for i in str(number)]
    while len(label) < 4:
        label.insert(0, 10)
    return label

