import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image

# torch
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# local imports
import config
from utils import split_label

def load_files():
    files = ['EMNIST_train_test.npz', 'ARDIS_train_test.npz']
    X_train = None
    for file in files:
        data = np.load(os.path.join(config.MNIST_MIX, file))
        if X_train is None:
            X_train, X_test = data['X_train'], data['X_test']
            y_train, y_test = data['y_train'], data['y_test']
        else:
            X_train = np.concatenate((X_train, data['X_train']), axis=0)
            X_test = np.concatenate((X_test, data['X_test']), axis=0)
            y_train = np.concatenate((y_train, data['y_train']), axis=0)
            y_test = np.concatenate((y_test, data['y_test']), axis=0)
    return X_train, X_test, y_train, y_test

def load_mnistmix_dataset():
    X_train, X_test, y_train, y_test = load_files()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                        stratify=y_train,
                                                        test_size=0.2)
    train_gen = MnistMixDataset(X_train, y_train)
    val_gen = MnistMixDataset(X_val, y_val)
    test_gen = MnistMixDataset(X_test, y_test)
    return train_gen, val_gen, test_gen

def load_dummy_mnistmix(num_samples=8):
    X_train, X_test, y_train, y_test = load_files()
    samples = np.random.randint(0, X_train.shape[0], num_samples)
    train_gen = MnistMixDataset(X_train[samples], y_train[samples])
    val_gen = MnistMixDataset(X_train[samples], y_train[samples])
    test_gen = MnistMixDataset(X_train[samples], y_train[samples])
    return train_gen, val_gen, test_gen

class MnistMixDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data
        self.tfrm = transforms.Compose([transforms.Resize((224,224)),
                                        #transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        y = torch.tensor(split_label(self.y[idx]))
        x = self.x[idx,:,:]
        x = np.repeat(np.expand_dims(x, axis=2), 3, axis=2)
        x = Image.fromarray(x)
        x = self.tfrm(x)
        return {'X': x, 'y': y}