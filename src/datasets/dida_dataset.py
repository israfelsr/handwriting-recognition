# common imports
import os
import numpy as np
import pandas as pd
from PIL import Image

# torch
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# local imports
import config
from utils import split_train_test, split_label

def get_single_df():
    files = []
    labels = []
    for folder in os.listdir(config.DIDA_SINGLE):
        files_in_folder = os.listdir(os.path.join(config.DIDA_SINGLE, folder))
        files += [os.path.join(folder, filename) \
                        for filename in files_in_folder]
        labels += ([int(folder)] * len(files_in_folder))
    df = pd.DataFrame({'file': files, 'label': labels})
    return df

def load_dida_dataset(dataset, train_test=0.8, train_val=0.8):
    if dataset == 'single':
        df = get_single_df()
    elif dataset == 'string':
        df = pd.read_csv(config.DIDA_STRING_LABELS, header=None,
                         names=['file', 'label'])
        df['file'] = df['file'].astype(str) + '.jpg'
        df = df.loc[df['label'].astype(str).str.len() <= 4]
        df.reset_index(drop=True, inplace=True)
    else:
        raise ValueError('Incorrect dataset attribute')
    df_train, df_test = split_train_test(df, train_test)
    df_train, df_val = split_train_test(df_train, train_val)
    train_gen = DidaDataset(df_train, dataset=dataset)
    val_gen = DidaDataset(df_val, dataset=dataset)
    test_gen = DidaDataset(df_test, dataset=dataset)
    return train_gen, val_gen, test_gen

def load_dummy_dida(num_samples=8, dataset='single'):
    if dataset == 'single':
        df = get_single_df()
    elif dataset == 'string':
        df = pd.read_csv(config.DIDA_STRING_LABELS, header=None,
                         names=['file', 'label'])
        df['file'] = df['file'].astype(str) + '.jpg'
        df = df.loc[df['label'].astype(str).str.len() <= 4]
        df.reset_index(drop=True, inplace=True)
    else:
        raise ValueError('Incorrect dataset attribute')
    df = df.sample(num_samples, axis=0, random_state=config.RANDOM_SEED)
    train_gen = DidaDataset(df, dataset=dataset)
    val_gen = DidaDataset(df, dataset=dataset)
    test_gen = DidaDataset(df, dataset=dataset)
    return train_gen, val_gen, test_gen

class DidaDataset(Dataset):
    def __init__(self, dataframe, dataset):
        self.df = dataframe
        if dataset == 'single':
            self.data_path = config.DIDA_SINGLE
        elif dataset == 'string':
            self.data_path = config.DIDA_STRING
        self.tfrm = transforms.Compose([transforms.Resize((224, 224)),
                                        #transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        y = torch.tensor(split_label(self.df.label.iloc[idx]))
        x = Image.open(os.path.join(self.data_path, self.df.file.iloc[idx]))
        x = self.tfrm(x)
        return {'X': x, 'y': y}