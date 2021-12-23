import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import config
from utils import split_train_test, split_label

def join_dataframes():
    df1 = pd.read_excel(os.path.join(config.ARDIS, 'Part I.xlsx'))
    df1['folder'] = 'Part I'
    df2 = pd.read_excel(os.path.join(config.ARDIS, 'Part II.xlsx'))
    df2['folder'] = 'Part II'
    df = pd.concat([df1, df2])
    df3 = df[['Image_Right', 'Date', 'folder']][~df['Image_Right'].isnull()]
    df3 = df3.rename(columns={'Image_Right':'file', 'Date': 'label'})
    df.drop(labels=['Category','City','Image_Right'], axis=1, inplace=True)
    df = df.rename(columns={'Image_Left':'file', 'Date': 'label'})
    df = pd.concat([df, df3])
    df4 = pd.read_excel(os.path.join(config.ARDIS, 'Part III.xlsx'))
    df4['folder'] = 'Part III'
    df4.drop(labels=['City', 'Category', 'Unnamed: 4'], axis=1, inplace=True)
    df4 = df4.rename(columns={'Image_name':'file', 'Date': 'label'})
    df = pd.concat([df, df4])
    df.reset_index(inplace=True, drop=True)
    df['file'] = df['file'].astype(str) + '.jpg' 
    return df

def load_ardis_dataset(train_test_ratio=0.8, train_val_ratio=0.8):
    df = join_dataframes()
    df_train, df_test = split_train_test(df, train_test_ratio)
    df_train, df_val = split_train_test(df_train, train_val_ratio)
    train_gen = ArdisDataset(df_train)
    val_gen = ArdisDataset(df_val)
    test_gen = ArdisDataset(df_test)
    return train_gen, val_gen, test_gen

def load_dummy_ardis(num_samples):
    df = join_dataframes()
    df = df.sample(num_samples, axis=0, random_state=config.RANDOM_SEED)
    train_gen = ArdisDataset(df)
    val_gen = ArdisDataset(df)
    test_gen = ArdisDataset(df)
    return train_gen, val_gen, test_gen

class ArdisDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tfrm = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        y = torch.tensor(split_label(self.df.label.iloc[idx]))
        x = Image.open(os.path.join(config.ARDIS,
                                    self.df.folder.iloc[idx],
                                    self.df.file.iloc[idx]))
        x = self.tfrm(x)
        return {'X': x, 'y': y}