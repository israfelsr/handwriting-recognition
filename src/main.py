# common imports
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import torch
import torch
from torch.utils import data
import torchmetrics
from torch.nn import functional
from torch.utils.data import DataLoader
import wandb

# import local files
import config
from datasets.dida_dataset import load_dida_dataset, load_dummy_dida
from datasets.mnistmix_dataset import load_mnistmix_dataset, load_dummy_mnistmix
from datasets.ardis_dataset import load_ardis_dataset, load_dummy_ardis
from trainer import Trainer
from models import ResNet
from metrics import LossMeter, AccMeter
from utils import split_label


def set_seed(seed):
    """Sets a seed in all available libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def train_model(experiment_name):
    # Set up
    wandb.init(project="handwritten-recognition", name=experiment_name)
    wandb.config = config.TRAINING_ARGS
    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataloaders
    #train_gen, val_gen, test_gen = load_dida_dataset(dataset='single')
    #train_gen, val_gen, test_gen = load_dummy_dida(8, dataset='string')
    #train_gen, val_gen, test_gen = load_dummy_ardis(8)
    #train_gen, val_gen, test_gen = load_mnistmix_dataset()
    train_gen, val_gen, test_gen = load_dummy_mnistmix()

    train_loader = DataLoader(train_gen, shuffle=True, drop_last=True,
                              batch_size=config.TRAINING_ARGS['batch_size'])
    val_loader = DataLoader(val_gen, drop_last=True)
    test_loader = DataLoader(test_gen, drop_last=True)
    
    dataloaders = {
        "train" : train_loader,
        "val" : val_loader,
        "test" : test_loader
        }

    model = ResNet(num_classes=11)

    trainer = Trainer(
        training_args=config.TRAINING_ARGS,
        model = model,
        device = device,
        score_meter = AccMeter,
        loss_meter = LossMeter,
    )
    trainer.best_val_score = 0.9
    trainer.create_optimizer()

    save_path = os.path.join(config.OUTPUT_PATH, experiment_name)

    history = trainer.fit(
       config.TRAINING_ARGS['num_epochs'],
       dataloaders,
       save_path
    )

def make_predictions(experiment_name):
    set_seed(config.RANDOM_SEED)
    #train_gen, val_gen, test_gen = load_dummy_dida(8, dataset='string')
    #train_gen, val_gen, test_gen = load_dummy_ardis(8)
    train_gen, val_gen, test_gen = load_dummy_mnistmix()
    train_loader = DataLoader(train_gen, shuffle=True, drop_last=True,
                              batch_size=config.TRAINING_ARGS['batch_size'])
    val_loader = DataLoader(val_gen, drop_last=True)
    test_loader = DataLoader(test_gen, drop_last=True)
    
    dataloaders = {
        "train" : train_loader,
        "val" : val_loader,
        "test" : test_loader
        }
    
    model = ResNet(num_classes=11)
    checkpoint = torch.load(os.path.join(config.OUTPUT_PATH,
                                         experiment_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for i, (batch) in enumerate(dataloaders["val"]):
        X = batch["X"]
        with torch.no_grad():
            scores = model(X)
            plt.imshow(torch.squeeze(X).permute(1,2,0))
            plt.show()
            prediction = np.argmax(scores, axis=1)
            print('y_pred:',prediction)
            print('y:',batch["y"])

        #if i == 0:
        #    break


if __name__ == '__main__':
    train_model('dummy-mnistmix')
    make_predictions('dummy-mnistmix')