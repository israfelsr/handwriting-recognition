import unittest
import torch

import config
from metrics import AccMeter
from models import ResNet
from torch.utils.data import DataLoader

from datasets.dida_dataset import load_dida_dataset
from datasets.mnistmix_dataset import load_mnistmix_dataset
from datasets.ardis_dataset import load_ardis_dataset


class TestSegmentation(unittest.TestCase):
    def test_batch_size(self):
        train_gen, val_gen, test_gen = load_ardis_dataset()
        train_loader = DataLoader(train_gen, batch_size=config.TRAINING_ARGS['batch_size'],
                                  shuffle=True, num_workers=1, drop_last=True)
        val_loader = DataLoader(val_gen, batch_size=config.TRAINING_ARGS['batch_size'],
                                num_workers=1, drop_last=True)
        test_loader = DataLoader(test_gen, batch_size=config.TRAINING_ARGS['batch_size'],
                                 num_workers=1, drop_last=True)
        dataloaders = {
            "train" : train_loader,
            "val" : val_loader,
            "test" : test_loader
            }
        for t, batch in enumerate(dataloaders['train']):
            X = batch['X']
            y = batch['y']
            self.assertTrue(X.shape, torch.Size([config.TRAINING_ARGS['batch_size'],
                                                 3, config.IMG_SIZE, 
                                                 config.IMG_SIZE]))
            self.assertTrue(y.shape, torch.Size([config.TRAINING_ARGS['batch_size']]))
            if t == 0:
                break
        for t, batch in enumerate(dataloaders['val']):
            X = batch['X']
            y = batch['y']
            self.assertTrue(X.shape, torch.Size([config.TRAINING_ARGS['batch_size'],
                                                 3, config.IMG_SIZE,
                                                 config.IMG_SIZE]))
            self.assertTrue(y.shape, torch.Size([config.TRAINING_ARGS['batch_size']]))
            if t == 0:
                break
        for t, batch in enumerate(dataloaders['test']):
            X = batch['X']
            y = batch['y']
            self.assertTrue(X.shape, torch.Size([config.TRAINING_ARGS['batch_size'],
                                                 3, config.IMG_SIZE,
                                                 config.IMG_SIZE]))
            self.assertTrue(y.shape, torch.Size([config.TRAINING_ARGS['batch_size']]))
            if t == 0:
                break

    def test_model(self):
        """
        Test forward pass of the model
        """
        X = torch.zeros((config.TRAINING_ARGS['batch_size'], 3,
                         config.IMG_SIZE, config.IMG_SIZE))
        model = ResNet(num_classes=10)
        scores = model(X)
        self.assertTrue(scores.shape, torch.Size([8, 10]))

def test_acc():
    y_pred = torch.tensor([[ 0.4858,  0.6958,  0.0306,  0.7770,  0.6259,  0.1276,
         -0.7542,  0.4666, 0.2959,  1.0777],
        [-0.1865,  0.2297,  0.1106,  0.3003,  0.2583,  0.4774, -0.1107,  1.4097,
         -0.4110,  0.6992],
        [-0.8680,  0.2484,  0.0055,  1.3186,  0.9454, -0.5465, -1.3505,  0.6664,
          0.4772,  0.4428],
        [-0.1565,  1.0046, -0.6214,  1.0686, -0.4949,  0.2359, -0.4943,  1.0331,
         -0.8928,  0.8868],
        [-0.2606,  0.1590, -0.3239,  0.8456, -0.0625,  0.1906, -0.3217,  1.7145,
          0.0380,  1.2183],
        [ 0.5721, -0.1222,  0.1233,  0.7182,  0.5146, -0.5065, -0.4782,  1.6363,
         -0.0776,  0.4507],
        [ 0.5142,  1.3534, -0.9732,  0.7146,  0.7011,  0.2375, -0.6774,  1.2856,
         -0.1855,  0.5992],
        [ 0.1284,  0.2926, -0.1391,  0.8577,  0.1958,  0.0829, -0.9126,  0.9421,
          0.1614,  0.0084]])
    y_true = torch.tensor([0, 1, 6, 8, 4, 1, 9, 0])
    print(y_true)
    metric = AccMeter()
    metric.update(y_pred, y_true)

def test_metrics():
    y_true = [[9, 5], [9, 8], [8, 7], [7, 5], [9, 9], [7, 8], [5, 8], [7, 5]]
    y_pred = [[[ 0.2827,  0.2667, -0.2153, -0.0416, -0.3625, -0.2428,  1.0710, 0.0155,  0.5113, 1.1490], 
              [-0.2540,  0.7986, -0.2418,  0.4245,  0.6900, 1.1756, 0.3966, -1.0166, -0.2620,  0.8249]],
              [[ 0.0193,  0.2537, -0.1915,  0.4631,  0.8307,  0.0843, -1.1208,  1.1814,0.3648,  0.7760],
              [ 0.0086,  0.6027, -0.3056,  1.2362, -0.2237,  0.0752, -0.1180,  1.1989, -0.6186,  0.5418]],
              [[-0.2110,  0.5981, -0.2653,  0.8711,  0.5563,  0.3553, -0.5755,  0.7600, -0.2186,  0.6029],
              [ 0.3113,  0.3593, -0.3073,  0.7527,  0.2458, -0.2163, -0.6943,  1.5707, 0.0453,  0.7562]],
              [[ 0.0695,  0.6485, -0.2518,  0.5851,  0.2107,  0.3691, -0.6426,  1.1622, 0.0828,  0.6352],
              [ 0.0185,  0.3866, -0.2758,  1.0274,  0.4827, -0.3040, -0.6358,  1.0679, -0.2265,  0.6724]],
              [[-0.1283, -0.0252, -0.1765,  0.2010,  0.6573, -0.1419, -1.4649,  0.8111, 0.5515,  0.8129],
              [ 0.3090,  0.2555, -0.3872,  0.8164,  0.1631, -0.5320,  0.0907,  1.0127, 0.1752, -0.3050]],
              [[ 0.2819, -0.0247,  0.1623,  1.6479,  0.1429, -0.4952, -0.2141,  1.2859, -0.1462,  0.6330],
              [-0.5672,  0.8263,  0.0564,  1.2683,  0.5838, -0.0593, -0.3519,  1.5972, -0.8540,  0.7530]],
              [[-0.5855,  0.2144, -0.8822,  0.3097,  0.2327,  0.3258, -1.0506,  1.0320, 0.4563,  0.9979],
              [-0.0488,  0.6220,  0.1875,  0.7308, -0.1109,  0.4877, -0.5141,  0.6752, 0.2541,  0.9662]],
              [[ 0.2654,  1.3532, -1.1583,  0.7904,  0.4025,  0.6587, -0.8657,  0.3796, -0.9395,  0.2809], 
              [ 0.1246,  0.6793, -0.3502,  1.7736,  0.2940,  0.1363, -0.7927,  0.9134, -0.1122,  0.3432]]]
    metric = AccMeter()
    for i in range(len(y_pred)):
        metric.update(y_pred=torch.tensor(y_pred[i]), y_true=torch.tensor(y_true[i]))

if __name__ == "__main__":  
    unittest.main()