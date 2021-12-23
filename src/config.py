# paths
DIDA_STRING_LABELS = './data.nosync/dida/DIDA_Label.csv'
DIDA_STRING = './data.nosync/dida/DIDA_1'
DIDA_SINGLE = './data.nosync/dida/250000_Final'
MNIST_MIX = './data.nosync/mnist_mix/MNIST-MIX-all'
ARDIS = './data.nosync/ardis'
OUTPUT_PATH = './models'

# general setup
RANDOM_SEED = 42
IMG_SIZE = 224

# training setup
TRAINING_ARGS = {
    'batch_size': 8,
    'num_epochs' : 10,
    'learning_rate' : 0.01,
    'use_wandb' : True
}