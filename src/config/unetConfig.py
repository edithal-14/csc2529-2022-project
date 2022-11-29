import torch
import os

############################################################
##################### DIRECTORIES ##########################
############################################################
# Source dir
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Base path of the dataset
DATASET_PATH = os.path.join(SRC_PATH, "data/MICCAI_BraTS2020/train")

# Define the path to the images 
CLASSES = ["flair", "seg", "t1ce", "t1", "t2"]
FLAIR_DATASET_PATH = os.path.join(DATASET_PATH, "flair")
SEG_DATASET_PATH = os.path.join(DATASET_PATH, "seg")
T1CE_DATASET_PATH = os.path.join(DATASET_PATH, "t1ce")
T1_DATASET_PATH = os.path.join(DATASET_PATH, "t1")
T2_DATASET_PATH = os.path.join(DATASET_PATH, "t2")

# Define the path to the base output directory
BASE_OUTPUT = os.path.join(SRC_PATH, "output")

# Define the path to the output serialized model, model training
# Plot, and testing image paths
GNet_PATH = os.path.join(BASE_OUTPUT, "gnet.pth")
DNet_PATH = os.path.join(BASE_OUTPUT, "dnet.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

############################################################
####################### HARDWARE ###########################
############################################################

# Determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# Define worker pool for dataloading
NUM_WORKERS = 10

############################################################
################### GEN ARCH (DCGAN) #######################
############################################################

# Size of latent vector z (input to generator)
LATENT_SZ = 100
# Size of feature maps for generator
NGF = 64
# Number of channels in the output
NGC = 1

# Define learning rate, beta1
GLR = 2e-4
GBETA1 = 0.5


############################################################
################# DISC TRAINING (UNET) #####################
############################################################

# Define the number of channels in the input, size of feature maps
# number of classes and number of levels in the U-Net model
NDC = 1
NDF = 64
NUM_CLASSES = 1
NUM_LEVELS = 4

# Define learning rate, beta1
DLR = 2e-4
DBETA1 = 0.5

# Define negative slope for LeakyReLU activation
NEG_SLOPE = 0.2

############################################################
################# COMMON TRAIN PARAMS ######################
############################################################

# Initialize umber of epochs to train for, and batch size
NUM_EPOCHS = 50
BATCH_SIZE = 16

############################################################
################### IMAGE TRANSFORMS #######################
############################################################

# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128