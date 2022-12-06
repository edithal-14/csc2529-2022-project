import config.unetConfig as cfg
import nibabel as nib
import torchvision.utils as vutils

from PIL import Image
from torch.utils.data import Dataset

import numpy as np

class BraTSDataset(Dataset):
    def __init__(self, image_paths, transforms=None) -> None:
        # store the image and mask filepaths
        # store the transforms
        self.imagePaths = image_paths
        self.transforms = transforms

    def __len__(self):
        # return the total number of samples
        return len(self.imagePaths)

    def __getitem__(self, index):
        # grab the image path from current index
        imagePath = self.imagePaths[index]

        # load the Nifti1 image and its mask from disk
        nii_image = nib.load(imagePath)
        # Get numpy array from Nifti1 and select the 77th slice (the middle slice)
        # Total slices = 155
        image = nii_image.get_fdata()[:,:,77]
        image = np.uint8(image/image.max()*255)
        image = Image.fromarray(image)

        # check if we need to apply transformations
        if self.transforms is not None:
            # apply the transforms
            image = self.transforms(image)

        # return a tuple of the image and its mask
        return image

    def save(self, store_path):
        for i, impath in enumerate(self.imagePaths):
            # load the Nifti1 image and its mask from disk
            nii_image = nib.load(impath)
            # Get numpy array from Nifti1 and select the 77th slice (the middle slice)
            # Total slices = 155
            image = nii_image.get_fdata()[:,:,77]
            image = np.uint8(image/image.max()*255)
            image = Image.fromarray(image)

            # check if we need to apply transformations
            if self.transforms is not None:
                # apply the transforms
                image = self.transforms(image)

            vutils.save_image(image, f'{store_path}/{i}.png')