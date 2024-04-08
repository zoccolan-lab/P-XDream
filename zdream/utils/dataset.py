import glob
import os
import shutil
import torch
import numpy as np
from os import path
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from typing import Callable, Dict, Tuple

from zdream.logger import Logger, MutedLogger
from zdream.utils.misc import default

# TODO Abstract class ExperimentDataset. With from file and get_item returning images and labels.


'''
NOTE: To download
Dataset from here: [https://www.kaggle.com/datasets/arjunashok33/miniimagenet?resource=download]
Classes from here: [https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57#file-map_clsloc-txt]
'''
class MiniImageNet(ImageFolder):

    def __init__(
        self,
        root: str,
        transform: Callable[..., Tensor] = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ]),
        target_transform=None
    ):

        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform
        )

        # Load the .txt file containing ImageNet labels (all 1000 categories)
        lbls_txt = glob.glob(path.join(root, '*.txt'))[0]

        with open(lbls_txt, "r") as f:
            lines = f.readlines()

        # Create a label dictionary
        self.label_dict = {
            line.split()[0]: line.split()[2].replace('_', ' ')
            for line in lines
        }
        
    def __str__ (self) -> str: return f'MiniImageNet[{len(self)} images]'
    def __repr__(self) -> str: return str(self)

    def class_to_lbl(self, lbl : int) -> str:
        # Takes in input the labels and outputs their categories
            return self.label_dict[self.classes[lbl]]

    def __getitem__(self, index: int) -> Dict[str, Tensor | int]: #if want to correct type error put  Tuple[Any, Any]
        img, lbl = super().__getitem__(index)
        return {
            'imgs' : img,
            'lbls' : lbl,
        }
        
    @staticmethod
    def resize_dataset(in_dir: str, out_dir: str | None = None, target_size: tuple=(256, 256), logger: Logger | None = None):
        '''
        Resizes mini-Imagenet dataset images to the target_size and stores them in a new folder.
        This is made to avoid the overhead of on-line resizing.
        
        :param in_dir: Directory where mini-Imagenet images are stored, defaults to None
        :type in_dir: str
        :param out_dir: Directory where to save the resized dataset.
                        If not given suffix `resized` is added to input directory.
        :type out_dir: str | None, optional
        :param target_size: Desired new size (H x W) of the images, defaults to (256, 256)
        :type target_size: tuple, optional
        :param logger: Optional logger to log resizing information. If non given defaults to MutedLogger.
        :type target_size: Logger | None
        '''
        
        # Logger default
        logger = default(logger, MutedLogger())
        
        # If output_dir is defined, add resized suffix
        if not out_dir:
            out_dir = f'{in_dir}_resized'
            
        # Check if output directory already exists. If True, terminate
        if os.path.exists(out_dir):
            logger.info(mess=f"Output directory '{out_dir}' already exists.")
            return
        
        # Create directory
        logger.info(f'Creating directory {out_dir}')
        os.makedirs(out_dir)
        
        # Iterate for all the subfolders in input_dir
        for item in os.listdir(in_dir):
            
            # If a file copy it
            if '.' in item:
                in_fp  = os.path.join(in_dir,  item)
                out_image_fp = os.path.join(out_dir, item)
                shutil.copyfile(in_fp, out_image_fp)
                
            # Folders
            else:
                
                logger.info(mess=f'Copying directory {item}')
                
                in_subfolder = os.path.join(in_dir, item)
                
                # Create output folder
                out_subfolder = os.path.join(out_dir, item)
                os.makedirs(out_subfolder, exist_ok=True)
                
                # Iterate through each file in each directory of input_dir
                for image in os.listdir(in_subfolder):
                    
                    in_image_fp = os.path.join(in_subfolder, image)
                    
                    # Copy image
                    if os.path.isfile(in_image_fp):
                        try:
                            with Image.open(in_image_fp) as img:
                                resized_img = img.resize(target_size)
                                out_image_fp = os.path.join(out_subfolder, image)
                                resized_img.save(out_image_fp)
                        except Exception as e:
                            logger.error(f"Failed to process {image}: {e}")


class RandomImageDataset(Dataset):
    '''
    Random image dataset to simulate natural images to be interleaved
    in the stimuli with synthetic ones.
    '''

    def __init__(self, n_img: int, img_size: Tuple[int, ...]):
        self.n_img = n_img
        self.image_size = img_size

    def __len__(self):
        return self.n_img

    def __getitem__(self, idx) -> Dict[str, Tensor]:

        # Simulate finite dataset
        if idx < 0 or idx >= len(self): raise ValueError(f"Invalid image idx: {idx} not in [0, {len(self)})")

        rand_img = torch.tensor(np.random.rand(*self.image_size), dtype=torch.float32)

        return {
            'imgs' : rand_img,
            'lbls' : torch.tensor([0]),
        }
        

