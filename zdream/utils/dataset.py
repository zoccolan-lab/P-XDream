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
            #transforms.Resize((256, 256)),
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

    def class_to_lbl(self, lbl : int) -> str:
        # Takes in input the labels and outputs their categories
            return self.label_dict[self.classes[lbl]]

    def __getitem__(self, index: int) -> Dict[str, Tensor | int]: #if want to correct type error put  Tuple[Any, Any]
        img, lbl = super().__getitem__(index)
        return {
            'imgs' : img,
            'lbls' : lbl,
        }


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
        
def resize_images(input_dir: str, output_dir:str | None = None, target_size: tuple=(256, 256)) -> str:
    """resizes tiny-imagenet images to the target_size and stores them 
    in a new folder output_dir
    
    :param input_dir: directory where tiny-imagenet images are stored
    :type input_dir: str
    :param output_dir: directory where tiny-imagenet images are stored, defaults to None
    :type output_dir: str | None, optional
    :param target_size: desired size (h x w) of the images, defaults to (256, 256)
    :type target_size: tuple, optional
    :return the main directory where resized images are located 
    :rtype: str
    """
    #if output_dir is defined, a default dir _resized is assigned
    if output_dir is None:
        fp_parts = input_dir.split(os.sep); fp_parts[-1] = fp_parts[-1]+"_resized"
        output_dir = os.sep.join(fp_parts)
        
    # Check if output directory already exists. If True, terminate
    if os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' already exists.")
        return output_dir
    
    #iterate for all the subfolders in input_dir
    for cl in os.listdir(input_dir):
        if '.' not in cl: #to avoid files present in input_dir
            cat_dir = os.path.join(input_dir, cl)
            print(cl)
            # Create output directory for class cl if it doesn't exist
            cat_dir_resize = os.path.join(output_dir, cl)
            if not os.path.exists(cat_dir_resize):
                os.makedirs(cat_dir_resize)
            # Iterate through each file in each directory of input_dir
            for filename in os.listdir(cat_dir):
                filepath = os.path.join(cat_dir, filename)
                if os.path.isfile(filepath):
                    try:
                        with Image.open(filepath) as img:
                            resized_img = img.resize(target_size)
                            output_filepath = os.path.join(cat_dir_resize, filename)
                            resized_img.save(output_filepath) # Save the resized image
                    except Exception as e:
                        print(f"Error in processing {filename}: {e}")
        #the imagenet_lbls file is also present in tiny-imagenet input_dir.
        #just copy it in the output_dir
        elif cl=='imagenet_lbls.txt':
            output_filepath = os.path.join(output_dir, cl)
            shutil.copyfile(os.path.join(input_dir, cl), output_filepath)
            
    return output_dir