'''
This file contains the classes and methods to handle the loading of natural images to be interleaved with synthetic ones.
It also provides a dataset class to handle the loading of natural images.
'''

from __future__ import annotations

from abc import abstractmethod
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

from typing import Callable, Dict, List, Tuple, cast
from torch.utils.data import DataLoader

from zdream.utils.logger import Logger, SilentLogger
from zdream.utils.io_ import read_json, read_txt
from zdream.utils.types import Mask, Stimuli
from zdream.utils.misc import device

class NaturalStimuliLoader:
    '''
    Class responsible for loading natural images to be interleaved with synthetic ones.
    It is designed to be used in the context of an experiment where the subject is presented
    with a sequence of stimuli composed of both synthetic and natural images.
    
    The presentation is associated with a boolean mask that specifies the order of the two type
    of stimuli. The loader loads the number of images corresponding to a specific mask template.
    '''
    
    def __init__(
        self,
        dataset      : ExperimentDataset | None = None,
        batch_size   : int = 2,
        template     : List[bool] = [True],
        shuffle      : bool = True
    ) -> None:
        '''
        Initialize the natural stimuli loader.

        :param DatasetClass: Class of the dataset to use for natural images.
            In the case it is not given it has a trivial behavior of
            loading no image and generating a mask with only True values.
        :type DatasetClass: ExperimentDataset
        :param batch_size: Batch size for the natural images dataloader.
        :type batch_size: int, optional
        :param template: Mask template specifying the order of synthetic and natural images,
            defaults to [True] (only synthetic images).
        :type template: List[bool], optional
        :param shuffle: If to shuffle the mask template, defaults to True.
        :type shuffle: bool, optional
        '''
        
        # Check template to contain only one True value
        if template.count(True) != 1:
            raise ValueError('At least one synthetic image is required in the template')
        
        # Check that if the dataset is not given the template consists of only one True value
        if not dataset and template != [True]:
            raise ValueError('Natural images dataset not provided, but template contains more than one True value')

        # Saving template
        self._template = template
        self._shuffle  = shuffle
        
        # Initialize dataloader in the case the template contains at least 
        # one false value, that is natural images are expected.
        if template.count(False):
            dataset = cast(ExperimentDataset, dataset)
            self._dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
            self._reset_iter()
            
        
    def _reset_iter(self):
        ''' 
        Method for setting dataloader iterator 
        It can be used to reset the iterator when all images are loaded.
        '''
        
        self._dataloader_iter = iter(self._dataloader)
        
    def _generate_mask(self, num_gen_img: int = 1, shuffle: bool = True) -> Mask:
        ''' 
        Generate the mask by concatenating the template pattern
        as many times as the number of synthetic images.        
        '''

        bool_l   = []
        template = self._template

        for _ in range(num_gen_img):
            if shuffle:
                np.random.shuffle(template)
            bool_l.extend(template)
            
        return np.array(bool_l, dtype=np.bool_)
    
    def load_natural_images(
            self, 
            num_gen_img    : int
        ) -> Tuple[Stimuli, List[int], Mask]:
        '''
        Load natural images to be interleaved with synthetic ones.
        The number of natural images is the mask False values resulting from 
        the number of synthetic images and the mask template.
        
        The method compute a mask which is also returned, and the labels of the natural images.
        
        :param num_gen_img: Number of synthetic images in the stimuli set.
        :type num_gen_img: int
        :return: Tuple containing three elements:
            - Stimuli: Tensor containing the natural images.
            - List[int]: List of labels associated to the natural images.
            - Mask: Boolean mask indicating the order of synthetic and natural images.
        :rtype: Tuple[Stimuli, Mask, List[int]]
        '''
        
        # Create the mask based on the number of synthetic images
        mask = self._generate_mask(num_gen_img=num_gen_img, shuffle=self._shuffle)
        
        # Count the number of synthetic and natural images i.e. the number of False in the mask
        num_nat_img = sum(~mask)
        
        if num_nat_img > 0:

            # List were to save batches of images
            nat_img_list : List[Tensor] = []
            labels_list  : List[int] = []
            
            # We continue extracting batches of natural images
            # until the required number
            batch_size = cast(int, self._dataloader.batch_size)
        
            while len(nat_img_list) * batch_size < num_nat_img:
                
                try:
                    
                    # Extract images and labels
                    batch = next(self._dataloader_iter)
                    nat_img_list.append(batch['images'])
                    labels_list .extend(batch['labels'])
                
                except StopIteration:
                    
                    # Circular iterator: when all images are loaded, we start back again.
                    self._reset_iter()
                    
            # We combine the extracted batches in a single tensor
            # NOTE: Since we necessary extract images in batches,
            #       we can have extracted more than required, for this purpose
            #       we may need to chop out the last few to match required number
            nat_img = torch.cat(nat_img_list)[:num_nat_img].to(device)
            labels  = labels_list[:num_nat_img]
        
        # In the case of no natural images we create an empty stimuli
        else:
            nat_img = torch.tensor([], device=device)
            labels  = []
            
        return nat_img, labels, mask
    
    @staticmethod
    def interleave_gen_nat_stimuli(
        nat_img: Stimuli,
        gen_img: Stimuli,
        mask   : Mask
    ) -> Stimuli:
        '''_summary_

        :param nat_img: _description_
        :type nat_img: Stimuli
        :param gen_img: _description_
        :type gen_img: Stimuli
        :param mask: _description_
        :type mask: Mask
        :raises NotImplementedError: _description_
        :raises FileExistsError: _description_
        :raises ValueError: _description_
        :return: _description_
        :rtype: Stimuli
        '''
        
        # Extract the number of images from the stimuli batch
        num_gen_img, *gen_img_shape = gen_img.shape
        num_nat_img, *nat_img_shape = nat_img.shape
        
        # In the case of no natural images generate a dummy one
        if not len(nat_img_shape):
            nat_img_shape = gen_img_shape
            nat_img       = torch.zeros(0, *nat_img_shape, device=device)
        
        # Perform sanity checks on the mask
        if num_gen_img + num_nat_img != len(mask): raise ValueError('Number of images in stimuli and mask do not match')
        if num_gen_img   != sum( mask):            raise ValueError('Number of synthetic images in stimuli and mask do not match')
        if num_nat_img   != sum(~mask):            raise ValueError('Number of natural images in stimuli and mask do not match')
        if gen_img_shape != nat_img_shape: raise ValueError('Synthetic and natural images have different shapes')   
    
        # Interleave the images according to the mask
        mask_ten = torch.tensor(mask, device=device)
        stimuli  = torch.zeros(num_nat_img + num_gen_img, *gen_img_shape, device=device)
        stimuli[ mask_ten] = gen_img.to(device)
        stimuli[~mask_ten] = nat_img
        
        return stimuli

class ExperimentDataset(Dataset):
    '''
    Class to define a dataset of natural images consisting of visual stimuli to 
    alternate with synthetic ones during stimuli presentation to the subject.
    '''
    
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def __getitem__(self, index) -> Dict[str, Tensor | int]:
        '''
        Load a batch of images and labels from the dataset.
        The batch is composed of a dictionary with the following keys:
        - 'images': Tensor containing the images
        - 'labels': List of integers containing the labels of the images
        '''
        raise NotImplementedError(
            'Method not implemented for ExperimentDataset which is just an interface. '\
            'Use its subclasses for actual data and implementation'
        )
        
    @abstractmethod
    def class_to_lbl(self, lbl: int) -> str: pass
    ''' Converts a class index to a label. '''


'''
NOTE: To download
Dataset from here: [https://www.kaggle.com/datasets/arjunashok33/miniimagenet?resource=download]
Classes from here: [https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57#file-map_clsloc-txt]
'''
class MiniImageNet(ImageFolder, ExperimentDataset):
    '''
    Class that defines the MiniImageNet dataset, a subset of the ImageNet dataset.
    The dataset is composed of 100 classes, each containing 600 images of size 84x84.
    
    The class requires to have the dataset stored in a folder with the following structure:
    - root, containing the dataset folders
    - root/inet_labels.txt, containing the labels of the dataset
    '''
    
    LABELS_FILE = 'inet_labels.txt'

    def __init__(
        self,
        root: str,
        transform: Callable[..., Tensor] = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
        target_transform: Callable[[Tensor], Tensor] | None = None
    ):
        '''
        Initialize the MiniImageNet dataset.

        :param root: The root directory where the dataset is stored.
        :type root: str
        :param transform: The transformation to apply to the images, 
            defaults to a resize to 224x224 and conversion to tensor.
        :type transform: Callable[..., Tensor], optional
        :param target_transform: Function to transform the target, 
            defaults to None in which case the target is not transformed.
        :type target_transform: _type_, optional
        '''

        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform
        )

        # Load the .txt file containing ImageNet labels (all 1000 categories)
        labels_path  = path.join(root, self.LABELS_FILE)
        labels_lines = read_txt(file_path=labels_path)

        # Create the label dictionary
        self.labels_dict = {
            line.split()[0]: line.split()[2].replace('_', ' ')
            for line in labels_lines
        }
        
    # --- STRING REPRESENTATION ---
    
    def __str__ (self) -> str: return f'MiniImageNet[{len(self)} images]'
    ''' Returns a string representation of the dataset.'''
    
    def __repr__(self) -> str: return str(self)
    ''' Returns a string representation of the dataset.'''
    
    # --- LABELS ---
    
    def class_to_lbl(self, lbl: int) -> str: return self.labels_dict[self.classes[lbl]]
    ''' Converts a class index to a label. '''
    
    # --- LOADING ---
    
    def __getitem__(self, index: int) -> Dict[str, Tensor | int]:
        '''
        Load a batch of images and labels from the dataset.
        The batch is composed of a dictionary with the following keys

        :param index: _description_
        :type index: int
        :return: _description_
        :rtype: Dict[str, Tensor | int]
        '''
        
        img, lbl = super().__getitem__(index)
        
        return {
            'images' : img,
            'labels' : lbl,
        }
        
    # --- RESIZING ---
    
    @staticmethod
    def resize_dataset(
        in_dir: str, 
        out_dir: str | None = None, 
        target_size: tuple = (256, 256), 
        logger: Logger = SilentLogger()
    ):
        '''
        Resizes the MiniImageNet dataset to the specified target size and saves it in a new folder.
        This is done to avoid the overhead of online resizing during experiment evaluation.
        
        :param in_dir: Directory where MiniImageNet dataset is stored.
        :type in_dir: str
        :param out_dir: Directory where to save the resized dataset.
            If not provided, a suffix 'resized' is added to the input directory.
        :type out_dir: str | None, optional
        :param target_size: Desired new size (H x W) of the images. Defaults to (256, 256).
        :type target_size: tuple, optional
        :param logger: Optional logger to log resizing information. If not provided, defaults to SilentLogger.
        :type logger: Logger
        '''
        
        # If output_dir is not defined, add resized suffix
        if not out_dir:
            out_dir = f'{in_dir}_resized'
            
        # Check if output directory already exists.
        if os.path.exists(out_dir):
            err_msg = f"Output directory `{out_dir}` already exists."
            logger.error(mess=err_msg)
            raise FileExistsError(err_msg)
        
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
        '''
        Initialize the random image dataset.

        :param n_img: Number of random images to generate.
        :type n_img: int
        :param img_size: Size of the random images to generate.
        :type img_size: Tuple[int, ...]
        '''
        
        self.n_img      = n_img
        self.image_size = img_size

    def __len__(self): return self.n_img
    ''' Returns the number of random images in the dataset. '''
    
    def class_to_lbl(self, lbl: int) -> str: return 'Random'
        

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        '''
        Load a random image from the dataset.

        :param idx: Index of the image to load.
        :type idx: int
        :return: Random image and a 0 label.
        :rtype: Dict[str, Tensor]
        '''

        # Simulate finite dataset
        if idx < 0 or idx >= len(self): raise ValueError(f"Invalid image idx: {idx} not in [0, {len(self)})")

        rand_img = torch.tensor(np.random.rand(*self.image_size), dtype=torch.float32)

        return {
            'images' : rand_img,
            'labels' : torch.tensor([0]),
        }


